from layer import *


class HIFINet(nn.Module):
    def __init__(self, conv_size, fc_size, pooling_pos, criterion=nn.CrossEntropyLoss()):
        """
        The implementation of Heterogeneous spiking Framework with self-inhibiting neurons model (HIFI).
        :param conv_size: shape of convolution layers
        :param fc_size: shape of full-connected layers
        :param pooling_pos: position of pooling layers (default: 2*2 average pooling)
        :param criterion: criterion that measures the accuracy
        """
        super(HIFINet, self).__init__()
        # arch parameters
        self.conv_num = len(conv_size)
        self.fc_num = len(fc_size) - 1
        self.criterion = criterion

        self.attributes = nn.ParameterList()
        self.weights = nn.ParameterList()

        # conv layers
        self.features = nn.Sequential()
        self.features_weight = nn.ParameterList()
        self.features_attr = nn.ParameterList()
        for i in range(self.conv_num):
            in_channel, out_channel, kernel, stride, padding = eval(conv_size[i])
            layer = Conv_HIFICell(in_channel, out_channel, kernel, stride, padding)
            self.features.add_module('conv{}'.format(i), layer)

            self.features_weight.append(layer.weight)
            self.features_attr.extend(layer.attr)

            self.weights.extend(layer.nonattr)
            self.attributes.extend(layer.attr)

            if i in pooling_pos:
                # default pooling: average pooling
                self.features.add_module('avgpool{}'.format(i), nn.AvgPool2d(2, 2))

            if i < self.conv_num - 1:
                self.features.add_module('dropout2d'.format(i), Dropout2d(0.2))

        self.dropout = Dropout(p=0.5)

        # fc layers
        self.classifier = nn.Sequential()
        self.classifier_weight = nn.ParameterList()
        self.classifier_attr = nn.ParameterList()
        for i in range(self.fc_num):
            input_size = fc_size[i]
            hidden_size = fc_size[i + 1]
            if i < self.fc_num - 1:
                layer = Linear_HIFICell(input_size, hidden_size)
            else:
                layer = Output_HIFICell(input_size, hidden_size)

            self.classifier.add_module('linear{}'.format(i), layer)

            self.classifier_weight.append(layer.weight)
            self.classifier_attr.extend(layer.attr)

            self.weights.extend(layer.nonattr)
            self.attributes.extend(layer.attr)

    def _reset(self):
        for layer in self.modules():
            if isinstance(layer, MemoryUnit):
                layer.reset()

    def forward(self, input: torch.FloatTensor, timesteps: int = 5):

        self._reset()

        outputs = []
        for t in range(timesteps):
            x = input
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
            outputs.append(x.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1).contiguous()
        spiking_rate = torch.mean(outputs, dim=1)

        return spiking_rate

    def attributes_clamp(self):
        for layer in self.modules():
            if isinstance(layer, HIFICell):
                layer.tau.data.clamp_(0.2 - 0.1, 0.2 + 0.1)
                layer.C.data.clamp_(0.9 - 0.1, 0.9 + 0.1)
                layer.gamma.data.clamp_(0.05 - 0.025, 0.05 + 0.025)
                layer.U_rest.data.clamp_(0.0 - 0.5, 0.0 + 0.5)
                layer.U_th.data.clamp_(-1.0 - 0.5, -1.0 + 0.5)

    def accuracy(self, input: torch.FloatTensor, target: torch.FloatTensor):
        return torch.mean(torch.eq(target, torch.argmax(input, 1)).float())

    def inner_loss(self, input: torch.FloatTensor, target: torch.FloatTensor):
        return self.criterion(input, target)

    def outer_loss(self, input: torch.FloatTensor, target: torch.FloatTensor):
        return torch.add(self.criterion(input, target), 1e-2 * self.smoothness())

    def smoothness(self):
        # for features conv layers
        weight = [param.sum(axis=[2, 3]).transpose(0, 1) for param in self.features_weight[1:]]
        attr = [param.view(-1).unsqueeze(1) for param in self.features_attr]
        attr = [torch.cat(attr[i: i + 5], dim=1) for i in range(0, len(attr), 5)]

        loss_features = 0
        for i in range(len(weight)):
            loss_features += laplacian_smoothness(weight[i], attr[i:i + 2])

        # for classifier linear layers
        weight = [param for param in self.classifier_weight[1:]]
        attr = [param.unsqueeze(1) for param in self.classifier_attr]
        attr = [torch.cat(attr[i: i + 5], dim=1) for i in range(0, len(attr), 5)]
        loss_classifier = 0
        for i in range(len(weight)):
            loss_classifier += laplacian_smoothness(weight[i], attr[i:i + 2])

        loss = loss_features + loss_classifier
        return loss


def laplacian_smoothness(weight, attr_group):
    """
    The implementation of Laplacian smoothness regularization in HIFI SNN.
    :param weight: list of synapse weights
    :param attr_group: list of neuron attributes
    :return: Laplacian smoothness of the given 2 layers
    """
    neuron_num = weight.size()[0] + weight.size()[1]

    # construct Matrix L, size of [neuron_num, neuron_num]
    A = torch.zeros((neuron_num, neuron_num)).cuda()

    w = torch.abs(weight).cuda()
    lay_f, lay_l = w.size()
    A[:lay_f, lay_f:(lay_f + lay_l)] = w
    A[lay_f:(lay_f + lay_l), :lay_f] = w.t()

    I = torch.eye(A.size()[0]).cuda()
    A_rsqrt = torch.rsqrt(torch.sum(A, dim=1))
    A_rsqrt = torch.where(torch.isinf(A_rsqrt), torch.full_like(A_rsqrt, 0), A_rsqrt)
    D = torch.diag(A_rsqrt).cuda()
    norm_A = torch.matmul(D, A)
    norm_A = torch.matmul(norm_A, D)
    L = I - norm_A

    # construct attribute matrix, size of (neuron_num, 5)
    X = torch.cat(attr_group, dim=0)

    smoothness = torch.matmul(X.t(), L)
    smoothness = torch.matmul(smoothness, X)
    smoothness = torch.sum(torch.diagonal(smoothness)) * 0.5

    smoothness = smoothness / neuron_num
    return smoothness
