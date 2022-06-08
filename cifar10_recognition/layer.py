import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # input = u - Vth, if input > 0, output 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        fu = torch.tanh(input)
        fu = 1 - torch.mul(fu, fu)

        return grad_input * fu

spikeplus = STEFunction.apply


class MemoryUnit(nn.Module):
    def __init__(self):
        """
        The base class of memory cells in the network,
        of which the reset method needs to be overridden so that the network resets its state variables
        (such as U and s in the cell, mask in the dropout layer) at the beginning of each simulation.
        """
        super(MemoryUnit, self).__init__()

    def reset(self):
        raise NotImplementedError


class Dropout(MemoryUnit):
    def __init__(self, p):
        """
        The implementation of Dropout in SNN,
        which has almost the same function as torch.nn.Dropout,
        but keeping the mask unchanged in each timestep and updating it manually.
        :param p: probability of an element to be zeroed
        """
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def reset(self):
        self.mask = None

    def _generate_mask(self, input):
        self.mask = F.dropout(torch.ones_like(input.data), self.p, training=True)

    def forward(self, input):
        if self.training:
            if self.mask is None:
                self._generate_mask(input)
            out = torch.mul(input, self.mask)
        else:
            out = input
        return out


class Dropout2d(Dropout):
    def __init__(self, p):
        """
        Same as Dropout implementation in SNN,
        but in the form of 2D.
        :param p: probability of an element to be zeroed
        """
        super().__init__(p)

    def _generate_mask(self, input):
        self.mask = F.dropout2d(torch.ones_like(input.data), self.p, training=True)


class HIFICell(MemoryUnit):
    def __init__(self, weight_size, attr_size, activation=spikeplus):
        """
        The implementation of self-inhibition neuron model,
        in linear architecture or convolutional architecture.
        :param weight_size: size of synapse weight
        :param attr_size: size of neuron attribute
        :param activation: simulate pulse function with surrogate gradient
        """
        super(HIFICell, self).__init__()
        self._g = nn.LeakyReLU(0.1)
        self._activation = activation

        # define inner weight trainable parameter
        self.weight = nn.Parameter(torch.FloatTensor(weight_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        # define outer cell attribute trainable parameter
        self.tau = nn.Parameter(torch.FloatTensor(attr_size))
        nn.init.constant_(self.tau, 0.2)
        self.C = nn.Parameter(torch.FloatTensor(attr_size))
        nn.init.constant_(self.C, 0.9)
        self.gamma = nn.Parameter(torch.FloatTensor(attr_size))
        nn.init.constant_(self.gamma, 0.05)
        self.U_rest = nn.Parameter(torch.FloatTensor(attr_size))
        nn.init.constant_(self.U_rest, 0.0)
        self.U_th = nn.Parameter(torch.FloatTensor(attr_size))
        nn.init.constant_(self.U_th, -1.0)

        self.nonattr = nn.ParameterList([self.weight])
        self.attr = nn.ParameterList([self.tau, self.C, self.gamma, self.U_rest, self.U_th])

        # define the state variable
        self.U = deepcopy(self.U_rest.data).cuda()
        self.s = torch.FloatTensor([0.]).cuda()

    def reset(self):
        self.U = deepcopy(self.U_rest.data).cuda()
        self.s = torch.FloatTensor([0.]).cuda()

    def ops(self, input: torch.FloatTensor):
        return input

    def forward(self, input: torch.FloatTensor):
        I = self.ops(input)

        self.U = torch.mul(self.U, 1. - self.s) + torch.mul(self.U_rest, self.s)  # reset
        self.U = torch.mul(self.U, 1. - self.tau)  # decay

        I = torch.add(I, - torch.mul(self.gamma, self.s))
        I = torch.mul(I, self.C)

        self.U = torch.add(self.U, I)
        self.U = torch.add(self.U, torch.mul(self.tau, self.U_rest))

        # apply the nonlinear function
        self.U = self._g(self.U)

        self.s = self._activation(self.U + self.U_th)  # calculate the spike
        return self.s


class Linear_HIFICell(HIFICell):
    def __init__(self, input_size, hidden_size):
        super(Linear_HIFICell, self).__init__(
            weight_size=torch.Size([input_size, hidden_size]),
            attr_size=torch.Size([hidden_size])
        )

    def ops(self, input: torch.FloatTensor):
        return torch.matmul(input, self.weight)


class Output_HIFICell(Linear_HIFICell):
    def __init__(self, input_size, hidden_size):
        super(Output_HIFICell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(self, input: torch.FloatTensor):
        I = self.ops(input)

        self.U = torch.mul(self.U, 1. - self.tau)  # decay
        I = torch.mul(I, self.C)

        self.U = torch.add(self.U, I)
        self.U = torch.add(self.U, torch.mul(self.tau, self.U_rest))

        # apply the nonlinear function
        self.U = self._g(self.U)

        return self.U


class Conv_HIFICell(HIFICell):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Conv_HIFICell, self).__init__(
            weight_size=torch.Size([out_channel, in_channel, kernel, kernel]),
            attr_size=torch.Size([out_channel, 1, 1]),
        )
        self.stride = stride
        self.padding = padding
        self.bn = nn.BatchNorm2d(out_channel)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        for param in self.bn.parameters():
            self.nonattr.append(param)

    def ops(self, input: torch.FloatTensor):
        return self.bn(F.conv2d(input, self.weight, stride=self.stride, padding=self.padding))

