import os
import sys
import yaml
import torch
import argparse
import utils
import torch.nn as nn
import torch.utils
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from network import HIFINet

parser = argparse.ArgumentParser("CIFAR10")
parser.add_argument('--net',            type=str,       default='alexnet',              help='architecture of network')
parser.add_argument('--data',           type=str,       default='./data',               help='location of the data corpus')
parser.add_argument('--batch_size',     type=int,       default=32,                     help='batch size')
parser.add_argument('--report_freq',    type=float,     default=50,                     help='report frequency')
parser.add_argument('--gpu',            type=int,       default=0,                      help='gpu device id')
parser.add_argument('--seed',           type=int,       default=0,                      help='random seed')
parser.add_argument('--Ns',             type=int,       default=5,                      help='number of time_steps')
parser.add_argument('--model_path',     type=str,       default='./record/alexnet.pt',  help='number of time_steps')
parser.add_argument('--comments',       type=str,       default='debug',                help='comments of exp')

args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not torch.cuda.is_available():
        sys.exit(1)
    torch.cuda.set_device(0)

    print("load model:")
    arch = yaml.safe_load(open('arch.yaml', mode='r', encoding='utf-8'))
    criterion = nn.CrossEntropyLoss().cuda()
    model = HIFINet(
        conv_size=arch[args.net]['conv_size'],
        fc_size=arch[args.net]['fc_size'],
        pooling_pos=arch[args.net]['pooling_pos'],
        criterion=criterion,
    ).cuda()
    utils.load(model, args.model_path)

    print("param size = %fMB" % utils.count_parameters_in_MB(model.parameters()))

    print("loading DataSet:")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dst = CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    test_queue = DataLoader(
        test_dst,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    with torch.no_grad():
        infer(test_queue, model)


def infer(test_queue, model):
    obj_acc = utils.AverageMeter()
    model.eval()
    for step, (input, target) in enumerate(test_queue):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        output = model(input, args.Ns)
        accuracy = model.accuracy(output, target)
        n = input.size(0)
        obj_acc.update(accuracy.item(), n)

    print('test_acc %f' % obj_acc.avg)

if __name__ == '__main__':
    main()
