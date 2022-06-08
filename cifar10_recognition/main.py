import os
import sys
import time
import glob
import yaml

import torch
import logging
import argparse
import utils
import torch.nn as nn
import torch.utils

import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from dataset import get_dataset
from attribute import Attribute
from network import HIFINet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        type=str,       default='cifar10',      help='architecture of network')
parser.add_argument('--net',            type=str,       default='alexnet',      help='architecture of network')
parser.add_argument('--data_path',      type=str,       default='./data',       help='location of the data corpus')
parser.add_argument('--batch_size',     type=int,       default=32,             help='batch size')
parser.add_argument('--inner_lr',       type=float,     default=1e-2,           help='init inner learning rate')
parser.add_argument('--outer_lr',       type=float,     default=1e-4,           help='init outer learning rate')
parser.add_argument('--momentum',       type=float,     default=0.9,            help='momentum of SGD')
parser.add_argument('--weight_decay',   type=float,     default=3e-4,           help='weight decay')
parser.add_argument('--report_freq',    type=float,     default=50,             help='report frequency')
parser.add_argument('--gpu',            type=int,       default=0,              help='gpu device id')
parser.add_argument('--epochs',         type=int,       default=200,            help='num of training epochs')
parser.add_argument('--seed',           type=int,       default=0,              help='random seed')
parser.add_argument('--Ns',             type=int,       default=5,              help='number of time_steps')
parser.add_argument('--comments',       type=str,       default='demo',         help='comments of exp')

args = parser.parse_args()

if not os.path.exists('./record'):
    os.makedirs('./record')
save_path = './record/TRAIN-{}-{}-{}'.format(args.net, time.strftime("%Y%m%d-%H%M%S"), args.comments)
utils.create_exp_dir(save_path, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    print("load model:")
    arch = yaml.safe_load(open('arch.yaml', mode='r', encoding='utf-8'))
    criterion = nn.CrossEntropyLoss().cuda()
    model = HIFINet(
        conv_size=arch[args.net]['conv_size'],
        fc_size=arch[args.net]['fc_size'],
        pooling_pos=arch[args.net]['pooling_pos'],
        criterion=criterion,
    ).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model.parameters()))

    optimizer = torch.optim.SGD(
        model.weights,
        args.inner_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-4)

    print("loading DataSet:")

    train_queue, valid_queue, test_queue = get_dataset(args.dataset, args.data_path, args.batch_size)

    attribute = Attribute(model, args)

    max_acc = max_acc_epoch = 0

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_obj, train_acc = train(train_queue, valid_queue, model, attribute, optimizer)
        logging.info('train_acc %f', train_acc)
        scheduler.step()
        attribute.scheduler.step()

        # validation
        with torch.no_grad():
            test_obj, test_acc = infer(test_queue, model)
        logging.info('test_acc %f', test_acc)

        if test_acc > max_acc:
            max_acc = test_acc
            max_acc_epoch = epoch

            utils.save(model, os.path.join(save_path, 'weights.pt'))

        logging.info('max_acc %f epoch %d', max_acc, max_acc_epoch)


def train(train_queue, valid_queue, model, attribute, optimizer):
    obj_inner_loss = utils.AverageMeter()
    obj_outer_loss = utils.AverageMeter()
    obj_acc = utils.AverageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        outer_loss = attribute.step(input_search, target_search)

        output = model(input, timesteps=args.Ns)
        loss = model.inner_loss(output, target)
        acc = model.accuracy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        obj_inner_loss.update(loss.item(), n)
        obj_outer_loss.update(outer_loss.item(), n)
        obj_acc.update(acc.item(), n)

        if step % args.report_freq == 0 and step != 0:
            logging.info('train %03d %e %e %f', step, obj_inner_loss.avg, obj_outer_loss.avg, obj_acc.avg)

    return obj_inner_loss.avg, obj_acc.avg


def infer(valid_queue, model):
    obj_loss = utils.AverageMeter()
    obj_acc = utils.AverageMeter()

    model.eval()

    for step, (input, target) in enumerate(valid_queue):

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        output = model(input, timesteps=args.Ns)
        loss = model.inner_loss(output, target)
        acc = model.accuracy(output, target)

        n = input.size(0)

        obj_loss.update(loss.item(), n)
        obj_acc.update(acc.item(), n)

        if step % args.report_freq == 0 and step != 0:
            logging.info('test %03d %e %f', step, obj_loss.avg, obj_acc.avg)

    return obj_loss.avg, obj_acc.avg


if __name__ == '__main__':
    main()
