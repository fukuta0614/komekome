# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os, sys, time
import copy
import datetime
from PIL import Image

import chainer
from chainer import cuda, serializers, iterators
from chainer.training import updaters
from chainer.dataset import convert

import logger
import debugger

from models import cnn, googlenet, resnet50, vgg
from dataset import KomeDataset


archs = {
    'texturecnn': cnn.TextureCNN(density=2, channel=3),
    'googlenet': googlenet.GoogLeNet,
    'resnet': resnet50.ResNet,
    'vgg': vgg.VGG16
}

init_path = {
    'googlenet': 'googlenet.npz',
    'resnet': 'ResNet-50-model.npz',
    'vgg': 'VGG_ILSVRC_16_layers.npz'
}

MODEL_PATH = '/data/unagi0/fukuta/chainer_model'


def progress_report(count, start_time, batchsize, whole_sample):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} / {} samples) time: {} ({:.2f} samples/sec)'.format(
            count, count * batchsize, whole_sample, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput
        )
    )


def evaluate(model, it, device):
    """
    evaluation
    """
    test_loss = 0
    test_accuracy = 0
    for batch in it:
        x, t = convert.concat_examples(batch, device)
        test_loss += model(x, t) * len(batch)
        test_accuracy += model.accuracy * len(batch)

    logger.plot('test loss', test_loss / len(it.dataset))
    logger.plot('test accuracy', test_accuracy / len(it.dataset))


def main():
    parser = argparse.ArgumentParser(description='Adult contents save the world')
    parser.add_argument("out")
    parser.add_argument('--init', '-i', default=None,
                        help='Initialize the model from given file')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--iterations', default=10 ** 5, type=int,
                        help='number of iterations to learn')
    parser.add_argument('--train_path', '-tr', default='../datasets/train_dataset.npy', type=str)
    parser.add_argument('--test_path', '-ta', default='../datasets/test_dataset.npy', type=str)
    parser.add_argument('--interval', default=1000, type=int,
                        help='number of iterations to evaluate')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='learning minibatch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--color_aug', dest='color_aug', action='store_true', default=False)
    parser.add_argument('--divide_type', type=str, default='normal')
    parser.add_argument('--loaderjob', type=int, default=8)
    parser.add_argument('--opt', default='adam', choices=['adam', 'momentum'])
    args = parser.parse_args()

    device = args.gpu

    # my logger
    logger.init(args)

    # load data
    train_data = np.load(args.train_path)
    test_data = np.load(args.test_path)

    # prepare dataset and iterator
    train = KomeDataset(train_data, random=True, color_augmentation=args.color_aug, divide_type=args.divide_type)
    train_iter = iterators.MultiprocessIterator(train, args.batch_size)

    test = KomeDataset(test_data, random=False, color_augmentation=False, divide_type=args.divide_type)
    test_iter = iterators.MultiprocessIterator(test, args.batch_size, repeat=False, shuffle=False)

    # model construct
    model = archs[args.arch](texture=args.texture, cbp=args.cbp, normalize=args.normalize)
    if args.finetune:
        model.load_pretrained(os.path.join(MODEL_PATH, init_path[args.arch]), num_class=2)
    else:
        model.convert_to_finetune_model(num_class=2)

    # optimizer
    if args.opt == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
    elif args.opt == 'momentum':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr)
    else:
        raise ValueError('invalid argument')
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

    cuda.get_device_from_id(device).use()
    model.to_gpu()

    # test only mode
    if args.test:
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            evaluate(model, test_iter, device)
        logger.flush()
        exit()

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)

    # start training
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    while updater.iteration < args.iterations:

        # train
        updater.update()
        progress_report(updater.iteration, start, args.batch_size, len(train))
        train_loss += model.loss.data
        train_accuracy += model.accuracy.data

        # evaluation
        if updater.iteration % args.interval == 0:
            logger.plot('train loss', cuda.to_cpu(train_loss) / args.interval)
            logger.plot('train accuracy', cuda.to_cpu(train_accuracy) / args.interval)
            train_loss = 0
            train_accuracy = 0

            # test
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                it = copy.copy(test_iter)
                evaluate(model, it, device)

            # logger
            logger.flush()

            # save
            serializers.save_npz(os.path.join(logger.out_dir, 'resume'), updater)
            serializers.save_hdf5(os.path.join(logger.out_dir, "models", "cnn_{}.model".format(updater.iteration)),
                                  model)

            if updater.iteration % 10000 == 0:
                if args.opt == 'adam':
                    optimizer.alpha *= 0.5
                else:
                    optimizer.lr *= 0.5


if __name__ == '__main__':
    main()
