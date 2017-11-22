import matplotlib
matplotlib.use("Agg")
import argparse
import numpy as np
import time
import multiprocessing
import chainer
from chainer import serializers
from chainer import training
from chainer.datasets import split_dataset_random
from chainer.training import extensions

from tag_dict import count_data
from Regressor_6layers import SoundNet5Layer
from Dataset_ta_org import Dataset
from Dataset_ta_org import ValModeEvaluator


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the mini_cifar to train')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--labels', type=int, default=2)
    parser.add_argument('--out', default='result')
    parser.add_argument('--dataset', '-d', default='/music',
                        help='Directory for train sound_net')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--dry_run', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    start = time.time()
    args = parse()


    # データセットイテレーターの定義
    debug_mode = args.dry_run
    train_dir = args.dataset
    train = Dataset(train_dir, debug_mode, True)
    print(train[1])
    data_num = count_data(train_dir)
    print(data_num)
    train, val = split_dataset_random(train, int(data_num*0.8))
    print('train: {:d} sounds found'.format(len(train)))
    print('val: {:d} movies found'.format(len(val)))
    train_iter = chainer.iterators.MultiprocessIterator(
        train,
        args.batchsize,
        n_processes=multiprocessing.cpu_count() - 1,
        repeat=True,
        shuffle=True
    )
    val_iter = chainer.iterators.SerialIterator(
        val,
        args.batchsize,
        repeat=False, shuffle=False
    )

    # モデルの定義
    model = SoundNet5Layer(args.labels)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=1e-2)
    optimizer.setup(model)
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out) #num iof epoch 1 temporalily

    trainer.extend(extensions.ExponentialShift('lr', np.power(0.1, 1 / 30)),
                   trigger=(5, 'epoch'))
    snapshot_interval = 1, 'epoch'
    log_interval = 10, 'iteration'
    trainer.extend(
        ValModeEvaluator(val_iter, model, device=args.gpu),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'sound_net_{.updater.epoch}'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=snapshot_interval))
    trainer.extend(extensions.observe_lr(), trigger=snapshot_interval)
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration',
            'main/loss',
            'validation/main/loss',
            'main/accuracy',
            'validation/main/accuracy',
            'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'),
            trigger=snapshot_interval)
        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                  'epoch', file_name='accuracy_6layers_dropout.png'),
            trigger=snapshot_interval)
    with chainer.using_config('train', True):
        trainer.run()

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
