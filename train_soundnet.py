import matplotlib

matplotlib.use("Agg")
import argparse
import numpy as np

import chainer
from chainer import serializers
from chainer import training
from chainer.datasets import split_dataset_random
from chainer.training import extensions

from config import global_value
from Regressor import SoundNet5Layer
from Regressor import SoundNet5LayerTrainer
from Dataset import Dataset
from Dataset import ValModeEvaluator


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--labels', type=int, default=50)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--dry_run', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    # グローバル変数定義
    global_value.initialize()

    # データセットイテレーターの定義
    debug_mode = args.dry_run
    train_dir = global_value.train_dir
    train = Dataset(train_dir, debug_mode, False)
    train, val = split_dataset_random(train, 1800)
    print('train: {:d} sounds found'.format(len(train)))
    print('val: {:d} movies found'.format(len(val)))
    train_iter = chainer.iterators.MultiprocessIterator(
        train,
        args.batchsize,
        n_processes=4,
        repeat=True,
        shuffle=True
    )
    val_iter = chainer.iterators.SerialIterator(
        val,
        args.batchsize,
        repeat=False, shuffle=False
    )

    # モデルの定義
    model = SoundNet5Layer()
    model_trainer = SoundNet5LayerTrainer(model, args.labels)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model_trainer.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=1e-2)
    optimizer.setup(model_trainer)

    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (1000, 'epoch'), args.out)

    trainer.extend(extensions.ExponentialShift('lr', np.power(0.1, 1 / 30)),
                   trigger=(5, 'epoch'))
    snapshot_interval = 1, 'epoch'
    log_interval = 10, 'iteration'
    trainer.extend(
        ValModeEvaluator(val_iter, model_trainer, device=args.gpu),
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
                                  'epoch', file_name='accuracy.png'),
            trigger=snapshot_interval)
    with chainer.using_config('train', True):
        trainer.run()
