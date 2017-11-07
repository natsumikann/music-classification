import chainer
from chainer.links import VGG16Layers
from PIL import Image
import argparse
import numpy as np

MODEL = VGG16Layers()


def _create_db(paths, gpu):
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        MODEL.to_gpu()
    with chainer.using_config('train', False):
        features = np.asarray(
            [chainer.cuda.to_cpu(
                MODEL.extract(
                    [Image.open(path, 'r').convert('RGB')],
                    ['fc6'],
                    size=(224, 224))['fc6'].data
            )
                for path in paths],
            np.float32
        )
    print('dataset size : {}'.format(len(features)))
    return features


def create_db(dir, gpu):
    from glob import glob
    import os
    paths = glob(os.path.join(dir, '*/*.png'))
    assert len(paths) != 0
    if not os.path.exists('db'):
        os.mkdir('db')
    np.save('db/features.npy', _create_db(paths, gpu))
    np.save('db/paths.npy', paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Practice: search')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default='mini_cifar/train',
                        help='Directory for train mini_cifar')
    args = parser.parse_args()
    create_db(args.dataset, args.gpu)
