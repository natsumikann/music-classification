import chainer
from chainer.links import VGG16Layers
import argparse
import numpy as np
from PIL import Image

MODEL = VGG16Layers()


def search(src, db_features, db_paths, k, gpu):
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        MODEL.to_gpu()
        db_features = chainer.cuda.to_gpu(db_features)
    # src_df(深層特徴)の計算
    with chainer.using_config('train', False):
        src_df = MODEL.extract([Image.open(src, 'r').convert('RGB')], ['fc6'], (224, 224))['fc6'].data
        print(src_df.shape)
    # sklearnならcdistが速い
    distances = np.array(
        [chainer.cuda.to_cpu(
            MODEL.xp.linalg.norm(src_df - target_df)
        ) for target_df in db_features]
    )
    ranking = np.argsort(distances)
    top_k = ranking[0:k]
    return top_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Practice: search')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input', '-i', default='mini_cifar/test/cat/cat_s_000019.png',
                        help='path to database features')
    parser.add_argument('--features', '-f', default='db/features.npy',
                        help='path to database features')
    parser.add_argument('--paths', '-p', default='db/paths.npy',
                        help='path to database paths')
    parser.add_argument('--k', '-k', type=int, default=5,
                        help='find num')
    args = parser.parse_args()

    db_features = np.load(args.features)
    db_paths = np.load(args.paths)

    assert args.k <= len(db_paths)
    assert len(db_features) == len(db_paths)

    hits = search(args.input, db_features, db_paths, args.k, args.gpu)
    for i in hits:
        print(db_paths[i])
