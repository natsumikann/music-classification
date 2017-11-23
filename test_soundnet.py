import argparse
import multiprocessing
import matplotlib

matplotlib.use('Agg')
import chainer
from Regressor import SoundNet5Layer
from Dataset import Dataset


def main():
    parser = argparse.ArgumentParser(description='Practice: SoundNet5Layer')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of songs in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='result/sound_net_10',
                        help='Path to the model')
    parser.add_argument('--dataset', '-d', default='/music/test',
                        help='Directory for train sound_net')
    parser.add_argument('--labels', type=int, default=2)
    parser.add_argument('--demo', type=bool, default=False)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    model = SoundNet5Layer(args.labels)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Load the soundnet
    # trainとvalに分ける
    test = Dataset(args.dataset)
    print('test data : {}'.format(len(test)))

    test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize, n_processes=multiprocessing.cpu_count() - 1,
                                                  repeat=False, shuffle=False)

    correct_cnt = 0
    while True:
        try:
            batch = test_iter.next()
        except StopIteration:
            break
        images = model.xp.array([image for image, _ in batch])
        labels = model.xp.array([label for _, label in batch])
        with chainer.using_config('train', False):
            predicts = model.predict(images)
        for i, l, p in zip(images, labels, predicts):
            if l == p:
                correct_cnt += 1
            if args.demo == True:
                print("label:", l, "predict:", p)

    if args.demo == False:
        print('accuracy : {}'.format(correct_cnt/len(test)))


if __name__ == '__main__':
    main()
