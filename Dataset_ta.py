# coding=utf-8
import numpy as np
import chainer
import multiprocessing
from chainer.training import extensions
from tag_dict import read_csv
from glob import glob
from mutagen.flac import FLAC
import librosa
import os
import random
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import soundfile as sf

SAMPLE_RATE = 22050
TAG_FIELD = 'genre'
GENRE_TO_VEC = {'rock': 0, 'pop': 1} # pop, rockのジャンルをラベル（ベクトル）に変換する辞書
batchsize = 30 #TODO 変更する

class Dataset(chainer.dataset.DatasetMixin):
    SOUND_LENGTH = 5
    SOUND_SHAPE = (SOUND_LENGTH * SAMPLE_RATE) #5 seconds

    def __init__(self, root_dir, debug=False, print_name=False):
        paths = glob(os.path.join(root_dir, '*/*/*.flac'))
        self.tag_dict = read_csv()
        self._paths = []
        self.labels = {}
        for i, file in enumerate(paths):
            tags = FLAC(file)
            tag = tags.get(TAG_FIELD)
            print(tag)
            if tag == None:
                # genreタグが含まれていなければ読み飛ばす
                continue
            label = self.tag_dict[tag[0]]
            # 音楽ファイルのパスとラベルをデータセットに追加
            self._paths.append(file)
            self.labels[file] = GENRE_TO_VEC[label]
        print("len of paths and len of labels")
        print(len(self._paths), len(self.labels))
        self.debug = debug
        self.print_name = print_name

    def get_example(self):
        print("in get_example")
        #multiple_paths = [self._paths[i:i+batchsize] for i in range(0, len(self._paths), batchsize)]
        #print(multiple_paths)
        multi = np.asarray(range(0, len(self._paths)))
        processes = max(1, multiprocessing.cpu_count()-1)
        p = multiprocessing.Pool(processes)
        return  p.map(self.get_example_single, multi)

    def __len__(self):
        return len(self._paths)

    def get_example_single(self, i):
        path = self._paths[i]
        if self.print_name:
            print(path)

        # 音楽ファイルを読み込みモノラル化
        raw_sound, samplerate = sf.read(path)
        raw_sound = raw_sound.T
        raw_sound = librosa.to_mono(raw_sound)

        # 音楽データSOUND_LENGTH秒分のshape
        sound_shape = self.SOUND_LENGTH * samplerate

        # 音量正規化（ランダムノイズあり）
        max_volume = np.max(raw_sound) * (1 + random.random())
        raw_sound = raw_sound.astype(np.float32) / max_volume

        # 音楽をランダムの位置からSOUND_LENGTH秒だけ切り出す
        raw_sound_range = len(raw_sound) - sound_shape
        start = random.randint(0, raw_sound_range)
        sound = raw_sound[start:start + sound_shape]
        # plt.plot(raw_sound.flatten())
        # plt.savefig('raw.png')
        # plt.close()

        # データ量を減らすためダウンサンプル
        # sound = librosa.resample(sound, samplerate, SAMPLE_RATE)
        sound = sound[::2]

        # ホワイトノイズ付加
        sound += np.random.random(self.SOUND_SHAPE) / 10

        # データ整形
        sound = sound.reshape((1, 1, -1))

        return (sound, self.get_label_from_path(path))

    def get_label_from_path(self, file):
        return self.labels[file]



class ValModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        with chainer.using_config('train', False):
           loss = super().evaluate()
        return loss
