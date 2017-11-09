import numpy as np
import chainer
from scipy.io import wavfile
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

SAMPLE_RATE = 16000
TAG_FIELD = 'genre'

class Dataset(chainer.dataset.DatasetMixin):
    SOUND_SHAPE = (1, 1, 5 * SAMPLE_RATE) #5 seconds

    def __init__(self, root_dir, debug, print_name=False):
        self._paths = glob(os.path.join(root_dir, '*/*/*.flac'))
        self.tag_dict = read_csv()
        self.labels = {file: self.tag_dict[FLAC(file)[TAG_FIELD]]
                       for i, file in enumerate(self._paths)}
        self.debug = debug
        self.print_name = print_name


    def __len__(self):
        return len(self._paths)


    def get_example(self, i):
        path = self._paths[i]
        if self.print_name:
            print(path)
        samplerate, raw_sound = sf.read(path[i])
        raw_sound = raw_sound.T
        raw_sound = librosa.to_mono(raw_sound)
        raw_sound = librosa.resample(raw_sound, samplerate, SAMPLE_RATE)
        max_volume = np.max(raw_sound) * (1 + random.random())
        raw_sound = raw_sound.astype(np.float32).reshape((1, 1, -1)) / max_volume
        raw_sound_range = len(raw_sound[0][0]) - self.SOUND_SHAPE[-1]
        raw_sound = raw_sound[0][0][random.randint(0, raw_sound_range):]
        # plt.plot(raw_sound.flatten())
        # plt.savefig('raw.png')
        # plt.close()
        _slice = [slice(0, np.min((x, y,)), None) for x, y in zip(raw_sound.shape, self.SOUND_SHAPE)]
        sound = np.zeros(self.SOUND_SHAPE, dtype=np.float32)
        sound[_slice] = raw_sound[_slice]

        sound += np.random.random((self.SOUND_SHAPE)) / 10  # 1*1*5*44100
        return sound, self.get_label_from_path(path)

    def get_label_from_path(self, file):   #いらない
        return self.labels[file]


class ValModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        with chainer.using_config('train', False):
           loss = super().evaluate()
        return loss
