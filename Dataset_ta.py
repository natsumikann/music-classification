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

SAMPLE_RATE = 22050
TAG_FIELD = 'genre'
GENRE_TO_VEC = {'rock': 0, 'pop': 1}

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
            if tag == None:
                continue
            label = self.tag_dict[tag[0]]
            self._paths.append(file)
            self.labels[file] = GENRE_TO_VEC[label]
        print(len(self._paths), len(self.labels))
        self.debug = debug
        self.print_name = print_name


    def __len__(self):
        return len(self._paths)


    def get_example(self, i):
        path = self._paths[i]
        if self.print_name:
            print(path)
        raw_sound, samplerate = sf.read(path)
        raw_sound = raw_sound.T
        raw_sound = librosa.to_mono(raw_sound)
        sound_shape = self.SOUND_LENGTH * samplerate
        max_volume = np.max(raw_sound) * (1 + random.random())
        raw_sound = raw_sound.astype(np.float32) / max_volume
        raw_sound_range = len(raw_sound) - sound_shape
        start = random.randint(0, raw_sound_range)
        sound = raw_sound[start:start + sound_shape]
        # plt.plot(raw_sound.flatten())
        # plt.savefig('raw.png')
        # plt.close()
        # sound = librosa.resample(sound, samplerate, SAMPLE_RATE)
        sound = sound[::2]
        sound += np.random.random(self.SOUND_SHAPE) / 10  # 1*1*5*16000
        sound = sound.reshape((1, 1, -1))
        return sound, self.get_label_from_path(path)

    def get_label_from_path(self, file):   #いらない
        return self.labels[file]


class ValModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        with chainer.using_config('train', False):
           loss = super().evaluate()
        return loss
