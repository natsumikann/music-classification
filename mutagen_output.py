# coding=utf-8
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

#tagの例を表示するためのプログラム

tags = FLAC("/music/Madonna/Confessions on a Dance Floor/03 Sorry.flac")
print(tags)