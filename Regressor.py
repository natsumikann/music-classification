import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainer import Variable
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.pooling.average_pooling_nd import average_pooling_nd

from chainer.functions.array.reshape import reshape

class RegressorOutputValues():
    SoundOutput = 64

# 音楽用
class SoundNet5Layer(chainer.Chain):
    SoundOutput = RegressorOutputValues.SoundOutput

    def __init__(self, out_num):
        super(SoundNet5Layer, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, 3, 2, 1)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(32, 64, 3, 2, 1)
            self.bn2 = L.BatchNormalization(64)
            self.conv3 = L.Convolution2D(64, 128, 3, 2, 1)
            self.bn3 = L.BatchNormalization(128)
            self.conv4 = L.Convolution2D(128, 256, 3, 1, 1)
            self.bn4 = L.BatchNormalization(256)
            self.conv5 = L.Convolution2D(256, self.SoundOutput, 3,
                                         1, 1)
            self.bn5 = L.BatchNormalization(self.SoundOutput) #２次元で出力 Linear 第一引数 none
            self.fc1 = L.Linear(RegressorOutputValues.SoundOutput, 100)
            self.fc2 = L.Linear(100, out_num)

    def __call__(self, sound, t=None):
        h = F.relu(self.bn1(self.conv1(sound)))
        h = F.max_pooling_2d(h, 3)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pooling_2d(h, 3)
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = self._global_average_pooling_nd(h)

        h = F.relu(self.fc1(h))
        h = F.softmax(self.fc2(h))

        if chainer.config.train:
            t = self.xp.asarray(t, self.xp.int32)
            loss = F.softmax_cross_entropy(h, t)
            accuracy = F.accuracy(h, t)
            chainer.report({'loss': loss}, self)
            chainer.report({'accuracy': accuracy}, self)
            return loss
        return h

    def _global_average_pooling_nd(self, x):
        n, channel, rows, cols = x.data.shape
        h = average_pooling_2d(x=x, ksize=(rows, cols), stride=1)
        h = reshape(h, (n, channel))
        return h

    def predict(self, x):
        return F.argmax(self(x), axis=1).data
