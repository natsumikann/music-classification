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
            self.fc1 = L.Linear(1, 128)
            self.lstm2 = L.LSTM(128, 128)
            self.fc3 = L.Linear(128, out_num)

    def __call__(self, sound, t=None):
        h = F.relu(self.fc1(sound))
        h = F.relu(self.lstm2(h))
        h = F.softmax(self.fc3(h))

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