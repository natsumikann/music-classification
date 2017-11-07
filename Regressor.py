import chainer
from chainer import functions as F
from chainer import links as L
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.pooling.average_pooling_nd import average_pooling_nd

from chainer.functions.array.reshape import reshape

class RegressorOutputValues():
    SoundOutput = 64

# 音楽用
class SoundNet5Layer(chainer.Chain):
    SoundOutput = RegressorOutputValues.SoundOutput

    def __init__(self):
        super(SoundNet5Layer, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, (1, 64), 2, (0, 32))
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(32, 64, (1, 32), 2, (0, 16))
            self.bn2 = L.BatchNormalization(64)
            self.conv3 = L.Convolution2D(64, 128, (1, 16), 2, (0, 8))
            self.bn3 = L.BatchNormalization(128)
            self.conv4 = L.Convolution2D(128, 256, (1, 8), 2, (0, 4))
            self.bn4 = L.BatchNormalization(256)
            self.conv5 = L.Convolution2D(256, self.SoundOutput, (1, 16),
                                         12, (0, 4))
            self.bn5 = L.BatchNormalization(self.SoundOutput) #２次元で出力 Linear 第一引数 none

    def __call__(self, sound):
        h = F.relu(self.bn1(self.conv1(sound)))
        h = F.max_pooling_2d(h, (1, 8), 8, 0)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, (1, 8), 8, 0)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pooling_2d(h, (1, 8), 8, 0)
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = self._global_average_pooling_nd(h)
        return h

    def _global_average_pooling_nd(self, x):
        n, channel, rows, cols = x.data.shape
        h = average_pooling_2d(x=x, ksize=(rows, cols), stride=1)
        h = reshape(h, (n, channel))
        return h

class SoundNet5LayerTrainer(chainer.Chain):
    def __init__(self, sound_net_5_layer, out_num):
        super().__init__()
        with self.init_scope():
            self.model = sound_net_5_layer
            self.fc1 = L.Linear(RegressorOutputValues.SoundOutput, 100)
            self.fc2 = L.Linear(100, out_num)

    def __call__(self, x, t):
        h = F.relu(self.fc1(self.model(x)))
        h = F.softmax(self.fc2(h))

        t = self.xp.asarray(t, self.xp.int32)
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)
        if chainer.config.train:
            return loss
        return h
