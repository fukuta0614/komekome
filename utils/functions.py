import chainer.functions as F
import numpy as np


def gram_matrix(x):
    b, ch, h, w = x.data.shape
    v = F.reshape(x, (b, ch, w * h))
    return F.batch_matmul(v, v, transb=True) / np.float32(w * h)


def compact_bilinear_pooling(x, randweight):
    h = F.convolution_2d(x, randweight['W1']) * F.convolution_2d(x, randweight['W2'])
    h = global_average_pooling_2d(h)
    h = power_normalize(h)
    h = F.normalize(h)
    return h


def power_normalize(x):
    pos = F.relu(x)
    neg = F.relu(-x)
    return F.sqrt(pos) - F.sqrt(neg)


def global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, -1))
    return h
