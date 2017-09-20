import os
import sys
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.externals import joblib

from chainer import cuda
import chainer
import chainer.functions as F
from chainer.links.model.vision.vgg import VGG16Layers
import copy

shared = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared')
sys.path.append(shared)

from utils.dataset import KomeDataset
from utils.functions import gram_matrix, compact_bilinear_pooling
from utils import debugger

parser = argparse.ArgumentParser(description='extract patch from TCGA data')
parser.add_argument('--gpu', '-g', type=int, default=0)
args = parser.parse_args()

base_dir = './'

vgg16 = VGG16Layers()
cuda.get_device_from_id(args.gpu).use()
vgg16.to_gpu()

cbp_sizes = [256, 1024, 4096]
randweight_dict = {}
for cbp_size in cbp_sizes:
    randweight = np.load(os.path.join(shared, 'cbp/randweight_256_to_{}.npz'.format(cbp_size)))
    randweight_dict[cbp_size] = {'W1': cuda.to_gpu(randweight['W1']), 'W2': cuda.to_gpu(randweight['W2'])}

batch_size = 256
image_size = 256
vggl = 'pool3'
norm_type = 'powerl2'
cbp_size = 1024

auth = [(os.path.join(base_dir, 'auth', img), 1)
        for img in os.listdir(os.path.join(base_dir, 'auth'))]

fake = [(os.path.join(base_dir, 'fake', img), 0)
        for img in os.listdir(os.path.join(base_dir, 'fake'))]

dataset = auth + fake

all_data = KomeDataset(dataset, original_size=image_size, crop_size=224, random=False, color_augmentation=False,
                       rotate=False, nocrop=True)
all_iter = chainer.iterators.MultiprocessIterator(all_data, batch_size, shuffle=False, repeat=False)

feat = np.zeros((len(dataset), cbp_size), dtype='float32')
count = 0
for i, batch in enumerate(all_iter):
    with chainer.no_backprop_mode():
        x = chainer.Variable(vgg16.xp.asarray(batch, 'float32'))
        vgg_feat = vgg16(x, layers=[vggl])[vggl]
        representations = cuda.to_cpu(compact_bilinear_pooling(vgg_feat, randweight_dict[cbp_size]).data)
        representations = normalize(np.sqrt(np.abs(representations)) * np.sign(representations)).astype('float32')
        feat[i * batch_size:(i + 1) * batch_size] = representations
    count += len(batch)
    print(count, '/', len(dataset))

print('np save : ', end='')
np.save(os.path.join('cbp_feat.npy'), feat)
np.save(os.path.join('dataset.npy'), dataset)
print('done')


from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score


X = feat
y = []

kf = StratifiedKFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    svm = LinearSVC()
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    print(accuracy_score(y_test, y_predict))



