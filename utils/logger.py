import numpy as np
import os
import datetime
import collections

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import chainer

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    use_tensorboard = True
except ImportError:
    print('tensorflow is not installed')
    use_tensorboard = False

out_dir = None

_since_beginning = collections.defaultdict(lambda: {})
_best_score = [0]
_best_pr_auc = [0]
_iter = [0]

# tensorboard
if use_tensorboard:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf_array = tf.placeholder(tf.float32)
    summaries = {}
    for x in ['train_accuracy', 'test_accuracy', 'test_auc', 'test_pr_auc', 'recall']:
        summaries[x] = tf.summary.scalar(x, tf_array)
    summary_writer = None


def init(args):
    global out_dir
    if args.resume is not None:
        out = args.resume
    else:
        out = datetime.datetime.now().strftime('%m%d%H') + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.log_dir, out))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'scores'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plot'), exist_ok=True)

    # setting
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    global summary_writer
    summary_dir = os.path.join(out_dir, "summaries")
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)


def calc_recall_with_certain_precision(ans_all, pred_all, precision=0.995):
    sup = 1
    inf = 0
    count = 0
    pred_normal = np.array([pred_all[i] for i in range(len(ans_all)) if ans_all[i] == 0])
    pred_tumor = np.array([pred_all[i] for i in range(len(ans_all)) if ans_all[i] == 1])
    while abs(sup - inf) > 1e-3:
        count += 1
        x = (sup + inf) / 2
        normal_acc = 1 - np.mean(pred_normal > x)
        if normal_acc > precision:
            sup = x
        else:
            inf = x
    return np.mean(pred_tumor > sup)


def plot_score(pred_all, ans_all, model):
    # accuracy
    test_accuracy = np.mean((1 * (np.array(pred_all) > 0.5)) == ans_all)
    plot('test_accuracy', test_accuracy)

    # calc_recall_with_certain_precision
    recall = calc_recall_with_certain_precision(ans_all, pred_all)
    plot('recall', recall)

    # plot roc curve
    lw = 2
    roc_auc = roc_auc_score(ans_all, pred_all)
    # print('test_auc', roc_auc)
    plot('test_auc', roc_auc)

    if roc_auc > _best_score[0]:
        plt.figure()

        _best_score[0] = roc_auc
        fpr, tpr, thresholds = roc_curve(ans_all, pred_all)

        plt.plot(fpr, tpr, lw=lw, label='ROC curve (AUC=%0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        # plt.savefig(os.path.join(out_dir, 'roc_{:03d}.png'.format(_iter[0] + 1)))
        plt.savefig(os.path.join(out_dir, 'roc_test.png'))

        np.save(os.path.join(out_dir, "plot", "pred_{:03d}.npy".format(_iter[0] + 1)), pred_all)
        np.save(os.path.join(out_dir, "plot", "ans_{:03d}.npy".format(_iter[0] + 1)), ans_all)

        chainer.serializers.save_npz(os.path.join(out_dir, "models", "cnn_{:03d}.model".format(_iter[0] + 1)), model)

    # plot pr curve
    pr_auc = average_precision_score(ans_all, pred_all)
    # print('test_pr_auc', pr_auc)

    plot('test_pr_auc', pr_auc)

    if pr_auc > _best_pr_auc[0]:
        plt.figure()

        _best_pr_auc[0] = pr_auc
        precision, recall, thresholds = precision_recall_curve(ans_all, pred_all)

        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall, precision, lw=lw, color='navy',
                 label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(pr_auc))
        plt.legend(loc="lower left")
        # plt.savefig(os.path.join(out_dir, 'pr_roc_{:03d}.png'.format(_iter[0] + 1)))
        plt.savefig(os.path.join(out_dir, 'pr_roc_test.png'))

        np.save(os.path.join(out_dir, "scores", "pred_{:03d}.npy".format(_iter[0] + 1)), pred_all)
        np.save(os.path.join(out_dir, "scores", "ans_{:03d}.npy".format(_iter[0] + 1)), ans_all)

        chainer.serializers.save_npz(os.path.join(out_dir, "models", "cnn_{:03d}.model".format(_iter[0] + 1)),
                                     model)


def plot(name, value):
    _since_beginning[name][_iter[0]] = value

    if use_tensorboard:
        if name in summaries:
            summary = sess.run(summaries[name], feed_dict={tf_array: value})
            summary_writer.add_summary(summary, _iter[0])


def flush():
    log = ''
    log += "epoch {}\n".format(_iter[0])
    for name, vals in sorted(_since_beginning.items()):

        if hasattr(vals[_iter[0]], '__iter__'):
            log += ' {}\t{}\n'.format(name, ['{:.3f}'.format(x) for x in vals[_iter[0]]])
            continue

        log += " {}\t{:.5f}\n".format(name, vals[_iter[0]])
        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(out_dir, 'plot', name.replace(' ', '_') + '.jpg'))

    print(log)
    with open(os.path.join(out_dir, 'log'), 'a+') as f:
        f.write(log + '\n')

    _iter[0] += 1
