import os
import datetime
import pandas as pd
import numpy as np
import pyhrv
import scipy
import argparse
from typing import Dict, Optional, Union, Tuple
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import plotly
import plotly.subplots
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from utils.parse import parse_data, get_path, get_unique_days


def plot_feature_histograms(list_of_feature_mtr, feature_names,
                            class_names, n_columns=5):
    '''
    Plots the histograms of all classes and features for a given
    classification task.
    :param list_of_feature_mtr: list of feature matrices
                                (n_samples x n_features) for each class
    :param feature_names:       list of feature names
    :param class_names:         list of class names, for each feature matr
    '''
    n_features = len(feature_names)
    n_bins = 12
    n_rows = int(n_features / n_columns) + 1
    figs = plotly.subplots.make_subplots(rows=n_rows, cols=n_columns,
                                         subplot_titles=feature_names)
    figs['layout'].update(height=(n_rows * 250))
    clr = get_color_combinations(len(class_names))
    for i in range(n_features):
        # for each feature get its bin range (min:(max-min)/n_bins:max)
        f = np.vstack([x[:, i:i + 1] for x in list_of_feature_mtr])
        bins = np.arange(f.min(), f.max(), (f.max() - f.min()) / n_bins)
        for fi, f in enumerate(list_of_feature_mtr):
            # load the color for the current class (fi)
            mark_prop = dict(color=clr[fi], line=dict(color=clr[fi], width=3))
            # compute the histogram of the current feature (i) and normalize:
            h, _ = np.histogram(f[:, i], bins=bins)
            h = h.astype(float) / h.sum()
            cbins = (bins[0:-1] + bins[1:]) / 2
            scatter_1 = go.Scatter(x=cbins, y=h, name=class_names[fi],
                                   marker=mark_prop, showlegend=(i == 0))
            # (show the legend only on the first line)
            figs.append_trace(scatter_1, int(i/n_columns)+1, i % n_columns+1)
    for i in figs['layout']['annotations']:
        i['font'] = dict(size=10, color='#224488')
    plotly.offline.plot(figs, filename="report.html", auto_open=True)


def get_color_combinations(n_classes):
    clr_map = plt.cm.get_cmap('jet')
    range_cl = range(int(int(255/n_classes)/2), 255, int(255/n_classes))
    clr = []
    for i in range(n_classes):
        clr.append('rgba({},{},{},{})'.format(clr_map(range_cl[i])[0],
                                              clr_map(range_cl[i])[1],
                                              clr_map(range_cl[i])[2],
                                              clr_map(range_cl[i])[3]))
    return clr


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, required=True, choices=[1, 2], help='track 1 or 2')
    parser.add_argument('--patient', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='which patient')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "val"], help='which split')
    parser.add_argument('--num', type=int, required=True, help='which number of split')
    parser.add_argument('--dtype', type=str, nargs='+', default=['all'])

    args = parser.parse_args()

    all_negs = []
    all_pos = []

    all_pos_train = []
    all_pos_test = []
    all_neg_train = []
    all_neg_test = []

    n_neg = len(os.listdir('data/track_01/P1/val_0/features/'))
    n_pos = len(os.listdir('data/track_01/P1/val_1/features/'))

    n_neg_train = int(n_neg * 0.8)
    n_pos_train = int(n_pos * 0.8)

    # list all files with .parquet extension in folder:
    for i_f, f in enumerate(os.listdir('data/track_01/P1/val_0/features/')):
        neg = pd.read_parquet(os.path.join('data/track_01/P1/val_0/features/', f), engine='fastparquet')
        #neg = neg[neg.columns.difference(['cos_t', 'sin_t'])]
        neg = neg.loc[:, (neg.columns != "DateTime")]
        cols = neg.columns
        neg = np.array(neg)
        all_negs.append(neg)
        if i_f < n_neg_train:
            all_neg_train.append(neg)
        else:
            all_neg_test.append(neg)
    all_negs = np.concatenate(all_negs, axis=0)
    all_neg_train = np.concatenate(all_neg_train, axis=0)
    all_neg_test = np.concatenate(all_neg_test, axis=0)

    print(all_neg_train.shape)
    print(all_neg_test.shape)    
    print(all_negs.shape)

    # list all files with .parquet extension in folder:
    for i_f, f in enumerate(os.listdir('data/track_01/P1/val_1/features/')):
        pos = pd.read_parquet(os.path.join('data/track_01/P1/val_1/features/', f), engine='fastparquet')
        #pos = pos[pos.columns.difference(['cos_t', 'sin_t'])]
        pos = pos.loc[:, pos.columns != "DateTime"]
        pos = np.array(pos)
        all_pos.append(pos)
        if i_f < n_pos_train:
            all_pos_train.append(pos)
        else:
            all_pos_test.append(pos)
    all_pos = np.concatenate(all_pos, axis=0)
    all_pos_train = np.concatenate(all_pos_train, axis=0)
    all_pos_test = np.concatenate(all_pos_test, axis=0)


    print(all_pos_train.shape)
    print(all_pos_test.shape)    
    print(all_pos.shape)

#    pos = pd.read_parquet('data/track_01/P1/val_1/features/day_00.parquet', engine='fastparquet')
#    pos = pos.loc[:, pos.columns != "DateTime"]
#    cols = neg.columns
#    pos = np.array(pos)
#    print(pos.shape)
    plot_feature_histograms([all_negs, all_pos], cols, ["neg", "pos"], n_columns=5)


    x_train = np.concatenate((all_neg_train, all_pos_train), axis=0)
    x_test = np.concatenate((all_neg_test, all_pos_test), axis=0)
    y_train = np.concatenate((np.zeros(all_neg_train.shape[0]), np.ones(all_pos_train.shape[0])), axis=0)
    y_test = np.concatenate((np.zeros(all_neg_test.shape[0]), np.ones(all_pos_test.shape[0])), axis=0)

    mean, std = x_train.mean(axis=0), np.std(x_train, axis=0)
    x_train = (x_train - mean) / (std)  
    x_test = (x_test - mean) / (std)
    cl = SVC(kernel='rbf', C=1, gamma="auto")
    cl.fit(x_train, y_train)
    y_pred = cl.predict(x_test)

    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
    f1 = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)

    print(cm, f1)


    features_names = cols
    svm = SVC(kernel='linear')
    svm.fit(x_train, y_train)
    print(svm.coef_)
    f_importances(svm.coef_[0], features_names)