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
from sklearn.model_selection import train_test_split

import plotly
import plotly.subplots
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from utils.util_funcs import get_pos_neg_samples

FEATURE_NAMES = [
    'heartRate_nanmean', 'rRInterval_nanmean', 'rRInterval_rmssd', 'rRInterval_sdnn',
    'rRInterval_lombscargle_power_high', 'gyr_mean', 'gyr_std', 'gyr_delta_mean', 'gyr_delta_std', 'acc_mean',
    'acc_std', 'acc_delta_mean', 'acc_delta_std', 'sin_t', 'cos_t'
]


def plot_feature_histograms(list_of_feature_mtr, feature_names, class_names, n_columns=5, out_file=None):
    '''
    Plots the histograms of all classes and features for a given
    classification task.
    :param list_of_feature_mtr: list of feature matrices
                                (n_samples x n_features) for each class
    :param feature_names:       list of feature names
    :param class_names:         list of class names, for each feature matr
    '''

    if not out_file:
        out_file = "report.html"

    n_features = len(feature_names)
    n_bins = 12
    n_rows = int(n_features / n_columns) + 1
    figs = plotly.subplots.make_subplots(rows=n_rows, cols=n_columns, subplot_titles=feature_names)
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
            scatter_1 = go.Scatter(x=cbins, y=h, name=class_names[fi], marker=mark_prop, showlegend=(i == 0))
            # (show the legend only on the first line)
            figs.append_trace(scatter_1, int(i / n_columns) + 1, i % n_columns + 1)
    for i in figs['layout']['annotations']:
        i['font'] = dict(size=10, color='#224488')
    plotly.offline.plot(figs, filename=out_file, auto_open=True)


def get_color_combinations(n_classes):
    clr_map = matplotlib.colormaps['jet']
    range_cl = range(int(int(255 / n_classes) / 2), 255, int(255 / n_classes))
    clr = []
    for i in range(n_classes):
        clr.append('rgba({},{},{},{})'.format(
            clr_map(range_cl[i])[0],
            clr_map(range_cl[i])[1],
            clr_map(range_cl[i])[2],
            clr_map(range_cl[i])[3]))
    return clr


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def execute_experiment(track_id: int,
                       patient_id: int,
                       plot_histograms: bool = False,
                       show_importances: bool = False,
                       verbose: bool = False) -> Tuple[float, float]:
    """Fit an SVM classifier on 5 Minute interval and show results.
    
    Args:
        track_id (int): Track number
        patient_id (int): Patient id
        plot_histograms (bool): Whether to plot histograms and importances
        show_importances (bool): Wheter to plow feature importances
        verbose (bool): Whether to print classification results

    Returns:
        Tuple[float, float]: F1 score, Accuracy of classification
    """

    all_pos, all_neg = get_pos_neg_samples(track_id, patient_id, FEATURE_NAMES)

    if plot_histograms:
        plot_feature_histograms([all_pos, all_neg], FEATURE_NAMES, ['pos', 'neg'], 5, f"patient_0{patient_id}.html")

    if verbose:
        print(f"{8* '*'} Running Experiments for Patient {patient_id} {8*'*'}\n")
        print(f"- Positive samples: {all_pos.shape[0]} | Negative samples: {all_neg.shape[0]}")

    # Split to train/test
    X, y = np.concatenate([all_pos, all_neg], axis=0), np.concatenate(
        [np.ones(all_pos.shape[0], dtype=np.int32),
         np.zeros(all_neg.shape[0], dtype=np.int32)], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run experiment
    mean, std = X_train.mean(axis=0), np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std)
    X_test = (X_test - mean) / (std)
    cl = SVC(kernel='rbf', C=1, gamma="auto")
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)

    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
    f1 = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f1_relapses = f1_score(y_pred=y_pred, y_true=y_test, pos_label=1, average='binary')

    if verbose:
        print(f"- Confusion matrix: {cm}\n- F1: {f1:3f}\n- Acc: {acc:3f}\n- F1 (relapses only): {f1_relapses:3f}\n")

    # Plot importances
    if show_importances:
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        print(svm.coef_)
        f_importances(svm.coef_[0], FEATURE_NAMES)

    return f1, acc, f1_relapses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=int, required=True, choices=[1, 2], help='track 1 or 2')
    parser.add_argument('--patients',
                        type=int,
                        nargs='+',
                        required=True,
                        help='List of patient ids to run the experiment.')
    parser.add_argument('--plot', type=bool, default=True, help='Plot histograms and importances.')
    parser.add_argument('--importances', type=bool, default=False, help='Whether to plot feature importances.')
    parser.add_argument('--verbose', type=bool, default=True, help='Controls verbosity of script.')

    args = parser.parse_args()
    track_id = args.track
    patients = args.patients
    plot = args.plot
    importances = args.importances
    verbose = args.verbose

    all_f1s, all_accs, f1_relapses_all = [], [], []

    for patient_id in patients:
        f1, acc, f1_relapses = execute_experiment(track_id=track_id,
                                                  patient_id=patient_id,
                                                  plot_histograms=plot,
                                                  show_importances=importances,
                                                  verbose=verbose)
        all_f1s.append(f1)
        all_accs.append(acc)
        f1_relapses_all.append(f1_relapses)

    print(f"\n{3*'*'} Aggregated Results for patients: {patients} {3*'*'}\n")

    print(f"- Accuracy: Mean -> {np.mean(all_accs):3f} | Std -> {np.std(all_accs):3f}")
    print(f"- F1 Score: Mean -> {np.mean(all_f1s):3f} | Std -> {np.std(all_f1s):3f}")
    print(f"- F1 Score (relapses only): Mean -> {np.mean(f1_relapses_all):3f} | Std -> {np.std(f1_relapses_all):3f}")
