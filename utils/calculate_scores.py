import numpy as np
import torch

def calculate_stats(originals, reconstructions, criterion, set=''):

    anomaly_scores = list()
    counter = 0
    for i in range(len(originals)):
        for j in range(len(originals[i])):
            anomaly_scores.append(criterion(reconstructions[i][j], originals[i][j]).item())


    print(f'Min MSE loss on {set} data:', round(np.min(anomaly_scores), 4))
    print(f'Average MSE loss on {set} data:', round(np.mean(anomaly_scores), 4))
    print(f'Median MSE loss on {set} data:', round(np.median(anomaly_scores), 4))
    print(f'Max MSE loss on {set} data:', round(np.max(anomaly_scores), 4))
    print('----------------------------------------------------')

    return anomaly_scores