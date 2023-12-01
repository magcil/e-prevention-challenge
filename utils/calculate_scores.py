import numpy as np
import torch

def calculate_stats(originals, reconstructions, criterion, set=''):
    anomaly_scores = [criterion(originals[i], reconstructions[i]).item() for i in range(len(originals))]

    print('Min MSE loss on train data:', round(np.min(anomaly_scores), 4))
    print('Average MSE loss on train data:', round(np.mean(anomaly_scores), 4))
    print('Median MSE loss on train data:', round(np.median(anomaly_scores), 4))
    print('Max MSE loss on train data:', round(np.max(anomaly_scores), 4))
    print('----------------------------------------------------')

    return anomaly_scores