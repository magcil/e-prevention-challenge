import numpy as np
import scipy

# read mse_train.npy and mse_test.npy:
mse_train = np.load('mse_train.npy')
mse_val_0 = np.load('mse_val_0.npy')
mse_val_1 = np.load('mse_val_1.npy')

h0, b0 = np.histogram(mse_val_0, bins=10)
h1, b1 = np.histogram(mse_val_1, bins=10)

b0 = (b0[:-1] + b0[1:]) / 2
b1 = (b1[:-1] + b1[1:]) / 2

m = np.mean(mse_train)
s = np.std(mse_train)

p0 = [1 - scipy.stats.norm(m, s).pdf(b) / scipy.stats.norm(m, s).pdf(m) for b in mse_val_0]
p1 = [1 - scipy.stats.norm(m, s).pdf(b) / scipy.stats.norm(m, s).pdf(m) for b in mse_val_1]

ps = np.concatenate((p0, p1))
ps_random = np.random.uniform(0, 1, len(ps))
ys = np.concatenate((np.zeros(len(p0)), np.ones(len(p1))))
for i in range(len(ps)):
    print(ps[i], ys[i])
# compute AUC:
from sklearn.metrics import roc_auc_score
print(np.mean(p0), np.mean(p1))
print(ps_random)
print(f'AUC: {roc_auc_score(ys, ps)}')
print(f'AUC random: {roc_auc_score(ys, ps_random)}')


# use plotly to plot the h0 and h1:
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=b0, y=h0, name='val_0'))
fig.add_trace(go.Scatter(x=b1, y=h1, name='val_1'))
fig.update_layout(title='MSE distribution of val_0 and val_1', xaxis_title='MSE', yaxis_title='count')
fig.show()

# show ps and ps_random histograms:
fig = go.Figure()
h0, b0 = np.histogram(p0, bins=10)
h1, b1 = np.histogram(p1, bins=10)

b0 = (b0[:-1] + b0[1:]) / 2
b1 = (b1[:-1] + b1[1:]) / 2
fig = go.Figure()
fig.add_trace(go.Scatter(x=b0, y=h0, name='val_0'))
fig.add_trace(go.Scatter(x=b1, y=h1, name='val_1'))
fig.update_layout(title='MSE distribution of val_0 and val_1', xaxis_title='MSE', yaxis_title='count')
fig.show()