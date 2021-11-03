# run OSD.py on a time series data {z_1, ..., z_T}
# ell_t(x_t) = |<w_t, x_t>-z_t|, where w_t = (z_{t-10}, ..., z_{t-1})
# g_t(x_t) = sign( <w_t, x_t>-z_t ) * w_t

### Define loss and gradient
# CAVEAT: note that t starts from 1 not 0

def loss_grad_time_series(t, x_t):
  '''
  input:
    - t: int, current round (indexing from 1)
    - x_t: ndarray (10, ), current state
  return:
    - loss: float, ell_t(x_t)
    - grad: ndarray (10, ), gradient of ell_t at x_t
  '''
  t_ = t - 1  # 0 indexing
  w_t = time_series[t_ : t_+10]
  z_t = time_series[t_+10]

  temp = np.dot(w_t, x_t) - z_t
  loss = abs(temp)
  grad = np.sign(temp) * w_t

  return loss, grad

### data directory
data_dir = "OSD_time_series_data"

if __name__ == "__main__":

  import numpy as np
  from matplotlib import pyplot as plt
  from OSD import OSD
  from util import plot_avg_cum_loss, save_loss, make_dir

  ### Load data
  time_series = np.loadtxt("time-series.txt")  # ndarray (T,)
  T = len(time_series)                         # 1E6
  # pad first 10 time series with 0
  time_series = np.concatenate([np.zeros(10), time_series])  # ndarray (T+10,)

  ### Save plots and losses
  make_dir(data_dir)

  ### Perform OSD
  x_init = np.ones(10) / 10     # uniform weight

  for i in range(-3, 4):
    alpha = 10**i
    avg_cum_loss, loss_history = OSD(T, x_init, alpha, loss_grad_time_series)
    # plot performance
    plot_avg_cum_loss(avg_cum_loss, start=100,
      title="alpha=1E%d"%(i), save_to="%s/alpha=1E%d.png"%(data_dir, i))
    # save loss history
    save_loss(loss_history, fname="%s/alpha=1E%d.txt"%(data_dir, i))