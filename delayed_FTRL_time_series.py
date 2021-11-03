import numpy as np
from OSD_time_series import data_dir
from delayed_FTRL import delayed_bandit_FTRL
from util import plot_avg_cum_loss

if __name__ == "__main__":

  ### Load OSD losses
  d = 7
  OSD_loss = np.concatenate([np.loadtxt("%s/alpha=1E%d.txt"%(data_dir, i))[:,None] for i in range(-3, 4)], axis=1)   # (T,d)

  delay = np.random.randint( 0, 50, size=(len(OSD_loss), ) )

  avg_cum_loss = delayed_bandit_FTRL(OSD_loss, delay)
  plot_avg_cum_loss(avg_cum_loss)