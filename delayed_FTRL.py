from operator import xor
import numpy as np
from math import sqrt, log
from regularizers import Zimmert_Seldin_regularizer
from optimizers import trust_constr_over_simplex

def delayed_bandit_FTRL(expert_loss, delay, tuning="simple"):
  '''
  Run FTRL in delayed bandit setting.

  Parameters
  ----------
  - expert_loss: ndarray (T, d)
      2D array, expert_loss[t, :] is the loss of each expert in round t
  - delay: array-like,
      array of delays
  - tuning: str,
      if "simple", use simple tuning; if "advanced", use advanced tuning

  Returns
  -------
  '''

  T, d = expert_loss.shape

  ### pre-process delays
  delayed_rounds = np.arange(1, T+1, dtype=int) + delay
  delayed_rounds_sorted = np.sort(delayed_rounds)
  delayed_rounds_sort_index = np.argsort(delayed_rounds)
  '''
  at most t-1 past losses are observed at the beginning of round t, so assuming all delays >= 0,
  if 
  '''

  ### initialize
  sum_expert_loss_est = np.zeros(d)    # sum of EXPERT loss estimators: sum( ell_{s,As}/x_{s,As} * e_{As} )
  total_missing_obs = 0       # for simple tuning

  action_history = []         # list of past action history A_s
  x_history = []              # list of x_{s,A_s}

  cum_loss_FTRL = 0           # cumulative FTRL loss
  avg_cum_loss_FTRL = []      # list of average cumulative FTRL loss

  for t in range(1, T+1):     # 1-indexing

    print(t)
    # simple tuning of eta_t
    total_missing_obs += np.sum( delayed_rounds[:t-1] >= t )
    eta_inv = sqrt( 2*total_missing_obs / log(d) )

    # update x_t = argmin F_t(x), where F_t(x) = <L^, x> + Regularizer(x)
    F_t = lambda x : np.dot(sum_expert_loss_est, x) + Zimmert_Seldin_regularizer(x, t, eta_inv)
    x, _ = trust_constr_over_simplex(F_t, d)

    # sample action A_t
    action = np.random.choice(d, p=x)
    action_history.append(action)
    x_history.append(x[action])

    # update cumulative loss
    cum_loss_FTRL += np.dot(expert_loss[t-1, :], x)     # t-1 because of 1-indexing
    avg_cum_loss_FTRL.append(cum_loss_FTRL / t)

    # update expert loss estimator
    receiving_from = np.where(delayed_rounds == t)[0]   # 0-indexing in this loop
    for s in receiving_from:
      A_s = action_history[s]
      sum_expert_loss_est[A_s] += expert_loss[s, A_s] / x_history[s]

    # verbose
    if True and t % (T//100) == 0:
      print( "progress: %d/100" % (t//(T//100)) )

    # test purpose
    print(x, action, sum_expert_loss_est)

  return avg_cum_loss_FTRL