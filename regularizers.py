from scipy.stats import entropy
from scipy.special import xlogy
import numpy as np

def negative_entropy(x):
  '''
    Computes nagtive entropy of x \in R^d.

    Parameters
    ----------
    - x: ndarray (d, ),
        d dimensional vector x

    Returns
    -------
    - entropy: float,
        \sum_{i=1}^d x_i\log(x_i)
  '''
  
  # quesition: does the log-base matter?
  # return -entropy(x)
  return np.sum(xlogy(x, x))


def Tsallis_entropy(x):
  '''
    Computes Tsallis entropy of x \in R^d.

    Parameters
    ----------
    - x: ndarray (d, ),
        d dimensional vector x

    Returns
    -------
    - entropy: float,
        \sum_{i=1}^d -2 \sum_{i=1}^d \sqrt{x_i}
  '''
  
  return -2 * np.sum(np.sqrt(x))


def Zimmert_Seldin_regularizer(x, t, eta_inv):
  '''
    Computes the Zimmert Seldin regularizer (mix of two entropies).

    Parameters
    ----------
    - x: ndarray (d, ),
        input x
    - t: int,
        round number
    - eta_inv: float,
        learning rate inversed (equivalently strong convexity constant)

    Returns
    -------
    - output: float,
        \sqrt{t} E_{Tsallis}(x) + \eta_t^{-1} E_{negative}(x)
  '''
  
  return np.sqrt(t) * Tsallis_entropy(x) + eta_inv * negative_entropy(x)