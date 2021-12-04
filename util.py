import os
import numpy as np
from matplotlib import pyplot as plt

def check_exists(path):
  '''
  check if there is a file on the path, raise an error if True
  '''
  if os.path.exists(path):
    raise TypeError("file already exists")

def make_dir(dir):
  if os.path.isdir(dir):
    raise TypeError("directory already exists")
  os.mkdir(dir)
  # try:
  #   if os.path.isdir(dir):
  #     raise TypeError("directory already exists")
  #   os.mkdir(dir)
  # except TypeError as e:
  #   print("make directory failed: "+str(e))

def plot_avg_cum_loss(avg_cum_loss, start=0, title=None, save_to=None):
  '''
  plots the average cumulative loss against time

  input:
    - loss: array-like
      array of average cumulative losses
    - start: int
      plot starts from t = start (0-indexing)
    - title: str
      if not None, set title
    - save_to: str
      if not None, save figure to the path
  '''

  T = len(avg_cum_loss)
  plt.plot( np.arange(start+1, T+1), avg_cum_loss[start:] )
  # styling
  plt.xlabel("Round number")
  plt.ylabel("Average cumulative loss")
  if title:
    plt.title(title)
  # save figure
  if save_to:
    try:
      check_exists(save_to)
      plt.savefig(save_to)
      plt.clf()
    except TypeError as e:
      print("save figure failed: "+str(e))
  else:
    plt.show()

  return None 

def save_loss(loss, fname):
  '''
  Save sequence of loss history into .txt file

  Input
  -----
  - loss: array-like
    array of loss history
  - fname: str
    path name to be saved
  '''
  
  try:
    check_exists(fname)
    np.savetxt(fname, loss)
  except TypeError as e:
    print("save loss failed: "+str(e))

def argmax_below_k(arr, k):
  '''
  Returns i such that arr[i] <=k and arr[i+1] > k

  Parameters
  ----------
  - arr: ndarray,
      an ascendingly sorted array
  - k: float,
      threshold of argmax
  '''

  # ugly O(n) implementation
  for i in range(len(arr)):
    if arr[i] > k:
      return i-1
  
  return len(arr)-1

  # to do: binary search implementation