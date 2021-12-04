def OSD(T, x_init, alpha, loss_grad):
  '''
  Performs Online Subgradient Descent algorithm.
    
  Update
  ------
  Subgradient descent using time-varying learning rate:
        x_{t+1} = x_t - eta_t * g_t,
  where eta_t = alpha / t and alpha is hyperparameter

  Input
  -----
  - T : int,
      total number of rounds
  - x_init : ndarray (d, )
      initial value in R^d
  - alpha : float,
      parameter of learning rate
  - loss_grad: func, 
      loss_grad(t, x_t) returns: 
        loss, grad
        - loss: float, the loss ell_t(x_t)
        - grad: ndarray (d, ), the sub-gradient of ell_t at x_t

  Returns
  -------
  avg_cum_loss, loss_history

  - avg_cum_loss: list[float],
      list of average cumulative loss
  - losses: list[float],
      list of history loss
  '''
  
  x = x_init.copy()
  cum_loss = 0
  avg_cum_loss = []
  loss_history = []   # for LFT purpose only

  for t in range(1, T+1): # indexing from 1 to make things easier
    # receive loss and subgradient
    ell_t, g_t = loss_grad(t, x)
    # OSD update
    eta_t = alpha / t
    x -= eta_t * g_t
    # pay loss and record loss
    cum_loss += ell_t
    avg_cum_loss.append( cum_loss / t )
    loss_history.append( ell_t )
  
  return avg_cum_loss, loss_history