import numpy as np
import argparse
from util import plot_avg_cum_loss, save_loss, make_dir


def OSD(T, x_init, alpha, loss_grad, time_series):
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

    for t in range(1, T+1):  # indexing from 1 to make things easier
        # receive loss and subgradient
        ell_t, g_t = loss_grad(t, x, time_series)
        # OSD update
        eta_t = alpha / t
        x -= eta_t * g_t
        # pay loss and record loss
        cum_loss += ell_t
        avg_cum_loss.append(cum_loss / t)
        loss_history.append(ell_t)

    return avg_cum_loss, loss_history


def loss_grad_time_series(t, x_t, time_series):
    '''
    Compute loss and gradient 
        Given a time series data {z_1, ..., z_T}, compute
        - ell_t(x_t) = |<w_t, x_t>-z_t|, where w_t = (z_{t-10}, ..., z_{t-1})
        - g_t(x_t) = sign( <w_t, x_t>-z_t ) * w_t

        input:
        - t: int, current round (indexing from 1)
        - x_t: ndarray (10, ), current state
        return:
        - loss: float, ell_t(x_t)
        - grad: ndarray (10, ), gradient of ell_t at x_t
    '''

    t_ = t - 1  # 0 indexing
    w_t = time_series[t_: t_+10]
    z_t = time_series[t_+10]

    temp = np.dot(w_t, x_t) - z_t
    loss = abs(temp)
    grad = np.sign(temp) * w_t

    return loss, grad


def generate_data(args):
    '''
    Generate OSD data
    '''

    # Load data
    time_series = np.loadtxt(args.time_series)  # ndarray (T,)
    T = len(time_series)                         # 1E6
    # pad first 10 time series with 0
    time_series = np.concatenate(
        [np.zeros(10), time_series])  # ndarray (T+10,)

    # Save plots and losses
    make_dir(args.data_dir)

    # Perform OSD
    x_init = np.ones(10) / 10     # uniform weight

    for i in range(-3, 4):
        alpha = 10**i
        avg_cum_loss, loss_history = OSD(
            T, x_init, alpha, loss_grad_time_series, time_series)
        # plot performance
        plot_avg_cum_loss(avg_cum_loss, start=100,
                          title="alpha=1E%d" % (i), save_to="%s/alpha=1E%d.png" % (args.data_dir, i))
        # save loss history
        save_loss(loss_history, fname="%s/alpha=1E%d.txt" % (args.data_dir, i))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate local OSD data")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Generated data folder path")
    parser.add_argument("--time_series", type=str,
                        default="time-series.txt", help="Time series path")
    args = parser.parse_args()

    generate_data(args)