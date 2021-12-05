import numpy as np
import argparse
import re
import os, shutil
from math import sqrt, log
from cvxopt import solvers, matrix, spdiag, mul as cvx_mul, log as cvx_log, sqrt as cvx_sqrt
from util import plot_avg_cum_loss, make_dir


class Model:
    '''
    Model for delayed MAB algorithm
    '''
    def __init__(self, K: int, reg: str="mixed_entropy", tune: str="simple", delay: str="zero", 
    name: str="experiment", result_dir: str="result") -> None:
        # parameters
        self.K = K  # number of arms
        self.reg = reg
        self.tune = tune
        self.delay = delay
        if not reg in ["negative_entropy", "Tsallis_entropy", "mixed_entropy", "no_name"]:
            raise("invalid regularizer")
        if not tune in ["simple", "advanced", "increasing"]:
            raise("invalid tuning method")
        # hidden
        self.actions = []       # history of actions
        self.weights = []       # history of x_{s, A_s}
        self.cum_loss = 0
        self.avg_cum_loss = []
        self.L_obs = np.zeros(self.K)   # unbiased loss estimator
        # check result directory
        self.result_dir = result_dir
        self.name = name
        self.save()

    def reset(self) -> None:
        '''
        Reset hidden variables for training
        '''
        self.actions = []
        self.weights = []
        self.cum_loss = 0
        self.avg_cum_loss = []
        self.L_obs = np.zeros(self.K)

    def optimize(self, L_obs, t, eta_inv) -> np.array:
        '''
        Optimize x_{t+1} = argmin <L_obs, x_t> + Reg(x_t) 
            Refer to https://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives for details
        '''

        G = -matrix(np.identity(self.K))
        h = matrix(0.0, (self.K, 1))
        A = matrix(1.0, (1, self.K))
        b = matrix(1.0, (1, 1))

        def F(x=None, z=None):
            if x == None:
                return 0, matrix(1.0/self.K, (self.K,1))
            if min(x) <= 0.0:
                return None
            # compute f, Df
            if self.reg == "negative_entropy":
                f = sqrt(t) * sum(cvx_mul(x, cvx_log(x)))
                Df = sqrt(t) * ( cvx_log(x) + matrix(1.0, (self.K,1)) ).T
            elif self.reg == "Tsallis_entropy":
                f = -sqrt(t) * 2 * sum(cvx_sqrt(x))
                Df = -sqrt(t) * (x**(-1/2)).T
            elif self.reg == "mixed_entropy":
                f = eta_inv * sum(cvx_mul(x, cvx_log(x))) + ( -sqrt(t) * 2 * sum(cvx_sqrt(x)) )
                Df = eta_inv * ( cvx_log(x) + matrix(1.0, (self.K,1)) ).T + ( -sqrt(t) * (x**(-1/2)).T )
            elif self.reg == "no_name":
                f = -sqrt(t) * sum(cvx_mul(x, 1-x)**1/2)
                Df = -sqrt(t) * cvx_mul(1-2*x, 1/2*(x-x**2)**(-1/2)).T
            # add <L_obs, x> to regularizer
            L_obs_mat = matrix(L_obs, (1, self.K))
            f += L_obs_mat * x
            Df += L_obs_mat
            if z == None:
                return f, Df
            # compute Hessian (Hessian of linear term is 0)
            if self.reg == "negative_entropy":
                H = z[0] * sqrt(t) * spdiag( x**-1 )
            elif self.reg == "Tsallis_entropy":
                H = z[0] * sqrt(t) * spdiag( 1/2 * x**(-3/2) )
            elif self.reg == "mixed_entropy":
                H = z[0] * ( eta_inv * spdiag( x**-1 ) + sqrt(t) * spdiag( 1/2 * x**(-3/2) ) )
            elif self.reg == "no_name":
                H = z[0] * sqrt(t) * spdiag( 1/4 * (x-x**2)**(-3/2) )
            return f, Df, H
        
        solvers.options['show_progress'] = False
        soln = np.array(solvers.cp(F, G=G, h=h, A=A, b=b)['x'])
        # need to do this slight normalization
        return np.array(soln).ravel() / sum(soln)

    def generate_delay(self, size) -> np.array:
        '''
        Generate sequence of delays
        '''
        if re.findall("^uniform", self.delay):
            _, a, b = self.delay.split("_")
            delays = np.random.uniform(low=float(a), high=float(b), size=(size,))
        elif re.findall("^Gaussian", self.delay):
            _, mu, sigma = self.delay.split("_")
            delays = np.random.normal(loc=float(mu), scale=float(sigma), size=(size,))
            delays = np.maximum(delays, np.zeros(size))
        else:
            delays = np.zeros(size)

        return delays.astype(int)

    def train(self, losses, verbose=False) -> None:
        '''
        Run delayed FTRL
        '''
        # initialize hidden variables
        self.reset()

        T = len(losses)

        # generate delays
        delayed_rounds = np.arange(1, T+1, dtype=int) + self.generate_delay(size=T)

        # simple tuning
        total_missing_obs = 0

        for t in range(1, T+1):
            # set learning rate
            if self.tune == "simple":
                total_missing_obs += np.sum( delayed_rounds[:t-1] >= t )
                eta_inv = sqrt( 2* total_missing_obs / log(self.K) )
            elif self.tune == "advanced":
                eta_inv = 0
            elif self.tune == "increasing":
                eta_inv = 1 / sqrt(t * log(self.K))

            # update x_t = argmin <L_obs, x> + Reg(x)
            x = self.optimize(self.L_obs, t, eta_inv)

            # sample action A_t
            action = np.random.choice(self.K, p=x)
            self.actions.append(action)
            self.weights.append(x[action])

            # update cumulative loss
            self.cum_loss += np.dot(losses[t-1, :], x)     # t-1 because of 1-indexing
            self.avg_cum_loss.append(self.cum_loss / t)

            # update expert loss estimator
            receiving_from = np.where(delayed_rounds == t)[0]   # 0-indexing in this loop
            for s in receiving_from:
                A_s = self.actions[s]
                self.L_obs[A_s] += losses[s, A_s] / self.weights[s]

            # verbose
            if verbose and t % (T//100) == 0:
                print( "progress: %d/100" % (t//(T//100)) )
    
    def save(self) -> None:
        '''
        Create result directory
        '''
        # make directory
        make_dir(f"{self.result_dir}/{self.name}")
        # save experiment config
        config = {
            "name": self.name,
            "reg": self.reg,
            "tune": self.tune,
            "delay": self.delay,
            "losses": "Local OSD on time series",
            "other notes": ""
        }
        with open(f"{self.result_dir}/{self.name}/config.txt", "w") as f:
            f.writelines( '\n'.join([ f"{key}: {value}" for (key, value) in config.items() ]) )

    def evaluate(self) -> None:
        '''
        Evaluate the model and save experiment results
        '''
        # save average cum loss
        np.savetxt(f"{self.result_dir}/{self.name}/avg_cum_loss.txt", self.avg_cum_loss)
        # save action history
        np.savetxt(f"{self.result_dir}/{self.name}/action_history.txt", self.actions)
        # plot and save average cum loss
        plot_avg_cum_loss(self.avg_cum_loss, save_to=f"{self.result_dir}/{self.name}/avg_cum_loss.png")


def run_experiment(args):
    # Load OSD losses
    K = 7
    print("------------loading data------------")
    OSD_loss = np.concatenate([np.loadtxt(
        "%s/alpha=1E%d.txt" % (args.data_dir, i))[:, None] for i in range(-3, 4)], axis=1)   # (T,d)

    T = len(OSD_loss)

    # rewrite directory
    dir = f"{args.result_dir}/{args.name}"
    if args.rewrite and os.path.isdir(dir):
        shutil.rmtree(dir)

    model = Model(K, reg=args.regularizer, tune=args.tuning, delay=args.delay, name=args.name, result_dir=args.result_dir)
    print("------------training------------")
    model.train(losses=OSD_loss, verbose=True)
    print("------------saving model------------")
    model.evaluate()

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run delayed MAB algorithm")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Local losses folder path")
    parser.add_argument("--regularizer", type=str, default="mixed_entropy",
                        help="Regularizer: mixed_entropy, negative_entropy or Tsallis_entropy")
    parser.add_argument("--tuning", type=str, default="simple",
                        help="Tuning method: simple, or advanced")
    parser.add_argument("--delay", type=str, default="zero", 
                        help="Communication delay distribution: zero, uniform_a_b, or Gaussian_mean_std")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Experiment result folder path")
    parser.add_argument("--name", type=str, default="experiment",
                        help="Experiment name")
    parser.add_argument("--rewrite", type=bool, default=False,
                        help="Rewrite existing experiment result")
    args = parser.parse_args()

    model = run_experiment(args)