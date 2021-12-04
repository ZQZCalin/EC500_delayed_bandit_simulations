from math import log, sqrt
import numpy as np
from numpy import random
import scipy.optimize
from scipy.special import lambertw
from scipy.optimize import LinearConstraint, Bounds
import matplotlib.pyplot as plt 
from regularizers import Zimmert_Seldin_regularizer as reg

def trust_constr_over_simplex(func, d, args=()):
    '''
    use scipy.optmize.minimize with method="trust-constr"

    Parameters
    ----------
    - func:
        objective function
    - d: int,
        dimension of the domain of objective
    - args: tuple,
        extra arguments of objective
    '''

    x_0 = np.ones(d) / d    # uniform
    simplex_bounds = Bounds(np.zeros(d), np.ones(d))                    # Bounds(lb, ub)
    simplex_constr = LinearConstraint( np.ones(d)[None,:], [1], [1] )   # LC(A, lb, ub)

    res = scipy.optimize.minimize(func, x_0, args=args, method="trust-constr", bounds=simplex_bounds, constraints=simplex_constr)

    return res.x, res.optimality


def optimize_mixed_entropy(L_obs, t, eta_inv):
    '''
    optimize loss + mixed-entropy <L,x> + Zimmert_Seldin_regularizer(x)

    Parameters
    ----------
    - L_obs: ndarray (d, ),
    - t: int,
        current round number
    - eta_inv: float,
        inverse of learning rate
    '''

    lambda_opt = optimize_V_shaped(reduced_KKT_equation, epsilon=1E-4, start=0, end=100, args=(L_obs, t, eta_inv))
    x_opt = find_x_from_lambda(lambda_opt, L_obs, t, eta_inv)

    return x_opt, lambda_opt

def lambert_z(lambda_, L, t, eta_inv):
    '''
    output z = w * exp(w) in the optimization problem
    '''

    z = np.exp( 1/2 + (L + 2*lambda_) / (2*eta_inv) + log(sqrt(t) / (2*eta_inv)) )
    
    return z

def reduced_KKT_equation(lambda_, L, t, eta_inv):
    '''
    helper function for optimize_mixed_entropy
    output sum( t/4 eta_inv^2 W^-2 ) - 1
    '''

    z = lambert_z(lambda_, L, t, eta_inv)
    LHS = t / (2*eta_inv)**2 * np.sum( lambertw(z)**(-2) ) - 1

    return np.absolute(LHS)

def find_x_from_lambda(lambda_, L, t, eta_inv):
    '''
    output optimal x of KKT condition from optimal lambda
    '''

    z = lambert_z(lambda_, L, t, eta_inv)
    x = t / (2*eta_inv)**2 * lambertw(z)**(-2)

    return np.absolute(x)

def optimize_V_shaped_v1(func, epsilon, start=0, end=100, args=()):
    '''
    optimize a V-shaped function and return its minimum

    idea: recursively check in n intervals,
    if func(k) < func(k+1), then narrow down the range to (k-1, k+1)
    '''
    if end - start <= epsilon/10:
        return (start + end) / 2

    n = 10
    delta = (end - start) / n

    left = func(start, *args)

    for i in range(1, n):
        right = func(start + i*delta, *args)
        if right > left:
            return optimize_V_shaped(func, epsilon, start=start + max(0, i-2)*delta, end=start + i*delta, args=args)
        else:
            left = right        # move to right


def optimize_V_shaped(func, epsilon, start=0, end=100, args=()):
    '''
    optimize a V-shaped function and return its minimum

    idea: recursively check the middle point whether it's increasing (go to left) or decreasing (go to right)
    '''
    if end - start <= epsilon/10:
        return (start + end) / 2

    middle = (start + end) / 2

    if func(middle, *args) < func(middle+epsilon, *args):
        # increasing at middle: go left
        end = middle
    else:
        # decreasing at middle: go right
        start = middle

    return optimize_V_shaped(func, epsilon, start=start, end=end, args=args)


def runif_in_simplex(n):
    ''' 
    Return uniformly random vector in the n-simplex 
    see post: https://stackoverflow.com/questions/65154622/sample-uniformly-at-random-from-a-simplex-in-python
    '''

    k = np.random.exponential(scale=1.0, size=n)
    return k / sum(k)


def target(x, L, t, eta_inv):
    return reg(x, t, eta_inv) + np.dot(L, x)


def test_optimizer(L, t, eta_inv):
    '''
    Test the correctness of the optimizer
    '''
    
    d = len(L)
    ld = np.arange(0, 100, 0.01)
    w = [np.absolute(reduced_KKT_equation(l, L, t, eta_inv)) for l in ld]

    plt.plot(ld, w)
    plt.ylim(0,10)
    plt.show()

    x, l = optimize_mixed_entropy(L, t, eta_inv)
    cost = target(x, L, t, eta_inv)
    print(f"optimal x is {x}, optimal lambda is {l}, optimal cost is {cost}")

    s = 0
    for _ in range(1000):
        randx = runif_in_simplex(d)
        cost_randx = target(randx, L, t, eta_inv)
        if cost_randx < cost:
            print(f"find a better distribution: {randx}; cost: {cost_randx}")
            s += 1
    
    if s == 0:
        print("with high probability, the solution is optimal!")


# test
if __name__ == "__main__":
    d = 5
    # L = np.random.multivariate_normal(np.ones(d), 100*np.identity(d))
    L = np.array([-20, 4, 31, 5])
    t = 100
    eta_inv = 0.2
    
    test_optimizer(L, t, eta_inv)