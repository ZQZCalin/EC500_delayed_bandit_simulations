import numpy as np
import scipy.optimize
from scipy.optimize import LinearConstraint, Bounds

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