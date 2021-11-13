import numpy as np


class Advanced_tunning:
    # T: total rounds
    # d_t: Delay vector
    # k: dimension

    def __init__(self, T, d_t, k):
        self.d_t = d_t
        self.a_st = np.ones((T, T))  # [s,t]
        self.k = k
        self.D_t = 0  # delay counter
        self.t = 1  # round number

    def tune(self):
        t = self.t
        if t == 1:
            sigma_t = 0
        else:
            indicator = np.arange(1, t) + self.d_t[:t - 1] >= t
            sigma_t = (indicator * self.a_st[:t - 1, t - 1]).sum()

        self.D_t += sigma_t
        inv_eta_t = np.sqrt(self.D_t / np.log(self.k))

        for s in range(1, t):
            if min(self.d_t[s - 1], self.t - s) > inv_eta_t:
                self.a_st[s - 1, t:] = 0

        self.t += 1

        return inv_eta_t