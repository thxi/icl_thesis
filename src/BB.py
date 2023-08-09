import bottleneck as bn
import numpy as np
import torch


class BB:
    # a simple bollinger bands class
    def __init__(self, init_obs, window_size=20, num_std=2):
        # assume we always have `window_size` initial observations
        assert len(init_obs) == window_size
        self.observations = init_obs
        self.window_size = window_size
        self.num_std = num_std

        self.sum = np.sum(init_obs)
        self.ss = np.sum(init_obs**2)
        self.mean = self.sum / window_size
        var = self.ss / window_size - (self.mean) ** 2
        self.std = np.sqrt(var)

    def update_state(self, x):
        # update the state with a new observation
        self.sum += x - self.observations[0]
        self.ss += x**2 - self.observations[0] ** 2
        self.mean = self.sum / self.window_size
        var = self.ss / self.window_size - (self.mean) ** 2
        var = max(var, 1e-6)
        self.std = np.sqrt(var)

        # TODO: very inefficient, should probably use a deque
        self.observations = np.concatenate((self.observations[1:], [x]))

    def fit(self, X, y):
        pass

    def predict_one(self, x):
        if abs(x - self.mean) > self.num_std * self.std:
            return 1
        return 0

    def predict_many(self, X):
        # assume that X is always >= window_size
        obs = np.concatenate((self.observations, X))

        mm = bn.move_mean(obs, window=self.window_size)
        ms = bn.move_std(obs, window=self.window_size)
        y = np.where(abs(obs - mm) > self.num_std * ms, 1, 0)[self.window_size :]
        self.observations = X[-self.window_size :]
        return y


_ZERO_GPU = torch.tensor(1e-6, device="cuda")


class BB_gpu:
    # a simple bollinger bands class
    def __init__(self, init_obs, window_size=20, num_std=2):
        # assume we always have `window_size` initial observations
        assert len(init_obs) == window_size
        self.observations = init_obs
        self.window_size = window_size
        self.num_std = num_std

        self.sum = torch.sum(init_obs)
        self.ss = torch.sum(init_obs**2)
        self.mean = self.sum / window_size
        var = self.ss / window_size - (self.mean) ** 2
        self.std = torch.sqrt(var)

    def update_state(self, x):
        # assuming that x is a tensor scalar on gpu
        # update the state with a new observation
        self.sum += x - self.observations[0]
        self.ss += x**2 - self.observations[0] ** 2
        self.mean = self.sum / self.window_size
        var = self.ss / self.window_size - (self.mean) ** 2
        var = torch.max(var, _ZERO_GPU)
        self.std = torch.sqrt(var)

        # TODO: incredibly inefficient
        self.observations = torch.concatenate((self.observations[1:], x.reshape(-1)))

    def fit(self, X, y):
        pass

    def predict_one(self, x):
        if abs(x - self.mean) > self.num_std * self.std:
            return 1
        return 0

    def predict_many(self, X):
        # https://krzjoa.github.io/2019/12/28/content/post/2019-12-28-pytorch-ts-v1/2019-12-28-pytorch-ts-v1/
        # assume that X is always >= window_size
        # rolling average: https://stackoverflow.com/a/63362753
        obs = torch.concatenate((self.observations, X))

        mm, ms = moving_mean_std(obs, self.window_size)
        mm = mm[1:]
        ms = ms[1:]
        # note that we have trimmed mm here so using X
        y = torch.where(abs(X - mm) > self.num_std * ms, 1, 0)
        self.observations = X[-self.window_size :]
        return y


def moving_mean_std(x, win):
    # moving_mean_std which trims the time series

    cs = torch.cumsum(x, dim=0)
    # trimmed moving sum
    m_sum = torch.concat((cs[win - 1].reshape(-1), cs[win:] - cs[:-win]))

    cs = torch.cumsum(x**2, dim=0)
    # trimmed moving sum of squares
    m_ss = torch.concat((cs[win - 1].reshape(-1), cs[win:] - cs[:-win]))

    m_mean = m_sum / win
    var = m_ss / win - (m_mean) ** 2
    std = torch.sqrt(var)

    return m_mean, std


# for testing moving mean
if __name__ == "__main__":
    import pandas as pd

    a = pd.Series([10, 1, 2, 3, 4, 5])
    win = 3
    print(a.rolling(win).mean())
    print((a).rolling(win).std(ddof=0))

    m_mean, m_std = moving_mean_std(torch.tensor(a), win)
    print(m_mean, m_std)
