import os

import numpy as np
from sklearn.mixture import GaussianMixture

from tools import  log

FOLDER = os.path.dirname(os.path.realpath(__file__))


def _filter_instance(ins, mask: np.ndarray):
    res = {}

    for key, value in ins.items():
        if key == 'capacity':
            res[key] = value
            continue

        if key == 'duration_matrix':
            res[key] = value[mask]
            res[key] = res[key][:, mask]
            continue

        res[key] = value[mask]

    return res


def _greedy(ins, **_):
    mask = np.copy(ins['must_dispatch'])
    mask[:] = True
    return _filter_instance(ins, mask)


def _lazy(ins, **_):
    mask = np.copy(ins['must_dispatch'])
    mask[0] = True
    return _filter_instance(ins, mask)


def _random(ins, rng: np.random.Generator, args, **_):
    mask = np.copy(ins['must_dispatch'])
    mask = (mask | rng.binomial(1, p=args['random_p'], size=len(mask)).astype(np.bool8))
    mask[0] = True
    return _filter_instance(ins, mask)


class Feature():
    def __init__(self, static, k1=1000, k2=600, mask_size=(100, 200, 400, 600, 800, 1000)):
        # static: is_depot / coords / demands / capacity / time_windows / service_times / duration_matrix
        self.static = static
        self.k1 = k1
        self.k2 = k2

        xy = static['coords']

        self.masks = sum([self._get_mask(static['coords'], i, i) for i in mask_size], [])
        self.masks += [np.array([region_id == i for i in region_id]) for region_id in (self.split(xy, i) for i in 'ace')]
        mask_name = 'cir_front cir_back band_front band_back'.split()
        self.mask_name = [f'{j}_{i}' for i in mask_size for j in mask_name] + [f'{i}_region' for i in 'ace']

        self.dist = dist = np.sqrt(np.sum(np.square(xy.reshape(-1, 1, 2) - xy.reshape(1, -1, 2)), 2))
        self.d_depot = dist[0, 1:]
        self.dist_ = dist = dist[1:, 1:]
        dist = dist.copy()
        np.fill_diagonal(dist, np.nan)
        self.d_min = np.nanmin(dist, 0)
        self.d_max = np.nanmax(dist, 0)
        self.d_mean = np.nanmean(dist, 0)
        self.d_median = np.nanmedian(dist, 0)

        self.time = duration = static['duration_matrix'].copy().astype(float)
        self.t_to_depot = duration[1:, 0]
        self.t_from_depot = duration[0, 1:]
        self.time_ = duration = duration[1:, 1:]
        duration = duration.copy()
        np.fill_diagonal(duration, np.nan)
        # 从其他点来的平均用时
        self.t_to_min = np.nanmin(duration, 0)
        self.t_to_max = np.nanmax(duration, 0)
        self.t_to_mean = np.nanmean(duration, 0)
        self.t_to_median = np.nanmedian(duration, 0)
        # 去其他点的平均用时
        self.t_from_min = np.nanmin(duration, 1)
        self.t_from_max = np.nanmax(duration, 1)
        self.t_from_mean = np.nanmean(duration, 1)
        self.t_from_median = np.nanmedian(duration, 1)

        # 计算静态特征
        self.feature_static_1 = np.c_[(
            # 4个区域内有多少点，之后要乘上产生点数的期望
            *(np.sum(m, 1) for m in self.masks),
            *(np.sum(m, 0) for m in self.masks),
            # 距离倒数和
            *(np.array([np.sum(1 / (k1 + a[b])) for a, b in zip(self.dist_, m)]) for m in self.masks),
            # 时间倒数和
            *(np.array([np.sum(1 / (k2 + a[b])) for a, b in zip(self.time_, m)]) for m in self.masks),
        )]
        self.f1_names = [f'{i}_{j}' for i in 'cnt cntinv dist time'.split() for j in self.mask_name]
        assert self.feature_static_1.shape[1] == len(self.f1_names)
        self.feature_static_2 = np.c_[
            self.d_depot,
            self.d_min,
            self.d_max,
            self.d_mean,
            self.d_median,
            self.t_to_depot,
            self.t_from_depot,
            self.t_to_min,
            self.t_to_max,
            self.t_to_mean,
            self.t_to_median,
            self.t_from_min,
            self.t_from_max,
            self.t_from_mean,
            self.t_from_median,
        ]
        self.f2_names = 'd_depot d_min d_max d_mean d_median t_to_depot t_from_depot t_to_min t_to_max t_to_mean t_to_median t_from_min t_from_max t_from_mean t_from_median'.split()

    def calc_feature(self, epoch, obs):
        # obs: is_depot / customer_idx / request_idx / coords / demands / capacity / time_windows / service_times / duration_matrix / must_dispatch
        n = len(self.static['coords']) - 1
        tw = self.static['time_windows']
        t1 = np.maximum((epoch + 1) * 3600 + self.t_from_depot.mean(), tw[1:, 0])
        t2 = t1 + self.static['service_times'].mean() + self.t_to_depot.mean()
        p_sample = np.sum((t1 <= tw[1:, 1]) & (t2 <= tw[0, 1])) / n * 100 / n
        cid = obs['customer_idx'][1:]
        fs_1 = self.feature_static_1[cid - 1, :] * p_sample

        tw = obs['time_windows']
        td = obs['duration_matrix']
        ts = obs['service_times']
        t1 = (np.maximum(td[0, 1:], tw[1:, 0]) + ts[1:]).reshape(-1, 1) + td[1:, 1:]
        t2 = np.maximum(t1, tw[1:, 0].reshape(1, -1)) + (ts[1:] + td[1:, 0]).reshape(1, -1)
        viable = (t1 <= tw[1:, 1].reshape(1, -1)) & (t2 <= tw[0, 1])
        viable |= viable.T

        ind = np.arange(n) + 1
        # 动态单
        masks = [np.array([np.isin(cid, ind[r]) for r in m[cid - 1]]) & viable for m in self.masks]
        fs_2 = np.c_[(
            *(np.sum(m, 1) for m in masks),
            *(np.sum(m, 0) for m in masks),
            *(np.array([np.sum(1 / (self.k1 + a[b])) for a, b in zip(self.dist[np.ix_(cid, cid)], m)]) for m in masks),
            *(np.array([np.sum(1 / (self.k2 + a[b])) for a, b in zip(td[1:, 1:], m)]) for m in masks),
        )]
        # 必送单
        masks = [m & obs['must_dispatch'][1:].reshape(1, -1) for m in masks]
        fs_3 = np.c_[(
            *(np.sum(m, 1) for m in masks),
            *(np.sum(m, 0) for m in masks),
            *(np.array([np.sum(1 / (self.k1 + a[b])) for a, b in zip(self.dist[np.ix_(cid, cid)], m)]) for m in masks),
            *(np.array([np.sum(1 / (self.k2 + a[b])) for a, b in zip(td[1:, 1:], m)]) for m in masks),
        )]
        assert np.all(self.time[np.ix_(cid, cid)] == td[1:, 1:])

        tw2 = np.clip(self.static['time_windows'] - (epoch + 1) * 3600, 0, None)
        tw2 = tw2[tw2[:, 1] > 0, :]
        td2 = self.time[cid, :]
        trs = []
        for f in [np.min, np.max, np.mean, np.median]:
            if len(tw2) == 0:
                trs.append(np.zeros(len(t1)))
            else:
                t1 = (np.maximum(td[0, 1:], tw[1:, 0]) + ts[1:] + f(td2, 1)).reshape(-1, 1)
                t2 = np.maximum(t1, tw2[:, 0].reshape(1, -1)) + f(ts[1:]) + f(td2[:, 0])
                trs.append(np.mean((t1 <= tw2[:, 1].reshape(1, -1)) & (t2 <= tw[0, 1]), 1) * p_sample)

        return np.c_[(
            fs_1,
            fs_2,
            fs_3,
            *trs,
            self.feature_static_2[cid - 1, :],
            np.minimum(tw[0, 1] - td[1:, 0] - ts[1:], tw[1:, 1]) - td[0, 1:],
            obs['demands'][1:] / obs['capacity'],
            obs['must_dispatch'][1:],
        )]

    @property
    def feature_names(self):
        return (
            [f'static_{i}' for i in self.f1_names] +
            [f'dynamic_{i}' for i in self.f1_names] +
            [f'must_{i}' for i in self.f1_names] +
            [f'tw_{i}' for i in 'min max mean median'.split()] +
            self.f2_names + ['latest_time', 'demand', 'is_must_dispatch']
        )

    @ staticmethod
    def _get_mask(coords, radius, width):
        o = coords[0]
        x = coords[1:]
        # 方向
        u = x - o
        u = u / np.linalg.norm(u, axis=1).reshape(-1, 1)
        # 法向
        n = np.c_[-u[:, 1], u[:, 0]]
        # 点与点之间的向量
        b = -x.reshape(-1, 1, 2) + x.reshape(1, -1, 2)
        # 直线距离在范围内
        m1 = np.sum(np.square(b), 2) <= radius**2
        # 投影距离在范围内
        m2 = np.abs(np.sum(n.reshape(-1, 1, 2) * b, 2)) <= width
        # 在前侧
        m3 = np.sum(u.reshape(-1, 1, 2) * b, 2) <= 0
        # 在后侧
        m4 = ~m3
        return [
            # 圆形区前侧的点
            m3 & m1,
            # 圆形区后侧的点
            m4 & m1,
            # 条带区前侧的点
            m3 & m2,
            # 条带区后侧的点
            m4 & m2,
        ]

    @staticmethod
    def split(x, method, split_k=7):
        if method not in 'abcdef':
            raise NotImplementedError
        if method == 'a':
            x = x[1:]
        elif method == 'b':
            d1 = np.sqrt(np.sum(np.square(x[1:] - x[0]), 1))
            x = np.c_[d1, x[1:]]
        elif method == 'c':
            d2 = np.sin(np.arctan2(*(x[1:] - x[0]).T[[1, 0]]))
            x = np.c_[d2, x[1:]]
        elif method == 'd':
            d1 = np.sqrt(np.sum(np.square(x[1:] - x[0]), 1))
            d2 = np.sin(np.arctan2(*(x[1:] - x[0]).T[[1, 0]]))
            x = np.c_[d1, d2, x[1:]]
        elif method == 'e':
            d2 = np.arctan2(*(x[1:] - x[0]).T[[1, 0]])
            x = np.c_[d2, x[1:]]
        elif method == 'f':
            d1 = np.sqrt(np.sum(np.square(x[1:] - x[0]), 1))
            d2 = np.arctan2(*(x[1:] - x[0]).T[[1, 0]])
            x = np.c_[d1, d2, x[1:]]
        return GaussianMixture(split_k, random_state=43).fit_predict(x)



def _ppo(ins, current_epoch, start_epoch, end_epoch, static=None, saved_state=None, **_):
    if static is None:
        return _greedy(ins)
    assert saved_state is not None
    mask = np.copy(ins['must_dispatch'])
    mask[0] = True
    if np.all(mask[1:]):
        return _greedy(ins)
    if 'ppo' in saved_state:
        feat = saved_state['feat']
        model = saved_state['ppo']
    else:
        from baselines.strategies._ppo import Predictor
        feat = saved_state['feat'] = Feature(static)
        model = saved_state['ppo'] = Predictor(FOLDER)
        log(F'Running PPO model on {model.device}')
    if not np.any(mask[1:]) and start_epoch == current_epoch:
        return _filter_instance(ins, mask)
    ft = feat.calc_feature(current_epoch, ins)[~mask[1:]]
    mask[~mask] |= model.predict(current_epoch, end_epoch, ft)
    return _filter_instance(ins, mask)


STRATEGIES = dict(
    greedy=_greedy,
    lazy=_lazy,
    random=_random,
    ppo=_ppo,
)
