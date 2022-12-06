import json

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def unzip_fid(s):
    s = s.split(',')
    out = []
    for i in s:
        if '~' in i:
            a, b = map(int, i.split('~'))
            out += range(a, b + 1)
        else:
            out.append(int(i))
    return out


def make_mlp(*shape, dropout=0.1, act=nn.Tanh, sigma=0):
    ls = [nn.Linear(i, j) for i, j in zip(shape, shape[1:])]
    if sigma > 0:
        for l in ls:
            nn.init.orthogonal_(l.weight, 2**0.5)
            nn.init.constant_(l.bias, 0)
        nn.init.orthogonal_(ls[-1].weight, sigma)
    return nn.Sequential(
        *sum(([
            l,
            act(),
            nn.Dropout(dropout),
        ] for l in ls[:-1]), []),
        ls[-1]
    )


class AgentGRU(nn.Module):
    def __init__(self, num_features, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            make_mlp(num_features, 64, 64),
            nn.Tanh()
        )
        self.pool = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.gru = nn.GRU(64, 64)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = make_mlp(128, 64, 2, dropout=dropout, act=nn.Tanh, sigma=0.01)
        self.critic = make_mlp(64, 64, 1, dropout=dropout, act=nn.Tanh, sigma=1)

    def _pool(self, x):
        encoded = [self.encoder(i) for i in x]
        return encoded, torch.stack([self.pool(torch.cat([i.mean(0), i.max(0).values])) for i in encoded])

    def get_value(self, obs):
        return self.critic(self._pool(obs)[1])

    def get_action_and_value(self, obs, gru_state, done, action=None, sample=True):
        encoded, pooled = self._pool(obs)
        batch_size = gru_state.shape[1]
        pooled = pooled.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))
        hidden = []
        for h, d in zip(pooled, done):
            h, gru_state = self.gru(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1) * gru_state,
            )
            hidden.append(h)
        hidden = torch.flatten(torch.cat(hidden), 0, 1)
        acts = []
        log_probs = []
        entropy = 0
        ls = []
        for i, (e, h) in enumerate(zip(encoded, hidden)):
            logits = self.actor(torch.hstack([h.view(1, -1).expand(e.shape[0], -1), e]))
            probs = Categorical(logits=logits)
            if action is None:
                if sample:
                    act = probs.sample()
                else:
                    raise NotImplementedError
                acts.append(act.view(-1))
            else:
                act = action[i]
            log_probs.append(probs.log_prob(act).sum())
            entropy = entropy + probs.entropy().mean()
            ls.append(logits)
        return (
            action or acts,
            (torch.stack(log_probs) if sample else None),
            ls,
            entropy / len(obs),
            self.critic(pooled),
            gru_state
        )


class Predictor():
    def __init__(self, path):
        self.fid = unzip_fid('0~242,270~344')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AgentGRU(len(self.fid) + 1)
        self.model.load_state_dict(torch.load(path + '/ppo_model.pt', map_location=self.device))
        self.model.eval()
        j = json.load(open(path + '/ppo_data.json'))
        f_m = np.array(j['f_m'])
        f_s = np.array(j['f_s'])
        self.f_m = f_m[self.fid]
        self.f_s = 1 / f_s[self.fid]
        self.gru_state = torch.zeros(self.model.gru.num_layers, 1, self.model.gru.hidden_size).to(self.device)
        self.done = torch.zeros(1, device=self.device)

    def predict(self, current_epoch, end_epoch, ft):
        with torch.no_grad():
            obs = torch.from_numpy(
                np.c_[[[(end_epoch - current_epoch) / 10]] * len(ft), (ft[:, self.fid] - self.f_m) * self.f_s]
            ).float().to(self.device)
            _, _, (logits,), _, _, self.gru_state = self.model.get_action_and_value([obs], self.gru_state, self.done)
            return (logits[:, 1] > logits[:, 0]).view(-1).cpu().numpy()
