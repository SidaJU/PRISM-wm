"""Actor-Critic（Dreamer 风格：在世界模型的「想象」中训练策略）。

世界模型只当「模拟器」用（参数冻结），策略在想象轨迹上学习，
因此**不需要真实环境交互**就能训练 —— 这是 model-based RL 的核心优势。

    Actor  : 策略网络  (h,z) -> 动作 a（连续，tanh 到 [-1,1]）
    Critic : 价值网络  (h,z) -> V（未来累积回报的估计）
"""
import torch
import torch.nn as nn


class Actor(nn.Module):
    """策略：给定模型状态 (h,z)，输出要执行的动作。"""

    def __init__(self, state_dim, action_dim, hidden=256, action_scale=1.0,
                 log_std_init=-1.0):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, action_dim),
        )
        # 随机策略：需要在动作上注入噪声才能做 REINFORCE（否则没有探索、也没有概率可优化）
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, hz):
        """确定性均值（评估时用）。tanh 压到 [-1,1] 再乘 action_scale。"""
        return torch.tanh(self.net(hz)) * self.action_scale

    def sample(self, hz):
        """随机采样动作（训练时用），返回 (动作, log_prob)。

        用重参数化采样，梯度可从 log_prob 回传到策略参数；
        但**不穿过世界模型**（rollout 用 no_grad），避免模型 OOD 时的错误梯度。
        """
        mean = self.forward(hz)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = raw.clamp(-1.0, 1.0)          # 环境动作范围 [-1,1]
        log_prob = dist.log_prob(raw).sum(-1)   # (B,)
        return action, log_prob


class Critic(nn.Module):
    """价值：估计当前状态未来的累积回报 V(s)。"""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hz):
        return self.net(hz).squeeze(-1)
