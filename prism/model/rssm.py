"""最小 RSSM（连续高斯版）—— DreamerV3 的动力学核心。

W3 阶段先用**连续高斯** latent（比 DreamerV3 的离散 categorical 简单、好调、易验证），
跑通后再升级到离散 latent + KL balancing（W4）。

结构：
    e_t    = Encoder(o_t)                       图像 -> 观测嵌入
    h_t    = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])   确定性路径
    prior     p(z_t | h_t)                      动力学预测（不看当前观测）
    posterior q(z_t | h_t, e_t)                 推理（看当前观测）
    重建      Decoder(h_t, z_t) -> o_t

损失 = 图像重建 MSE + latent 预测 MSE + KL(posterior || prior)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    """64x64x3 -> emb"""

    def __init__(self, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),    # 32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),   # 16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(), # 4
        )
        self.fc = nn.Linear(256 * 4 * 4, emb_dim)

    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h)


class ObsDecoder(nn.Module):
    """emb -> 64x64x3"""

    def __init__(self, emb_dim=256):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).reshape(-1, 256, 4, 4)
        return self.deconv(h)


class RSSM(nn.Module):
    def __init__(self, obs_shape=(3, 64, 64), proprio_dim=10, action_dim=4,
                 latent_dim=32, hidden_dim=200, emb_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        self.encoder = ObsEncoder(emb_dim)
        self.decoder = ObsDecoder(emb_dim)

        # 确定性循环
        self.gru = nn.GRUCell(latent_dim + action_dim + proprio_dim, hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        # 由 (h, z) 预测下一帧的观测嵌入与本体增量
        self.obs_pred = nn.Linear(hidden_dim + latent_dim, emb_dim)
        self.proprio_pred = nn.Linear(hidden_dim + latent_dim, proprio_dim)
        # v5：物体位置预测头。从 (h,z) 直接预测物体 xyz —— 逼 z 装载"物体在哪"的信息，
        # 提供"难而重要"的梯度（当前 emb/proprio 早期就饱和到 1e-5，后期几乎无梯度）。
        self.object_pred = nn.Linear(hidden_dim + latent_dim, 3)

    @staticmethod
    def _rsample(mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.latent_dim, device=device))

    def forward(self, images, proprio, actions, state=None):
        """images(T,B,3,H,W) proprio(T,B,P) actions(T,B,A)"""
        T, B = images.shape[0], images.shape[1]
        device = images.device

        # 观测嵌入（一次性编码整段，效率高）
        flat = images.reshape(T * B, *images.shape[2:])
        emb = self.encoder(flat).reshape(T, B, -1)

        h, z = state if state is not None else self.init_state(B, device)

        outs = {'emb': [], 'emb_pred': [], 'proprio_pred': [], 'object_pred': [], 'kl': []}
        kls = []
        for t in range(T):
            gru_in = torch.cat([z, actions[t], proprio[t]], dim=-1)
            h = self.gru(gru_in, h)

            prior = self.prior_net(h)
            mu_p, lv_p = prior.chunk(2, dim=-1)

            post = self.post_net(torch.cat([h, emb[t]], dim=-1))
            mu_q, lv_q = post.chunk(2, dim=-1)
            z = self._rsample(mu_q, lv_q)

            # KL(q || p)，两个对角高斯的解析形式
            kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp().clamp(min=1e-4) - 1)
            kls.append(kl.sum(-1))

            hz = torch.cat([h, z], dim=-1)
            outs['emb_pred'].append(self.obs_pred(hz))
            outs['proprio_pred'].append(self.proprio_pred(hz))
            outs['object_pred'].append(self.object_pred(hz))

        out = {
            'emb': emb,                                        # (T,B,E) 真实观测嵌入
            'emb_pred': torch.stack(outs['emb_pred']),          # (T,B,E) 预测
            'proprio_pred': torch.stack(outs['proprio_pred']),  # (T,B,P)
            'object_pred': torch.stack(outs['object_pred']),     # (T,B,3) 预测的物体 xyz
            'kl': torch.stack(kls),                             # (T,B)
        }
        # 图像重建只解码最后一帧太浪费，这里解码全部帧（T*B 一次前向）
        out['recon'] = self.decoder(out['emb_pred'].reshape(T * B, -1)).reshape(T, B, *images.shape[2:])
        return out

    # ---------- 想象（做梦）----------
    def observe(self, images, proprio, actions):
        """用一段真实观测建立初始状态，返回最后的 (h, z)。

        z 走 posterior（看得到图像），与 forward 完全一致 —— 相当于"先看清现在"。
        """
        T, B = images.shape[0], images.shape[1]
        device = images.device
        flat = images.reshape(T * B, *images.shape[2:])
        emb = self.encoder(flat).reshape(T, B, -1)

        h, z = self.init_state(B, device)
        for t in range(T):
            gru_in = torch.cat([z, actions[t], proprio[t]], dim=-1)
            h = self.gru(gru_in, h)
            post = self.post_net(torch.cat([h, emb[t]], dim=-1))
            mu_q, lv_q = post.chunk(2, dim=-1)
            z = self._rsample(mu_q, lv_q)
        return h, z

    def imagine(self, h, z, actions, proprio, horizon, sample=True):
        """不看任何图像，只用 prior 往前推 horizon 步。

        与 forward 的关键区别：z 从 **prior** 采样（posterior 需要 emb，想象时没有）。
        本体也用模型自己的预测值（没有真值可喂）。

        参数
        ----
        h, z       : observe() 给出的初始状态，形状 (B,H) / (B,Z)
        actions    : (horizon, B, A) 未来要执行的动作
        proprio    : (B, P) 起始本体（第一步 GRU 输入）
        horizon    : 往前推几步

        返回
        ----
        emb_pred : (horizon, B, E) 想象出的画面摘要（可送 decoder 得到想象画面）
        prop_pred: (horizon, B, P) 想象出的本体
        obj_pred : (horizon, B, 3) 想象出的物体位置（v5）
        """
        emb_hats, prop_hats, obj_hats = [], [], []
        for t in range(horizon):
            # 1) 用上一步的 z 和本体推进记忆
            gru_in = torch.cat([z, actions[t], proprio], dim=-1)
            h = self.gru(gru_in, h)
            # 2) 从 prior 得到 z（没有图像可看）
            #    sample=True  : 按分布采样（带随机噪声，画面可能出现噪点）
            #    sample=False : 直接用均值 mu（确定性推演，画面稳定）
            mu_p, lv_p = self.prior_net(h).chunk(2, dim=-1)
            z = self._rsample(mu_p, lv_p) if sample else mu_p
            # 3) 预测这一"想象帧"的摘要与本体
            hz = torch.cat([h, z], dim=-1)
            emb_hats.append(self.obs_pred(hz))
            proprio = self.proprio_pred(hz)      # 下一步用自己预测的本体
            prop_hats.append(proprio)
            obj_hats.append(self.object_pred(hz))   # v5：想象的物体位置

        return torch.stack(emb_hats), torch.stack(prop_hats), torch.stack(obj_hats)


def rssm_loss(out, images, proprio, kl_weight=1.0, emb_weight=1.0,
              recon_weight=1.0, proprio_weight=1.0, free_bits=0.0,
              object_pos=None, object_weight=1.0):
    """object_pos: (T,B,3) 物体位置真值（监督目标，不是模型输入）。
    给了它就会多一项 object_mse —— 这是 v5 逼 z 装载物体信息的关键。"""
    T, B = images.shape[0], images.shape[1]
    recon = torch.mean((out['recon'] - images) ** 2)
    emb_mse = torch.mean((out['emb_pred'] - out['emb'].detach()) ** 2)
    proprio_mse = torch.mean((out['proprio_pred'] - proprio) ** 2)

    # v5：物体位置预测误差（单位：米²；典型值 1e-4 量级）
    object_mse = (torch.mean((out['object_pred'] - object_pos) ** 2)
                  if object_pos is not None else torch.zeros((), device=images.device))

    kl = out['kl']
    if free_bits > 0:      # 防止 posterior collapse
        kl = torch.clamp(kl, min=free_bits)
    kl = torch.mean(kl)

    total = (recon_weight * recon + emb_weight * emb_mse
             + proprio_weight * proprio_mse + kl_weight * kl
             + object_weight * object_mse)
    return {
        'loss': total, 'recon': recon.detach(), 'emb': emb_mse.detach(),
        'proprio': proprio_mse.detach(), 'kl': kl.detach(),
        'object': object_mse.detach(),
    }
