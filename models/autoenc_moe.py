import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN1
import torch.nn.functional as F
import numpy as np
from utils.utils import MODEL
from torch.optim import * 
from .moe_layer import MoE_Layer
from utils.c_adamw import CAdamW

@MODEL.register
class AutoEnc_MoE(nn.Module):
    
    def __init__(self, out_dim=1024, vec_dim=2048, num_steps=1,  
                 norm=True, normvec=True, residul=False, identify=False, query_num=701, 
                 temperature_init=14.28, bias_init=-10,
                 num_experts=4, top_k=2, 
                 noisy_gating=False,
                 num_chunks=20, nhead=4, dropout=0.1,
                 **opt):
        super(AutoEnc_MoE, self).__init__()
        self.num_steps = num_steps

        self.shared_enc = Seq(
            Lin(out_dim, vec_dim),
            nn.LayerNorm(vec_dim)
        )

        self.moe_layer = MoE_Layer(input_dim=vec_dim, output_dim=vec_dim, 
                                   num_experts=num_experts, k=top_k,
                                   noisy_gating=noisy_gating,
                                   num_chunks=num_chunks,
                                   nhead=nhead,
                                   dropout=dropout)

        self.dec1 = Seq(
            Lin(vec_dim, out_dim),
            BN1(out_dim)
        )
        self.dec2 = self.dec1 

        self.res = False 
        self.identify = identify
        self.query_num = query_num
        temperature = torch.nn.Parameter(torch.ones([]) * np.log(temperature_init))
        bias = torch.nn.Parameter(torch.ones([]) * bias_init)
        self.logit_scale = {"t":temperature, "b":bias}
        self.l2norm = lambda x: F.normalize(x, dim=-1) if norm else lambda s: s
        self.l2normvec = lambda x: F.normalize(x, dim=-1) if normvec else lambda x: x

    def run_train(self, data):
        src_x = data['x_s'] 
        tgt_x = data['x_t'] 
        
        if tgt_x.dim() == 3:
            b, n, c = tgt_x.shape
            tgt_x = tgt_x.reshape(b*n, c)[:self.query_num] 
        else:
            tgt_x = data['x_t'][:self.query_num] 
        data['x_t'] = tgt_x

        enc_a = self.shared_enc(src_x)
        
        feat_base = self.shared_enc(tgt_x)
        feat_delta, aux_loss, expert_loads = self.moe_layer(feat_base)
        enc_b = feat_base + feat_delta 
        
        data['aux_loss'] = aux_loss
        data['expert_loads'] = expert_loads

        fake_AA = self.dec1(enc_a) 
        fake_AB = self.dec2(enc_a) 
        fake_BA = self.dec1(enc_b) 
        fake_BB = self.dec2(enc_b) 

        feat_ab_base = self.shared_enc(fake_AB)
        feat_ab_delta, _, _ = self.moe_layer(feat_ab_base)
        enc_ab = feat_ab_base + feat_ab_delta
        fake_ABA = self.dec1(enc_ab) 

        enc_ba = self.shared_enc(fake_BA)
        fake_BAB = self.dec2(enc_ba) 

        data['fake_AA'] = self.l2norm(fake_AA)
        data['fake_BB'] = self.l2norm(fake_BB)
        data['fake_ABA'] = self.l2norm(fake_ABA)
        data['fake_BAB'] = self.l2norm(fake_BAB)
        data['norm_src'] = self.l2norm(src_x)
        data['norm_tgt'] = self.l2norm(tgt_x)
        data['enc_b'] = self.l2norm(enc_b)
        data['enc_a'] = self.l2norm(enc_a)
        data['enc_ba'] = self.l2norm(enc_ba)
        return data

    def run_eval(self, data):
        src = data['x'] 
        if self.identify:
            enc_x = src
            fake_src = src 
        elif data['mode'][0] == 'sat': 
            enc_x = self.shared_enc(src)
            fake_src = self.dec1(enc_x)
        else: 
            feat_base = self.shared_enc(src)
            feat_delta, _, _ = self.moe_layer(feat_base)
            enc_x = feat_base + feat_delta
            fake_src = self.dec2(enc_x)

        data['out'] = self.l2norm(enc_x)
        data['vec'] = self.l2normvec(fake_src)

        return data 

    def forward(self, data):       
        if self.training:
            data = self.run_train(data)
        else:
            data = self.run_eval(data)
        return data 

    def build_opt(self, opt):
        params = [
            {'params': self.shared_enc.parameters(), 'lr': opt.train.lr},
            {'params': self.moe_layer.parameters(), 'lr': opt.train.lr},
            {'params': self.dec1.parameters(), 'lr': opt.train.lr},
            {'params': self.logit_scale['t'], 'lr': opt.train.lr},
            {'params': self.logit_scale['b'], 'lr': opt.train.lr}
        ]
        optimizer = eval(opt.train.optim)(params)
        return optimizer