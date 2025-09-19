import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from Modules import *
import warnings
warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--p1', type=float, default=0.1,
                        help='threshold of edit sequences.')
    parser.add_argument('--p2', type=float, default=0.1,
                        help='threshold of omit items.')
    parser.add_argument('--eval', type=int, default=5,
                        help='evaluation frequency')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stop patience.')
    parser.add_argument('--linespace', type=int, default=100,
                        help='linespace of DDIM sampling')
    parser.add_argument('--data', nargs='?', default='zhihu',
                        help='yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--w', type=float, default=2.0,
                        help='classifier-free guidance weight for sampling')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')

    # --- Arguments for strategies ---
    parser.add_argument('--strategy', type=str, default='none',
                        help='Strategy: none, nas, cl, nas_cl, pred_future, pred_future_cl')
    parser.add_argument('--nas_p', type=float, default=0.2,
                        help='Strength of noise-aware down-weighting (if enabled).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for the contrastive loss.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss.')

    # --- Predictive Future Strategy ---
    parser.add_argument('--future_window_size', type=int, default=3,
                        help='Window size (W) for calculating future interest average.')
    parser.add_argument('--continuity_threshold', type=float, default=0.1,
                        help='Threshold to identify low-continuity items for robust weighting.')
    parser.add_argument('--score_temp', type=float, default=1.0,
                        help='Temperature for mapping significance score to attention weight.')
    parser.add_argument('--w_min', type=float, default=0.0,
                        help='Minimum attention weight after mapping (clipping).')
    parser.add_argument('--w_max', type=float, default=2.0,
                        help='Maximum attention weight after mapping (clipping).')

    # --- Mutual exclusion (routing) ---
    parser.add_argument('--stability_threshold', type=float, default=0.8,
                        help='Entropy threshold to route sequences: high entropy -> pred_future; low entropy -> DTS.')

    parser.add_argument('--save_path', type=str, default='checkpoints/card_default_best.pt',
                        help='Path to save the best performing model checkpoint.')

    return parser.parse_args()

args = parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# --- Contrastive loss for optional CL strategy ---
def info_nce_loss(query, positive_key, negative_keys, temperature=0.1, reduction='mean'):
    query = F.normalize(query, dim=1)
    positive_key = F.normalize(positive_key, dim=1)
    negative_keys = F.normalize(negative_keys, dim=1)
    logits = torch.matmul(query, negative_keys.transpose(0, 1)) / temperature
    mask = torch.eye(query.shape[0], device=query.device, dtype=torch.bool)
    positive_logits = logits[mask].view(query.shape[0], -1)
    negative_logits = logits[~mask].view(query.shape[0], -1)
    final_logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    loss = F.cross_entropy(final_logits, labels, reduction=reduction)
    return loss

class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w,linespace):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w
        self.linespace = linespace

        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # DDIM Reverse Process
        indices = list(range(0, self.timesteps+1, self.linespace))
        self.sub_timesteps = len(indices)
        indices_now = [indices[i]-1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)

        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(1.-self.alphas_cumprod_ddim_prev)/ self.sqrt_recipm1_alphas_cumprod_ddim
        self.posterior_ddim_coef2 = torch.sqrt(1.-self.alphas_cumprod_ddim_prev) / torch.sqrt(1. - self.alphas_cumprod_ddim)

        # x_{t-1} = self.posterior_mean_coef1 * x_0 + self.posterior_mean_coef2 * x_t + self.posterior_variance * eps
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model(x_noisy, h, t)
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()
        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x 
        model_mean = (
            self.posterior_ddim_coef1[t_index] * x_start +
            self.posterior_ddim_coef2[t_index] * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h):
        x = torch.randn_like(h)
        for n in reversed(range(0, self.sub_timesteps)):
            step = torch.full((h.shape[0], ), n*self.linespace, device=device, dtype=torch.long)
            x = self.p_sample(model_forward, model_forward_uncon, x, h,step, n)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device_t = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device_t) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
        
class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )

        # --- Modules for Predictive Future Strategy ---
        if 'pred_future' in args.strategy:
            self.causal_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.causal_encoder = nn.TransformerEncoder(self.causal_encoder_layer, num_layers=1)
            self.aux_prediction_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, x, h, step):
        t = self.step_mlp(step)
        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, self.hidden_size)]*x.shape[0], dim=0)
        t = self.step_mlp(step)
        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def cacu_x(self, x):
        x = self.item_embeddings(x)
        return x

    def cacu_h(self, states, len_states,p,p1,p2):
        inputs_emb = self.item_embeddings(states)
        seq_emb = inputs_emb.clone()
        B, L= states.size(0), states.size(1)

        # --- Continuity and stability ---
        # continuity_scores: [B, L-1]
        continuity_scores = torch.nn.functional.cosine_similarity(seq_emb[:, :L-1], seq_emb[:, 1:L], dim=2)
        continuity_probs = F.softmax(continuity_scores, dim=1) # time-softmax
        # entropy as stability proxy (higher -> less stable)
        stability_raw = -torch.sum(continuity_probs * torch.log(continuity_probs + 1e-9), dim=1) # [B]

        # routing masks
        is_low_stability = stability_raw > args.stability_threshold
        is_high_stability = ~is_low_stability

        # init weights (will be applied after positional emb + dropout)
        weights = torch.ones(B, L, device=self.device)

        # --- Path A: Low Stability -> Predictive Future Weighting ---
        if 'pred_future' in args.strategy and torch.any(is_low_stability):
            W = args.future_window_size
            low_idx = torch.where(is_low_stability)[0]
            low_embs = inputs_emb[low_idx]
            if low_embs.size(0) > 0:
                # causal mask: upper triangle set to -inf
                causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=self.device), 1)
                causal_hidden = self.causal_encoder(low_embs, mask=causal_mask) # [b_low, L, D]
                # future average targets via sliding window
                padded = F.pad(low_embs, (0, 0, 0, W)) # [b_low, L+W, D]
                future_patches = padded.unfold(dimension=1, size=W, step=1)[:, 1:L+1, :, :] # [b_low, L, D, W]
                future_targets = future_patches.mean(dim=3) # [b_low, L, D]
                h_with = causal_hidden
                h_without = F.pad(causal_hidden[:, :-1, :], (0, 0, 1, 0))
                pred_with = self.aux_prediction_head(h_with)
                pred_without = self.aux_prediction_head(h_without)
                loss_with = F.mse_loss(pred_with, future_targets, reduction='none').mean(dim=2) # [b_low, L]
                loss_without = F.mse_loss(pred_without, future_targets, reduction='none').mean(dim=2)
                scores = loss_without - loss_with # [b_low, L]

                # low-continuity positions only (pad first timestep to 1.0)
                cont_probs_padded = F.pad(continuity_probs[low_idx], (1, 0), value=1.0) # [b_low, L]
                is_candidate = cont_probs_padded < args.continuity_threshold
                filtered_scores = torch.where(is_candidate, scores, torch.zeros_like(scores))
                robust_weights = 1.0 + torch.tanh(filtered_scores / args.score_temp)
                robust_weights = torch.clamp(robust_weights, min=args.w_min, max=args.w_max)
                weights[low_idx] = robust_weights

        # --- Path B: High Stability -> DTS data sparsity ---
        states_dts = states.clone()
        len_states_dts = len_states.clone()
        high_idx = torch.where(is_high_stability)[0]
        if high_idx.numel() > 0:
            # sequence-level trigger probability: batch-softmax over entropy
            stability_prob = F.softmax(stability_raw, dim=0) # [B]
            for r_idx in high_idx:
                r = r_idx.item()
                prob_seq = float(args.p1) * float(stability_prob[r]) * float(B)
                if prob_seq > 1.0:
                    prob_seq = 1.0
                if random.random() < prob_seq:
                    # item-level removal prefers high continuity positions
                    current_len = int(len_states_dts[r].item()) if isinstance(len_states_dts[r], torch.Tensor) else int(len_states_dts[r])
                    for c in range(current_len - 1):
                        prob_c = float(args.p2) * float(L) * float(continuity_probs[r, c])
                        if prob_c > 1.0:
                            prob_c = 1.0
                        if random.random() < prob_c:
                            states_dts[r, c:L-1] = states_dts[r, c+1:L].clone()
                            states_dts[r, L-1] = self.item_num
                            len_states_dts[r] = len_states_dts[r] - 1

        # --- Final encoding ---
        inputs_emb = self.item_embeddings(states_dts)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        seq = seq * weights.unsqueeze(-1) # apply weights (low-stability sequences only)
        mask = torch.ne(states_dts, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states_dts - 1)
        h = state_hidden.squeeze()                                                  
        B_h, D_h = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B_h) - p) + 1) / 2
        maske1d = mask1d.view(B_h, 1)
        mask2 = torch.cat([maske1d] * D_h, dim=1)
        mask2 = mask2.to(self.device)
        h = h * mask2 + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask2)
        return h  
    
    def predict(self, states, len_states, diff):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()
        x = diff.sample(self.forward, self.forward_uncon, h)
        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))
        return scores


def evaluate(model, test_data, diff, device):
    eval_data=pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    total_purchase = 0.0
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)
    num_total = len(seq)

    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction = model.predict(states, np.array(len_seq_b), diff)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        sorted_list2 = sorted_list2
        calculate_hit(sorted_list2,topk,target_b,hit_purchase,ndcg_purchase)
        total_purchase+=batch_size

    hr_list = []
    ndcg_list = []
    metrics = {}
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])
        metrics[f'HR@{topk[i]}'] = hr_purchase
        metrics[f'NDCG@{topk[i]}'] = ng_purchase[0,0]
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    return metrics


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # Make data path robust to current working directory by anchoring to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, 'data', args.data)
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))
    seq_size = data_statis['seq_size'][0]
    item_num = data_statis['item_num'][0]
    topk=[10, 20, 50]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timesteps = args.timesteps

    model = Tenc(args.hidden_factor,item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w,args.linespace)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

    best_test_hr20 = 0
    best_test_epoch = 0
    best_test_metrics_overall = {}

    best_val_hr20 = 0
    best_val_epoch = 0
    test_metrics_at_best_val = {}
    early_stop_counter = 0
    patience = args.patience 

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for i in range(args.epoch):
        start_time = Time.time()
        model.train()
        total_loss_value = 0
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target=list(batch['next'].values())

            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)

            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)

            x_start = model.cacu_x(target)
            h = model.cacu_h(seq, len_seq,args.p,args.p1,args.p2)
            n = torch.randint(0, args.timesteps, (args.batch_size, ), device=device).long()
            loss_diff, predicted_x = diff.p_losses(model, x_start, h, n, loss_type='l2')
            total_loss = loss_diff

            if 'cl' in args.strategy:
                loss_cl = info_nce_loss(h, x_start, x_start, temperature=args.temperature)
                total_loss = total_loss + args.alpha * loss_cl

            total_loss.backward()
            optimizer.step()
            total_loss_value += total_loss.item()

        model.eval()
        avg_loss = total_loss_value / num_batches
        if args.report_epoch:
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(avg_loss) + "Time cost: " + Time.strftime(
                        "%H: %M: %S", Time.gmtime(Time.time()-start_time)))

            if (i + 1) % args.eval == 0:
                eval_start = Time.time()
                print('-------------------------- VAL PHRASE --------------------------')
                val_metrics = evaluate(model, 'val_data.df', diff, device)
                print('-------------------------- TEST PHRASE -------------------------')
                test_metrics = evaluate(model, 'test_data.df', diff, device)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
                
                # Check for best test performance and save model
                if test_metrics['HR@20'] > best_test_hr20:
                    best_test_hr20 = test_metrics['HR@20']
                    best_test_epoch = i
                    best_test_metrics_overall = test_metrics
                    print(f"*** New best test HR@20: {best_test_hr20:.6f} at epoch {i}. Saving model... ***")
                    
                    # Ensure the save directory exists
                    save_dir = os.path.dirname(args.save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    
                    torch.save(model.state_dict(), args.save_path)

                # Early stopping logic based on validation set
                if val_metrics['HR@20'] > best_val_hr20:
                    best_val_hr20 = val_metrics['HR@20']
                    best_val_epoch = i
                    test_metrics_at_best_val = test_metrics
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                print(f'--- Epoch {i}: Best Val HR@20: {best_val_hr20:.6f} (at epoch {best_val_epoch})')
                print(f'--- Early stopping counter: {early_stop_counter}/{patience}')
                print('----------------------------------------------------------------')

    print("\n\n" + "="*80)
    print("                      TRAINING FINISHED")
    print("="*80)

    print("\n--- Overall Best Performance on Test Set (Peak Performance) ---")
    if best_test_metrics_overall:
        print(f"Achieved at Epoch: {best_test_epoch}")
        for k, v in best_test_metrics_overall.items():
            print(f"{k}: {v:.6f}")
    else:
        print("No evaluation was performed on the test set.")

    print("\n--- Early Stopping Result (Test performance at best validation epoch) ---")
    if test_metrics_at_best_val:
        print(f"Best validation performance was at Epoch: {best_val_epoch} (HR@20: {best_val_hr20:.6f})")
        print("Corresponding performance on Test Set:")
        for k, v in test_metrics_at_best_val.items():
            print(f"{k}: {v:.6f}")
    else:
        print("No evaluation was performed for early stopping.")
    print("\n" + "="*80)


