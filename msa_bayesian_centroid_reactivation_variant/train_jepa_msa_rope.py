import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import math
import queue
import random
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# 1. Config: Scaled for RTX 5080 (16GB VRAM)
# ============================================================================
class TrainConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 2048
    
    # MSA Chunking & Memory Settings
    chunk_size: int = 64          # P in MSA paper: intra-document chunk size
    top_k_chunks: int = 4         # k in MSA paper: number of chunks retrieved
    max_seq_len: int = 1024       # Training slice length (extrapolates to 100M at inference)
    
    # RoPE Settings
    rope_base: float = 50000.0    # Long-horizon rotary base frequency
    
    # Cognitive Map & Bayesian Centroids
    map_dim: int = 16
    map_freqs: int = 8
    n_centroids: int = 512
    
    # Loss Coefficients
    ema_decay: float = 0.996
    lambda_jepa: float = 0.25      # Latent predictive regularization
    lambda_kl: float = 0.005       # Bayesian prior regularization
    lambda_balance: float = 0.01   # Centroid load-balancing regularization

    # Batch Sizing: Micro-batch 8 * Seq 1024 * 8 Accum = 65,536 tokens/step
    micro_batch_size: int = 8
    grad_accum_steps: int = 8
    
    total_steps: int = 100000
    warmup_steps: int = 2000
    max_lr: float = 4e-4
    min_lr: float = 4e-5
    weight_decay: float = 0.05
    clip_grad: float = 1.0

    log_interval: int = 25
    save_interval: int = 2500
    checkpoint_dir: str = "./checkpoints_jepa_msa_rope"
    resume_checkpoint: str = ""    # Path to resume, or leave empty to train from step 0


# ============================================================================
# 2. Rotary Position Embedding Engine
# ============================================================================
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) operating on arbitrary position index tensors.
    """
    def __init__(self, dim: int, base: float = 50000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor):
        # positions: [B, T]
        angles = positions.unsqueeze(-1).float() * self.inv_freq.view(1, 1, -1) # [B, T, dim // 2]
        emb = torch.cat([angles, angles], dim=-1)                                # [B, T, dim]
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)                   # [B, 1, T, dim]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, T, dim], cos/sin: [B, 1, T, dim]
    return (x * cos) + (rotate_half(x) * sin)


# ============================================================================
# 3. Modern Primitives (RMSNorm & SwiGLU)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ============================================================================
# 4. Attention Modules with Document-Wise & Global Query RoPE
# ============================================================================
class LocalCausalAttention(nn.Module):
    """
    Lower Layers: Local Attention with Document-Wise RoPE.
    Positions reset every chunk_size (intra-document invariance).
    """
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.chunk_size = cfg.chunk_size

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(dim=self.head_dim, base=cfg.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # Document-wise RoPE: Positions reset every chunk_size (0 ... P-1)
        doc_positions = (torch.arange(T, device=x.device) % self.chunk_size).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope(doc_positions)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class MemorySparseAttention(nn.Module):
    """
    Upper Layers: MSA with Document-Wise RoPE and Global Query RoPE.
    
    1. Document-Wise RoPE: Applied to document chunks before pooling.
    2. Global Query RoPE: Active query positions offset by k (k + pos) so
       retrieved compressed chunks (positions 0 ... k-1) act as causal history.
    """
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.chunk_size = cfg.chunk_size
        self.top_k_chunks = cfg.top_k_chunks

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.router_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.router_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(dim=self.head_dim, base=cfg.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        P = self.chunk_size
        num_chunks = T // P

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        kr = self.router_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        qr = self.router_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Fallback for ultra-short sequences
        if num_chunks <= 1:
            doc_pos = (torch.arange(T, device=x.device) % P).unsqueeze(0).expand(B, -1)
            cos, sin = self.rope(doc_pos)
            out = F.scaled_dot_product_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, is_causal=True)
            return self.proj(out.transpose(1, 2).reshape(B, T, C))

        # --------------------------------------------------------------------
        # 1. Document-Wise RoPE (Section 3.2.2 in Paper)
        # --------------------------------------------------------------------
        doc_pos = (torch.arange(T, device=x.device) % P).unsqueeze(0).expand(B, -1)
        cos_doc, sin_doc = self.rope(doc_pos)
        k_doc = apply_rope(k, cos_doc, sin_doc)

        # --------------------------------------------------------------------
        # 2. Chunk-Wise Mean Pooling (K, V, K^R)
        # --------------------------------------------------------------------
        k_chunks = k_doc[:, :, :num_chunks * P, :].reshape(B, self.n_heads, num_chunks, P, self.head_dim)
        v_chunks = v[:, :, :num_chunks * P, :].reshape(B, self.n_heads, num_chunks, P, self.head_dim)
        kr_chunks = kr[:, :, :num_chunks * P, :].reshape(B, self.n_heads, num_chunks, P, self.head_dim)

        k_bar = k_chunks.mean(dim=3)   # [B, n_heads, num_chunks, head_dim]
        v_bar = v_chunks.mean(dim=3)   # [B, n_heads, num_chunks, head_dim]
        kr_bar = kr_chunks.mean(dim=3) # [B, n_heads, num_chunks, head_dim]

        # --------------------------------------------------------------------
        # 3. Router Cosine Relevance Scoring & Block-Causal Routing
        # --------------------------------------------------------------------
        qr_norm = F.normalize(qr, dim=-1)
        kr_bar_norm = F.normalize(kr_bar, dim=-1)
        sim_scores = torch.einsum("bhtd,bhcd->bhtc", qr_norm, kr_bar_norm).mean(dim=1) # [B, T, num_chunks]

        chunk_indices = torch.arange(num_chunks, device=x.device).view(1, 1, num_chunks)
        token_chunk_pos = (torch.arange(T, device=x.device) // P).view(1, T, 1)
        causal_mask = chunk_indices < token_chunk_pos
        sim_scores = sim_scores.masked_fill(~causal_mask, float("-inf"))

        k_val = min(self.top_k_chunks, num_chunks - 1)
        topk_scores, topk_indices = torch.topk(sim_scores, k=k_val, dim=-1) # [B, T, k]

        # Gather Top-k compressed K and V pairs
        idx_expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_heads, -1, -1, self.head_dim)
        k_bar_expanded = k_bar.unsqueeze(2).expand(-1, -1, T, -1, -1)
        v_bar_expanded = v_bar.unsqueeze(2).expand(-1, -1, T, -1, -1)

        retrieved_k = torch.gather(k_bar_expanded, dim=3, index=idx_expanded) # [B, n_heads, T, k, head_dim]
        retrieved_v = torch.gather(v_bar_expanded, dim=3, index=idx_expanded) # [B, n_heads, T, k, head_dim]

        # --------------------------------------------------------------------
        # 4. Global Query RoPE (Section 3.2.2 in Paper)
        # --------------------------------------------------------------------
        # Retrieved chunks act as memory context at positions 0 ... k_val-1
        mem_pos = torch.arange(k_val, device=x.device).unsqueeze(0).expand(B, -1)
        cos_mem, sin_mem = self.rope(mem_pos) # [B, 1, k_val, head_dim]
        retrieved_k = apply_rope(retrieved_k, cos_mem.unsqueeze(2), sin_mem.unsqueeze(2))

        # Active Query tokens offset by k_val: pos = k_val + (t % P)
        query_pos = (k_val + (torch.arange(T, device=x.device) % P)).unsqueeze(0).expand(B, -1)
        cos_q, sin_q = self.rope(query_pos)
        q_global = apply_rope(q, cos_q, sin_q)
        local_k_global = apply_rope(k, cos_q, sin_q)

        # --------------------------------------------------------------------
        # 5. Sparse Generation Attention
        # --------------------------------------------------------------------
        q_exp = q_global.unsqueeze(3) # [B, n_heads, T, 1, head_dim]
        retrieved_attn = (q_exp * retrieved_k).sum(dim=-1) * (1.0 / math.sqrt(self.head_dim))
        
        valid_retrieved = topk_scores.unsqueeze(1) > float("-inf")
        retrieved_attn = retrieved_attn.masked_fill(~valid_retrieved, float("-inf"))

        local_attn = (q_global * local_k_global).sum(dim=-1, keepdim=True) * (1.0 / math.sqrt(self.head_dim))
        combined_logits = torch.cat([retrieved_attn, local_attn], dim=-1)
        attn_weights = F.softmax(combined_logits, dim=-1)

        retrieved_val = (attn_weights[:, :, :, :k_val].unsqueeze(-1) * retrieved_v).sum(dim=3)
        local_val = attn_weights[:, :, :, -1:].unsqueeze(-1).squeeze(3) * v
        
        out = retrieved_val + local_val
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TrainConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        
        # Paper Section 3.2.1: Lower half = Local, Latter half = MSA Routing
        if layer_idx < (cfg.n_layers // 2):
            self.attn = LocalCausalAttention(cfg)
        else:
            self.attn = MemorySparseAttention(cfg)
            
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 5. Cognitive Map & Bayesian Centroid Modules
# ============================================================================
class DynamicCognitiveMap(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.map_dim = cfg.map_dim
        self.step_net = nn.Linear(cfg.d_model, cfg.map_dim)
        
        freqs = torch.exp(torch.linspace(0, math.log(16.0), cfg.map_freqs))
        self.register_buffer("freqs", freqs, persistent=False)
        
        in_dim = cfg.map_dim * cfg.map_freqs * 2
        self.map_proj = nn.Linear(in_dim, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta_p = torch.tanh(self.step_net(h))
        coords = torch.cumsum(delta_p, dim=1)
        angles = coords.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        angles = angles.flatten(start_dim=-2)
        basis = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.norm(self.map_proj(basis))


class BayesianCentroidMemory(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.n_centroids = cfg.n_centroids
        self.d_model = cfg.d_model
        self.mu = nn.Parameter(torch.randn(cfg.n_centroids, cfg.d_model) * 0.02)
        self.log_var = nn.Parameter(torch.zeros(cfg.n_centroids, cfg.d_model))
        self.scale = 1.0 / math.sqrt(cfg.d_model)

    def compute_centroid_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        B, T, K = routing_weights.shape
        flat_routing = routing_weights.view(-1, K).float()
        
        P_k = flat_routing.mean(dim=0)
        top_indices = flat_routing.argmax(dim=-1)
        assigned = torch.zeros(K, device=routing_weights.device, dtype=torch.float)
        assigned.index_add_(0, top_indices, torch.ones_like(top_indices, dtype=torch.float))
        f_k = (assigned / (B * T)).detach()
        
        return K * torch.sum(f_k * P_k)

    def forward(self, h: torch.Tensor):
        clamped_log_var = torch.clamp(self.log_var, -5.0, 5.0)
        inv_var = torch.exp(-clamped_log_var)

        term1 = torch.matmul(torch.square(h), inv_var.t())
        mu_inv_var = self.mu * inv_var
        term2 = 2.0 * torch.matmul(h, mu_inv_var.t())
        term3 = torch.sum(torch.square(self.mu) * inv_var, dim=-1).view(1, 1, -1)

        weighted_sq_dist = torch.clamp(term1 - term2 + term3, min=0.0)
        log_prob = -0.5 * (weighted_sq_dist + torch.sum(clamped_log_var, dim=-1).view(1, 1, -1))
        routing_weights = F.softmax(log_prob * self.scale, dim=-1)

        memory_readout = torch.matmul(routing_weights, self.mu)

        kl_loss = -0.5 * torch.sum(1 + clamped_log_var - torch.square(self.mu) - torch.exp(clamped_log_var))
        kl_loss = kl_loss / (self.n_centroids * self.d_model)
        balance_loss = self.compute_centroid_balance_loss(routing_weights)

        return memory_readout, kl_loss, balance_loss


# ============================================================================
# 6. Complete Generative-JEPA Architecture
# ============================================================================
class GenerativeJEPALM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        # No absolute pos_embed: handled natively by Document-Wise & Global RoPE
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.norm_backbone = RMSNorm(cfg.d_model)

        self.cognitive_map = DynamicCognitiveMap(cfg)
        self.bayesian_memory = BayesianCentroidMemory(cfg)
        
        self.aux_fusion = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
        self.final_norm = RMSNorm(cfg.d_model)

        self.latent_predictor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.SiLU(),
            nn.Linear(cfg.d_ff, cfg.d_model)
        )

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
        nn.init.zeros_(self.aux_fusion.weight)

        # EMA Target Encoder (No positional lookup table needed)
        self.target_token_embed = copy.deepcopy(self.token_embed)
        self.target_blocks = copy.deepcopy(self.blocks)
        self.target_norm = copy.deepcopy(self.norm_backbone)
        
        for p in self.target_token_embed.parameters(): p.requires_grad = False
        for p in self.target_blocks.parameters(): p.requires_grad = False
        for p in self.target_norm.parameters(): p.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def update_target_encoder(self):
        tau = self.cfg.ema_decay
        for p, tp in zip(self.token_embed.parameters(), self.target_token_embed.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
        for p, tp in zip(self.blocks.parameters(), self.target_blocks.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
        for p, tp in zip(self.norm_backbone.parameters(), self.target_norm.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        x = self.token_embed(input_ids)

        for block in self.blocks:
            x = block(x)
        h = self.norm_backbone(x)

        map_repr = self.cognitive_map(h)
        mem_repr, kl_loss, balance_loss = self.bayesian_memory(h)
        
        aux_context = self.aux_fusion(torch.cat([map_repr, mem_repr], dim=-1))
        fused = self.final_norm(h + aux_context)
        
        logits = self.lm_head(fused)
        logits = 30.0 * torch.tanh(logits / 30.0)

        loss = None
        metrics = {}
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            
            z_pred = self.latent_predictor(fused[:, :-1, :])
            with torch.no_grad():
                xt = self.target_token_embed(input_ids)
                for b in self.target_blocks:
                    xt = b(xt)
                z_target = F.normalize(self.target_norm(xt)[:, 1:, :], dim=-1)

            z_pred = F.normalize(z_pred, dim=-1)
            jepa_loss = 2.0 - 2.0 * (z_pred * z_target).sum(dim=-1).mean()

            loss = (
                ce_loss 
                + (self.cfg.lambda_jepa * jepa_loss) 
                + (self.cfg.lambda_kl * kl_loss) 
                + (self.cfg.lambda_balance * balance_loss)
            )
            metrics = {
                "ce_loss": ce_loss.item(),
                "jepa_loss": jepa_loss.item(),
                "kl_loss": kl_loss.item(),
                "balance_loss": balance_loss.item()
            }

        return logits, loss, metrics


# ============================================================================
# 7. Background Data Streamer
# ============================================================================
class BackgroundPackedStreamer:
    def __init__(self, tokenizer, max_seq_len: int = 1024, max_queue_size: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        domains = [
            load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True),
            load_dataset("HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True),
            load_dataset("HuggingFaceTB/smollm-corpus", name="python-edu", split="train", streaming=True),
        ]
        weights = [0.50, 0.30, 0.20]
        iterators = [iter(d) for d in domains]
        buffer = []
        eos_id = self.tokenizer.eos_token_id

        while not self.stop_event.is_set():
            idx = random.choices([0, 1, 2], weights=weights, k=1)[0]
            try:
                sample = next(iterators[idx])
                text = sample.get("text", "")
            except Exception:
                iterators[idx] = iter(domains[idx])
                continue

            if not text:
                continue

            tokens = self.tokenizer.encode(text) + [eos_id]
            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]
                
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                self.queue.put((input_ids, targets))

    def get_batch(self, batch_size: int, device: torch.device):
        inputs, targets = [], []
        for _ in range(batch_size):
            inp, tgt = self.queue.get()
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs).to(device, non_blocking=True), torch.stack(targets).to(device, non_blocking=True)


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step > cfg.total_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.max_lr - cfg.min_lr)


# ============================================================================
# 8. Training Loop
# ============================================================================
def train():
    cfg = TrainConfig()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Micro-Batch: {cfg.micro_batch_size} | Accum: {cfg.grad_accum_steps}")
    print(f"Tokens Per Step: {cfg.micro_batch_size * cfg.grad_accum_steps * cfg.max_seq_len:,}")
    print(f"Positional Mechanism: Document-Wise RoPE + Global Query RoPE (Extrapolates to 100M)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)
    
    streamer = BackgroundPackedStreamer(tokenizer, max_seq_len=cfg.max_seq_len)
    
    print("Pre-filling streaming buffer...")
    while streamer.queue.qsize() < 32:
        time.sleep(0.1)

    model = GenerativeJEPALM(cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Active Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Total Parameters (inc. EMA): {total_params:,} ({total_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True
    )

    step = 0

    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"\n--> Loading Checkpoint: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"--> Notice: {len(missing)} keys initialized randomly.")
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            step = ckpt.get("step", 0)
            print(f"--> Successfully Resumed from Step {step}!\n")
        except Exception:
            print("--> Optimizer state incompatible with updated architecture; starting optimizer from step 0.\n")
            step = 0
    else:
        print("\n--> Starting Training from Scratch (Step 0)...\n")

    t0 = time.time()
    accum = {"ce": 0.0, "jepa": 0.0, "kl": 0.0, "bal": 0.0}

    model.train()
    optimizer.zero_grad()

    while step < cfg.total_steps:
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for _ in range(cfg.grad_accum_steps):
            input_ids, targets = streamer.get_batch(cfg.micro_batch_size, device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, metrics = model(input_ids, targets=targets)
                loss_scaled = loss / cfg.grad_accum_steps

            loss_scaled.backward()

            accum["ce"] += metrics["ce_loss"] / cfg.grad_accum_steps
            accum["jepa"] += metrics["jepa_loss"] / cfg.grad_accum_steps
            accum["kl"] += metrics["kl_loss"] / cfg.grad_accum_steps
            accum["bal"] += metrics["balance_loss"] / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        model.update_target_encoder()
        step += 1

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = (cfg.micro_batch_size * cfg.grad_accum_steps * cfg.max_seq_len * cfg.log_interval) / dt
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            avg_ce = accum["ce"] / cfg.log_interval
            avg_jepa = accum["jepa"] / cfg.log_interval
            avg_kl = accum["kl"] / cfg.log_interval
            avg_bal = accum["bal"] / cfg.log_interval
            ppl = math.exp(min(avg_ce, 15.0))

            print(
                f"Step {step:6d}/{cfg.total_steps} | "
                f"PPL: {ppl:8.2f} | "
                f"CE: {avg_ce:.4f} | "
                f"JEPA: {avg_jepa:.4f} | "
                f"KL: {avg_kl:.4f} | "
                f"Bal: {avg_bal:.3f} | "
                f"Speed: {tok_per_sec:,.0f} tok/s | "
                f"Peak VRAM: {vram_gb:.2f} GB"
            )
            accum = {"ce": 0.0, "jepa": 0.0, "kl": 0.0, "bal": 0.0}

        if step % cfg.save_interval == 0 or step == cfg.total_steps:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"--> Saved Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()