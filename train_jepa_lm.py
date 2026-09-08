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
# 1. Config: Tuned for ~10-11 GB Peak VRAM on RTX 5080
# ============================================================================
class TrainConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 1024
    window_size: int = 512
    
    map_dim: int = 16
    map_freqs: int = 8
    n_centroids: int = 512
    
    ema_decay: float = 0.996
    lambda_jepa: float = 0.25      # Balanced latent regularizer
    lambda_kl: float = 0.005

    # Target: 65,536 tokens per optimizer step
    # Micro-batch 8 * Seq 1024 * 8 Accum = 65,536 tokens/step
    micro_batch_size: int = 8      # UPGRADED from 2 to 8 (Saturates Tensor Cores)
    grad_accum_steps: int = 8      # REDUCED from 32 to 8
    
    total_steps: int = 100000
    warmup_steps: int = 2000
    max_lr: float = 4e-4
    min_lr: float = 4e-5
    weight_decay: float = 0.05
    clip_grad: float = 1.0

    log_interval: int = 25
    save_interval: int = 5000
    checkpoint_dir: str = "./checkpoints_jepa_lm"
    resume_checkpoint: str = "./checkpoints_jepa_lm/step_85000.pt"

# ============================================================================
# 2. Modern Primitives
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


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.window_size = cfg.window_size

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        seq = torch.arange(cfg.max_seq_len)
        i = seq.unsqueeze(1)
        j = seq.unsqueeze(0)
        valid = (j <= i) & (j >= (i - cfg.window_size))
        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf"))
        mask[valid] = 0.0
        self.register_buffer("sliding_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = self.sliding_mask[:, :, :T, :T]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(attn.transpose(1, 2).reshape(B, T, C))


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

        return memory_readout, kl_loss


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = SlidingWindowAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 3. Stabilized Architecture (Residual Fusion + Final RMSNorm)
# ============================================================================
class GenerativeJEPALM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_backbone = RMSNorm(cfg.d_model)

        self.cognitive_map = DynamicCognitiveMap(cfg)
        self.bayesian_memory = BayesianCentroidMemory(cfg)
        
        # Zero-initialized residual injection
        self.aux_fusion = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)
        self.final_norm = RMSNorm(cfg.d_model)

        self.latent_predictor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.SiLU(),
            nn.Linear(cfg.d_ff, cfg.d_model)
        )

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        # 1. Apply proper LLM weight initialization (fixes CE = 500)
        self.apply(self._init_weights)

        # Scale residual projections (GPT-2 / Megatron standard)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
        nn.init.zeros_(self.aux_fusion.weight)

        # 2. Initialize EMA Target Encoder AFTER weight init
        self.target_token_embed = copy.deepcopy(self.token_embed)
        self.target_pos_embed = copy.deepcopy(self.pos_embed)
        self.target_blocks = copy.deepcopy(self.blocks)
        self.target_norm = copy.deepcopy(self.norm_backbone)
        
        for p in self.target_token_embed.parameters(): p.requires_grad = False
        for p in self.target_pos_embed.parameters(): p.requires_grad = False
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
        for p, tp in zip(self.pos_embed.parameters(), self.target_pos_embed.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
        for p, tp in zip(self.blocks.parameters(), self.target_blocks.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
        for p, tp in zip(self.norm_backbone.parameters(), self.target_norm.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x)
        h = self.norm_backbone(x)

        map_repr = self.cognitive_map(h)
        mem_repr, kl_loss = self.bayesian_memory(h)
        
        aux_context = self.aux_fusion(torch.cat([map_repr, mem_repr], dim=-1))
        fused = self.final_norm(h + aux_context)
        
        # Compute raw logits
        logits = self.lm_head(fused)
        
        # Soft-capping: bounds logit dynamic range safely within [-30, 30]
        logits = 30.0 * torch.tanh(logits / 30.0)

        loss = None
        metrics = {}
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            
            z_pred = self.latent_predictor(fused[:, :-1, :])
            with torch.no_grad():
                xt = self.target_token_embed(input_ids) + self.target_pos_embed(pos)
                for b in self.target_blocks:
                    xt = b(xt)
                z_target = F.normalize(self.target_norm(xt)[:, 1:, :], dim=-1)

            z_pred = F.normalize(z_pred, dim=-1)
            jepa_loss = 2.0 - 2.0 * (z_pred * z_target).sum(dim=-1).mean()

            loss = ce_loss + (self.cfg.lambda_jepa * jepa_loss) + (self.cfg.lambda_kl * kl_loss)
            metrics = {
                "ce_loss": ce_loss.item(),
                "jepa_loss": jepa_loss.item(),
                "kl_loss": kl_loss.item()
            }

        return logits, loss, metrics


# ============================================================================
# 4. Background Streamer with Expanded Queue
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
# 5. Training Loop with Fixed Averaging & Metrics
# ============================================================================
def train():
    cfg = TrainConfig()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Micro-Batch Size: {cfg.micro_batch_size} | Grad Accum Steps: {cfg.grad_accum_steps}")
    print(f"Tokens Per Optimizer Step: {cfg.micro_batch_size * cfg.grad_accum_steps * cfg.max_seq_len:,}")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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

    # ========================================================================
    # Resume from Checkpoint
    # ========================================================================
    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"\n--> Loading Checkpoint: {cfg.resume_checkpoint}")
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)
        
        # 1. Restore Model (restores BOTH context backbone and EMA target encoder)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Restore Optimizer Momentum States
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Ensure optimizer tensor states are on GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        # 3. Restore Step Counter
        step = checkpoint.get("step", 85000)
        print(f"--> Successfully Resumed from Step {step}!\n")
    else:
        print("\n--> Starting Training from Scratch (Step 0)...\n")

    t0 = time.time()
    accum = {"ce": 0.0, "jepa": 0.0, "kl": 0.0}

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        model.update_target_encoder()
        step += 1

        # Corrected Logging (Divided by log_interval)
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = (cfg.micro_batch_size * cfg.grad_accum_steps * cfg.max_seq_len * cfg.log_interval) / dt
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            avg_ce = accum["ce"] / cfg.log_interval
            avg_jepa = accum["jepa"] / cfg.log_interval
            avg_kl = accum["kl"] / cfg.log_interval
            ppl = math.exp(min(avg_ce, 15.0))

            print(
                f"Step {step:6d}/{cfg.total_steps} | "
                f"PPL: {ppl:8.2f} | "
                f"CE: {avg_ce:.4f} | "
                f"JEPA: {avg_jepa:.4f} | "
                f"KL: {avg_kl:.4f} | "
                f"Speed: {tok_per_sec:,.0f} tok/s | "
                f"Peak VRAM: {vram_gb:.2f} GB"
            )
            accum = {"ce": 0.0, "jepa": 0.0, "kl": 0.0}

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