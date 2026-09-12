import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer

from train_jepa_lm import (
    TrainConfig,
    GenerativeJEPALM
)

class SFTConfig:
    checkpoint_in: str = "./checkpoints_jepa_lm/step_100000.pt"
    checkpoint_out_dir: str = "./checkpoints_sft/3.0/"
    
    total_steps: int = 7500
    warmup_steps: int = 200
    max_lr: float = 5e-5         # 10x lower than pretraining
    min_lr: float = 5e-6
    weight_decay: float = 0.01
    
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_seq_len: int = 1024
    
    lambda_jepa: float = 0.05    # Softened during SFT
    lambda_kl: float = 0.001
    log_interval: int = 25
    save_interval: int = 1000


class SmolTalkSFTDataset(IterableDataset):
    """
    Streams and formats instruction-response pairs with prompt-loss masking.
    User prompt tokens are masked with -100 so loss is ONLY computed on the response.
    """
    def __init__(self, tokenizer, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # smoltalk contains curated daily conversation, reasoning, and python code
        self.dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)

    def __iter__(self):
        for sample in self.dataset:
            messages = sample.get("messages", [])
            if len(messages) < 2:
                continue

            # Construct ChatML string
            formatted_input_ids = []
            formatted_targets = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                header = f"<|im_start|>{role}\n"
                body = f"{content}<|im_end|>\n"
                
                header_tokens = self.tokenizer.encode(header)
                body_tokens = self.tokenizer.encode(body)
                
                tokens = header_tokens + body_tokens
                formatted_input_ids.extend(tokens)
                
                if role == "assistant":
                    # Only train on assistant responses
                    formatted_targets.extend([-100] * len(header_tokens) + body_tokens)
                else:
                    # Mask user/system tokens
                    formatted_targets.extend([-100] * len(tokens))

            if len(formatted_input_ids) < 16:
                continue

            # Truncate or pad to max_seq_len
            if len(formatted_input_ids) > self.max_seq_len:
                input_ids = formatted_input_ids[:self.max_seq_len]
                targets = formatted_targets[1:self.max_seq_len + 1]
                if len(targets) < len(input_ids):
                    targets.extend([-100] * (len(input_ids) - len(targets)))
            else:
                pad_len = self.max_seq_len - len(formatted_input_ids)
                input_ids = formatted_input_ids + [self.tokenizer.eos_token_id] * pad_len
                targets = formatted_targets[1:] + [-100] * (pad_len + 1)
                targets = targets[:self.max_seq_len]

            yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def get_sft_lr(step: int, cfg: SFTConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    ratio = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.max_lr - cfg.min_lr)


def train_sft():
    cfg = SFTConfig()
    os.makedirs(cfg.checkpoint_out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base checkpoint: {cfg.checkpoint_in} ...")
    checkpoint_data = torch.load(cfg.checkpoint_in, map_location=device, weights_only=False)
    base_cfg = checkpoint_data.get("config", TrainConfig())
    base_cfg.lambda_jepa = cfg.lambda_jepa
    base_cfg.lambda_kl = cfg.lambda_kl

    model = GenerativeJEPALM(base_cfg).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    
    # Differential LR: Freeze centroids or update them with 0.1x LR
    centroid_params = [model.bayesian_memory.mu, model.bayesian_memory.log_var]
    backbone_params = [p for n, p in model.named_parameters() if "bayesian_memory" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.max_lr},
        {"params": centroid_params, "lr": cfg.max_lr * 0.05} # 10x slower on centroids
    ], betas=(0.9, 0.95), weight_decay=cfg.weight_decay, fused=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SmolTalkSFTDataset(tokenizer, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.micro_batch_size)
    loader_iter = iter(loader)

    print("=" * 65)
    print("Starting SFT Instruction Tuning on smoltalk")
    print(f"Total Steps: {cfg.total_steps} | Peak LR: {cfg.max_lr} | Prompt Masking: ENABLED")
    print("=" * 65)

    step = 0
    t0 = time.time()
    accum_ce, accum_jepa = 0.0, 0.0
    model.train()
    optimizer.zero_grad()

    while step < cfg.total_steps:
        lr = get_sft_lr(step, cfg)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * 0.1

        for _ in range(cfg.grad_accum_steps):
            try:
                input_ids, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                input_ids, targets = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Standard model forward pass
                logits, loss, metrics = model(input_ids, targets=targets)
                loss_scaled = loss / cfg.grad_accum_steps

            loss_scaled.backward()
            accum_ce += metrics["ce_loss"] / cfg.grad_accum_steps
            accum_jepa += metrics["jepa_loss"] / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        model.update_target_encoder()
        step += 1

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_ce = accum_ce / cfg.log_interval
            avg_jepa = accum_jepa / cfg.log_interval
            ppl = math.exp(min(avg_ce, 15.0))

            print(f"SFT Step {step:5d}/{cfg.total_steps} | Response PPL: {ppl:6.2f} | CE: {avg_ce:.4f} | JEPA: {avg_jepa:.4f} | LR: {lr:.2e}")
            accum_ce, accum_jepa = 0.0, 0.0

        if step % cfg.save_interval == 0 or step == cfg.total_steps:
            save_path = os.path.join(cfg.checkpoint_out_dir, f"sft_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": base_cfg,
            }, save_path)
            print(f"--> Saved SFT Checkpoint: {save_path}")

    print("\nFine-tuning completed successfully!")


if __name__ == "__main__":
    train_sft()