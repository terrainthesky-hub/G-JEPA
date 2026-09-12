import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
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

class CoTConfig:
    checkpoint_in: str = "./checkpoints_sft/sft_step_5000.pt"
    checkpoint_out_dir: str = "./checkpoints_cot_sft"
    
    total_steps: int = 3000
    warmup_steps: int = 150
    max_lr: float = 4e-5         # Low LR to protect pretrained weights
    min_lr: float = 4e-6
    weight_decay: float = 0.01
    
    micro_batch_size: int = 4
    grad_accum_steps: int = 8    # 4 * 1024 * 8 = 32,768 tokens/step
    max_seq_len: int = 1024
    
    # In CoT, JEPA helps anticipate the next deduction step
    lambda_jepa: float = 0.10    
    lambda_kl: float = 0.001
    
    log_interval: int = 25
    save_interval: int = 1000


# ============================================================================
# CoT Dataset Engine with <think> Tag Injection
# ============================================================================
class CompactCoTDataset(IterableDataset):
    """
    Streams GSM8K and SmolTalk reasoning data, structuring solutions into:
    <|im_start|>user\n{question}<|im_end|>\n
    <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n{answer}<|im_end|>
    
    Masks user tokens with -100 so the model only learns to think and answer.
    """
    def __init__(self, tokenizer, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 1. GSM8K Grade-School Math
        self.gsm8k = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        # 2. SmolTalk Reasoning Split
        self.smoltalk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)

    def __iter__(self):
        iter_gsm = iter(self.gsm8k)
        iter_smol = iter(self.smoltalk)

        while True:
            # 60% GSM8k Math, 40% SmolTalk general logic
            if random.random() < 0.60:
                try:
                    sample = next(iter_gsm)
                    question = sample["question"]
                    # GSM8K separates reasoning from final answer with '####'
                    parts = sample["answer"].split("####")
                    if len(parts) == 2:
                        reasoning = parts[0].strip()
                        final_ans = f"The final answer is {parts[1].strip()}."
                    else:
                        reasoning = parts[0].strip()
                        final_ans = ""
                except Exception:
                    iter_gsm = iter(self.gsm8k)
                    continue
            else:
                try:
                    sample = next(iter_smol)
                    msgs = sample.get("messages", [])
                    if len(msgs) < 2: continue
                    question = msgs[0]["content"]
                    full_resp = msgs[1]["content"]
                    
                    # If smoltalk response has step-by-step, wrap first half as think
                    lines = full_resp.split("\n")
                    if len(lines) > 3:
                        reasoning = "\n".join(lines[:-1]).strip()
                        final_ans = lines[-1].strip()
                    else:
                        reasoning = "Analyze the request and apply logical rules."
                        final_ans = full_resp
                except Exception:
                    iter_smol = iter(self.smoltalk)
                    continue

            # Format into structured CoT ChatML
            prompt_str = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            response_str = f"<think>\n{reasoning}\n</think>\n{final_ans}<|im_end|>\n"

            prompt_tokens = self.tokenizer.encode(prompt_str)
            response_tokens = self.tokenizer.encode(response_str)

            total_tokens = prompt_tokens + response_tokens
            if len(total_tokens) > self.max_seq_len:
                continue  # Skip samples that exceed 1024 to avoid cutting off reasoning

            # Mask prompt tokens (-100), compute loss on think + answer
            input_ids = total_tokens
            targets = [-100] * len(prompt_tokens) + response_tokens

            # Pad up to max_seq_len
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.eos_token_id] * pad_len
            targets = targets[1:] + [-100] * (pad_len + 1)
            targets = targets[:self.max_seq_len]

            yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def get_lr(step: int, cfg: CoTConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    ratio = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.max_lr - cfg.min_lr)


# ============================================================================
# Main CoT Fine-Tuning Execution
# ============================================================================
def train_cot():
    cfg = CoTConfig()
    os.makedirs(cfg.checkpoint_out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Base Checkpoint: {cfg.checkpoint_in} ...")
    checkpoint_data = torch.load(cfg.checkpoint_in, map_location=device, weights_only=False)
    
    base_cfg = checkpoint_data.get("config", TrainConfig())
    base_cfg.lambda_jepa = cfg.lambda_jepa
    base_cfg.lambda_kl = cfg.lambda_kl

    model = GenerativeJEPALM(base_cfg).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    
    # Differential learning rate: Update centroids gently (0.1x LR)
    centroid_params = [model.bayesian_memory.mu, model.bayesian_memory.log_var]
    backbone_params = [p for n, p in model.named_parameters() if "bayesian_memory" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.max_lr},
        {"params": centroid_params, "lr": cfg.max_lr * 0.1}
    ], betas=(0.9, 0.95), weight_decay=cfg.weight_decay, fused=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = CompactCoTDataset(tokenizer, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.micro_batch_size)
    loader_iter = iter(loader)

    print("=" * 70)
    print("🧠 Starting Chain-of-Thought (CoT) Fine-Tuning")
    print(f"Target Sequence Length: {cfg.max_seq_len} | Steps: {cfg.total_steps}")
    print("Formatting: <think> ... </think> reasoning tags with Prompt Masking")
    print("=" * 70)

    step = 0
    t0 = time.time()
    accum_ce, accum_jepa = 0.0, 0.0
    model.train()
    optimizer.zero_grad()

    while step < cfg.total_steps:
        lr = get_lr(step, cfg)
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

            print(f"CoT Step {step:4d}/{cfg.total_steps} | Response PPL: {ppl:6.2f} | CE: {avg_ce:.4f} | JEPA: {avg_jepa:.4f} | LR: {lr:.2e}")
            accum_ce, accum_jepa = 0.0, 0.0

        if step % cfg.save_interval == 0 or step == cfg.total_steps:
            save_path = os.path.join(cfg.checkpoint_out_dir, f"cot_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": base_cfg,
            }, save_path)
            print(f"--> Saved CoT Checkpoint: {save_path}")

    print("\nChain-of-Thought Fine-Tuning Completed Successfully!")


if __name__ == "__main__":
    train_cot()