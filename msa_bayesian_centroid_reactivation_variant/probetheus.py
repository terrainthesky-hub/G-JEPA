import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import the RoPE architecture directly from your training script
try:
    from train_jepa_msa_rope import GenerativeJEPALM, TrainConfig
except ImportError:
    try:
        from train_jepa_msa_lm import GenerativeJEPALM, TrainConfig
    except ImportError:
        from train_jepa_lm import GenerativeJEPALM, TrainConfig


# ============================================================================
# 1. Text Generation Probe (Unconstrained by Embedding Tables)
# ============================================================================
@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_p: float = 0.9, device: str = "cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_new_tokens):
        # With Document-Wise RoPE, we don't strictly need to truncate, but we keep 
        # a reasonable sliding window if generating ultra-long sequences
        idx_cond = input_ids if input_ids.size(1) <= 4096 else input_ids[:, -4096:]
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _, _ = model(idx_cond)
            
        logits = logits[:, -1, :] / max(temperature, 1e-4)
        
        # Top-p (Nucleus) Filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ============================================================================
# 2. Bayesian Centroid Uncertainty & MoE Load-Balance Audit
# ============================================================================
@torch.no_grad()
def audit_bayesian_centroids(model, sample_tokens: torch.Tensor):
    """
    Analyzes learned variance (uncertainty spread) and empirical routing balance.
    """
    model.eval()
    memory_mod = model.bayesian_memory
    K, D = memory_mod.n_centroids, memory_mod.d_model
    
    # 1. Parameter Inspection
    mu = memory_mod.mu.detach()
    clamped_log_var = torch.clamp(memory_mod.log_var.detach(), -5.0, 5.0)
    var = torch.exp(clamped_log_var)
    centroid_vars = var.mean(dim=-1)
    sorted_vars, sorted_var_idx = torch.sort(centroid_vars)
    
    # 2. Forward Hook to capture latent state h before memory injection
    activations = {}
    def hook_fn(module, inp, out):
        activations["h"] = out
    handle = model.norm_backbone.register_forward_hook(hook_fn)
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(sample_tokens)
    handle.remove()
    
    h = activations["h"]
    B, T, _ = h.shape
    
    # Recompute routing weights
    inv_var = torch.exp(-clamped_log_var)
    term1 = torch.matmul(torch.square(h), inv_var.t())
    mu_inv_var = mu * inv_var
    term2 = 2.0 * torch.matmul(h, mu_inv_var.t())
    term3 = torch.sum(torch.square(mu) * inv_var, dim=-1).view(1, 1, -1)
    
    weighted_sq_dist = torch.clamp(term1 - term2 + term3, min=0.0)
    log_prob = -0.5 * (weighted_sq_dist + torch.sum(clamped_log_var, dim=-1).view(1, 1, -1))
    routing_weights = F.softmax(log_prob * memory_mod.scale, dim=-1) # [B, T, K]
    
    # 3. Compute Load-Balancing Loss
    balance_loss = memory_mod.compute_centroid_balance_loss(routing_weights).item()
    
    # 4. Token Assignment Distribution
    flat_routing = routing_weights.view(-1, K)
    top_assignments = flat_routing.argmax(dim=-1)
    counts = torch.bincount(top_assignments, minlength=K).float()
    
    active_centroids = (counts > 0).sum().item()
    dead_centroids = K - active_centroids
    
    probs = counts / counts.sum()
    valid_probs = probs[probs > 0]
    entropy = -torch.sum(valid_probs * torch.log(valid_probs)).item()
    max_entropy = math.log(K)
    effective_k = math.exp(entropy)
    
    return {
        "total_k": K,
        "active_k": active_centroids,
        "dead_k": dead_centroids,
        "effective_k": effective_k,
        "entropy_ratio": (entropy / max_entropy) * 100,
        "balance_loss": balance_loss,
        "min_var": sorted_vars[0].item(),
        "median_var": sorted_vars[K // 2].item(),
        "max_var": sorted_vars[-1].item(),
        "top_confident_idx": sorted_var_idx[:3].tolist(),
        "top_uncertain_idx": sorted_var_idx[-3:].tolist()
    }


# ============================================================================
# 3. Memory Sparse Attention & KV Cache Efficiency
# ============================================================================
def audit_msa_efficiency(cfg: TrainConfig, eval_context_len: int = 1024):
    """
    Measures working memory footprint and parameter efficiency with RoPE.
    """
    T = eval_context_len
    P = cfg.chunk_size          # 64
    k_retrieved = cfg.top_k_chunks # 4
    n_msa_layers = cfg.n_layers // 2 # Top 6 layers
    
    dense_active_kv = T
    msa_active_kv = k_retrieved + 1
    reduction_pct = (1.0 - (msa_active_kv / dense_active_kv)) * 100
    
    kv_per_token_bytes = 2 * cfg.d_model * 2  # K and V in BF16
    dense_mb = (dense_active_kv * kv_per_token_bytes * n_msa_layers) / (1024 ** 2)
    msa_mb = (msa_active_kv * kv_per_token_bytes * n_msa_layers) / (1024 ** 2)
    
    # Positional Embedding savings
    saved_pos_params = cfg.max_seq_len * cfg.d_model
    
    return {
        "context_length": T,
        "dense_active_tokens": dense_active_kv,
        "msa_active_tokens": msa_active_kv,
        "reduction_pct": reduction_pct,
        "dense_mb": dense_mb,
        "msa_mb": msa_mb,
        "saved_pos_params": saved_pos_params
    }


# ============================================================================
# 4. Context Degradation & Length Extrapolation (NIAH Probe)
# ============================================================================
@torch.no_grad()
def audit_needle_retrieval(model, tokenizer, total_len: int = 1000, device: str = "cuda"):
    """
    Tests recall of a buried numerical fact across varying context depths and sequence scales.
    """
    model.eval()
    needle = " The secret access code is 84920."
    query = " What is the secret access code? The secret access code is"
    target_token_id = tokenizer.encode(" 84920")[0]
    
    filler_sentence = " The solar system consists of planets orbiting the star. Galaxies populate the observable universe."
    filler_ids = tokenizer.encode(filler_sentence)
    
    needle_ids = tokenizer.encode(needle)
    query_ids = tokenizer.encode(query)
    
    depths = [0.10, 0.50, 0.90]
    results = {}
    
    avail_len = total_len - len(needle_ids) - len(query_ids)
    if avail_len <= 0:
        return {}

    for depth in depths:
        prefix_len = int(avail_len * depth)
        suffix_len = avail_len - prefix_len
        
        prefix = (filler_ids * (prefix_len // len(filler_ids) + 1))[:prefix_len]
        suffix = (filler_ids * (suffix_len // len(filler_ids) + 1))[:suffix_len]
        
        full_ids = torch.tensor([prefix + needle_ids + suffix + query_ids], device=device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _, _ = model(full_ids)
            
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        pred_token_id = torch.argmax(probs).item()
        
        target_prob = probs[target_token_id].item() * 100
        rank = (torch.argsort(next_token_logits, descending=True) == target_token_id).nonzero().item() + 1
        
        results[f"Depth {int(depth*100)}%"] = {
            "predicted_token": tokenizer.decode([pred_token_id]),
            "target_probability": f"{target_prob:.2f}%",
            "target_rank": rank
        }
        
    return results


# ============================================================================
# 5. Main Diagnostic Runner
# ============================================================================
def run_diagnostics():
    parser = argparse.ArgumentParser(description="RoPE-MSA-JEPA Diagnostic Probe")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_jepa_msa_rope/step_15000.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence and memory sparse attention allow models to", help="Test prompt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "=" * 80)
    print(f"       DIAGNOSTIC AUDIT REPORT: {os.path.basename(args.checkpoint)}")
    print("=" * 80)

    if not os.path.exists(args.checkpoint):
        print(f"[!] Checkpoint '{args.checkpoint}' not found.")
        print(f"[!] Please pass a valid path via --checkpoint <path>")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", TrainConfig())
    step = ckpt.get("step", "Unknown")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e9)
    
    model = GenerativeJEPALM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Checkpoint Step: {step} | Active Parameters: ~{trainable_params / 1e6:.1f}M")
    print(f"Positional Mechanism: Document-Wise RoPE + Global Query RoPE (Zero Absolute Embedding Tables)")

    # ------------------------------------------------------------------------
    # Probe 1: Bayesian Centroid Uncertainty & MoE Load Balance
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[1] BAYESIAN CENTROID UNCERTAINTY & LOAD-BALANCING DYNAMICS")
    print("-" * 80)
    
    sample_eval_batch = torch.randint(0, cfg.vocab_size, (4, 1024), device=device)
    centroid_stats = audit_bayesian_centroids(model, sample_eval_batch)
    
    print(f"  • Total Prototypes (K):         {centroid_stats['total_k']}")
    print(f"  • Active Centroids (>0 hits):   {centroid_stats['active_k']} / {centroid_stats['total_k']} ({centroid_stats['active_k']/centroid_stats['total_k']*100:.1f}%)")
    print(f"  • Dead Centroids (0 hits):     {centroid_stats['dead_k']}")
    print(f"  • MoE Balance Loss (Target ~1): {centroid_stats['balance_loss']:.4f}")
    print(f"  • Effective Diversity (exp(H)): {centroid_stats['effective_k']:.1f} / {centroid_stats['total_k']} ({centroid_stats['entropy_ratio']:.1f}% of uniform)")
    print(f"  • Learned Variance Spread:      Min: {centroid_stats['min_var']:.3f} | Median: {centroid_stats['median_var']:.3f} | Max: {centroid_stats['max_var']:.3f}")
    print(f"  • Top-3 Confident Centroids:    IDs: {centroid_stats['top_confident_idx']} (Sharp semantic anchors)")
    print(f"  • Top-3 Uncertain Centroids:    IDs: {centroid_stats['top_uncertain_idx']} (Broad / noise-absorbing)")

    # ------------------------------------------------------------------------
    # Probe 2: Memory Sparse Attention (MSA) Efficiency
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[2] MSA SPARSITY & KV CACHE FOOTPRINT REDUCTION")
    print("-" * 80)
    
    msa_stats = audit_msa_efficiency(cfg, eval_context_len=1024)
    print(f"  • Evaluated Context Length:     {msa_stats['context_length']} tokens")
    print(f"  • Active Attention Tokens:      Dense: {msa_stats['dense_active_tokens']}  ==>  MSA: {msa_stats['msa_active_tokens']}")
    print(f"  • Working Cache Reduction:      {msa_stats['reduction_pct']:.1f}% reduction in active KV footprint")
    print(f"  • MSA Active Working VRAM:      {msa_stats['msa_mb']:.2f} MB (vs {msa_stats['dense_mb']:.2f} MB for dense layers)")
    print(f"  • Positional Memory Saved:      {msa_stats['saved_pos_params']:,} parameters eliminated via RoPE")

    # ------------------------------------------------------------------------
    # Probe 3: Context Degradation & Length Extrapolation (NIAH Probe)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3] CONTEXT DEGRADATION & NEEDLE RETRIEVAL (NIAH)")
    print("-" * 80)
    
    # Test A: In-Distribution Length (1000 tokens)
    print("  [A] In-Context Test (1,000 tokens):")
    niah_1k = audit_needle_retrieval(model, tokenizer, total_len=1000, device=device)
    for depth, res in niah_1k.items():
        status = "PASSED" if res["target_rank"] == 1 else f"Rank #{res['target_rank']}"
        print(f"      • {depth:10s} | Status: {status:10s} | Pred: '{res['predicted_token']}' | Target Prob: {res['target_probability']}")

    # Test B: Extrapolated Length (2000 tokens - enabled by Document-Wise RoPE)
    print("\n  [B] Extrapolation Test (2,000 tokens - 2× Training Context):")
    niah_2k = audit_needle_retrieval(model, tokenizer, total_len=2000, device=device)
    for depth, res in niah_2k.items():
        status = "PASSED" if res["target_rank"] == 1 else f"Rank #{res['target_rank']}"
        print(f"      • {depth:10s} | Status: {status:10s} | Pred: '{res['predicted_token']}' | Target Prob: {res['target_probability']}")

    # ------------------------------------------------------------------------
    # Probe 4: Qualitative Text Generation
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[4] QUALITATIVE TEXT GENERATION")
    print("-" * 80)
    print(f"Prompt: \"{args.prompt}\"")
    generated = generate_text(model, tokenizer, prompt=args.prompt, max_new_tokens=40, temperature=0.7, device=device)
    print(f"Output: \"{generated}\"")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_diagnostics()