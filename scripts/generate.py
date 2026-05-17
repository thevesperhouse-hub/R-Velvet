"""
Generate text from a trained R-Velvet checkpoint.

Usage:
    # Interactive mode
    python scripts/generate.py --model 1_3b --checkpoint outputs/phase1_fresh/ckpt_step16000.pt

    # Single prompt
    python scripts/generate.py --model 1_3b --checkpoint outputs/phase1_fresh/ckpt_step16000.pt \
        --prompt "La physique quantique est"

    # With sampling params
    python scripts/generate.py --model 1_3b --checkpoint outputs/phase1_fresh/ckpt_step16000.pt \
        --prompt "Le soleil" --max-tokens 200 --temperature 0.8 --top-k 50 --top-p 0.9
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

from rvelvet.model import RVelvet


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model(mcfg: dict) -> RVelvet:
    return RVelvet(
        vocab_size=mcfg['vocab_size'],
        d_model=mcfg['d_model'],
        n_local_layers=mcfg['n_local_layers'],
        n_global_layers=mcfg['n_global_layers'],
        n_local_heads=mcfg['n_local_heads'],
        n_global_heads=mcfg['n_global_heads'],
        window_size=mcfg['window_size'],
        segment_size=mcfg['segment_size'],
        n_concepts=mcfg['n_concepts'],
        n_refine_layers=mcfg['n_refine_layers'],
        memory_size=mcfg['memory_size'],
        n_read_steps=mcfg['n_read_steps'],
        ffn_mult=mcfg['ffn_mult'],
        dropout=0.0,
        max_seq_len=mcfg['max_seq_len'],
        use_acr=False,
        use_iterative_reasoning=False,
        max_reasoning_iterations=mcfg.get('max_reasoning_iterations', 8),
        lora_rank=mcfg.get('lora_rank', 8),
        halt_threshold=mcfg.get('halt_threshold', 0.5),
        lambda_p=mcfg.get('lambda_p', 0.5),
    )


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """Filter logits with top-k and/or top-p (nucleus) sampling."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')

    return logits


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=100, temperature=0.8,
             top_k=50, top_p=0.9, eos_token_id=0, repetition_penalty=1.2):
    """Autoregressive generation with sampling."""
    device = input_ids.device
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate to max_seq_len if needed
        context = generated[:, -model.max_seq_len:]

        output = model(context, causal=True)
        logits = output['logits'][:, -1, :]  # (batch, vocab)

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Temperature
        if temperature > 0:
            logits = logits / temperature
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == eos_token_id:
            break

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from R-Velvet checkpoint")
    parser.add_argument("--model", default="1_3b", help="Model config name")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", default="data/velvet_tok_100k_unigram",
                        help="Tokenizer path")
    parser.add_argument("--prompt", default=None, help="Text prompt (interactive if omitted)")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    # Load model config
    config_path = f"configs/model/{args.model}.yaml"
    mcfg = load_yaml(config_path)
    print(f"Model: {args.model} (d={mcfg['d_model']}, layers={mcfg['n_global_layers']})")

    # Build model
    model = build_model(mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
        print("  Stripped _orig_mod. prefix")
    model.load_state_dict(state, strict=True)
    print(f"  Loaded from step {ckpt.get('step', '?')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id or 0
    print(f"Tokenizer: vocab={tokenizer.vocab_size}, eos={eos_id}")
    print("-" * 60)

    def run_prompt(prompt_text):
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        print(f"\n[Prompt] ({input_ids.shape[1]} tokens)")
        print(prompt_text, end="", flush=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
            output_ids = generate(
                model, input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_id,
                repetition_penalty=args.repetition_penalty,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(generated_text)
        print(f"\n[Generated {len(new_tokens)} tokens]")

    if args.prompt:
        run_prompt(args.prompt)
    else:
        print("Interactive mode (type 'quit' to exit)\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ('quit', 'exit', 'q'):
                break
            if prompt.strip():
                run_prompt(prompt.strip())


if __name__ == "__main__":
    main()
