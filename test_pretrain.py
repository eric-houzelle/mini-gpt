"""
Test du modèle pré-entraîné MiniGPT.

Usage:
    python test_pretrain.py
    python test_pretrain.py --checkpoint checkpoints/best_miniGPT.pt
    python test_pretrain.py --temperature 0.3 --max_tokens 100
    python test_pretrain.py --recurrent-steps 16
    python test_pretrain.py --interactive
"""

import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.configuration import MiniGPTConfig
from model.modeling_minigpt import MiniGPTForCausalLM


def infer_config_from_checkpoint(state_dict):
    """Infer model hyperparameters directly from checkpoint weight shapes."""
    embed_dim = state_dict["model.ln_f.weight"].shape[0]
    vocab_size = state_dict["model.token_emb.weight"].shape[0]
    hidden_dim = state_dict["model.prelude.0.ff.w.weight"].shape[0]
    q_dim = state_dict["model.prelude.0.attn.q_proj.weight"].shape[0]
    k_dim = state_dict["model.prelude.0.attn.k_proj.weight"].shape[0]

    # q_proj: (num_heads * head_dim, embed_dim), k_proj: (num_kv_heads * head_dim, embed_dim)
    # Try standard head_dim values (64 is most common in modern LLMs)
    num_heads, num_kv_heads = None, None
    for head_dim in [64, 128, 96, 80, 48, 32, 256]:
        if q_dim % head_dim == 0 and k_dim % head_dim == 0 and embed_dim % head_dim == 0:
            nh = q_dim // head_dim
            nkv = k_dim // head_dim
            if nh == embed_dim // head_dim:
                num_heads = nh
                num_kv_heads = nkv
                break
    if num_heads is None:
        num_heads = q_dim // (embed_dim // 16)
        num_kv_heads = k_dim // (embed_dim // 16)

    num_prelude = 0
    while f"model.prelude.{num_prelude}.attn.q_proj.weight" in state_dict:
        num_prelude += 1

    num_coda = 0
    while f"model.coda.{num_coda}.attn.q_proj.weight" in state_dict:
        num_coda += 1

    num_loras = 0
    while f"model.depth_loras.{num_loras}.down.weight" in state_dict:
        num_loras += 1

    lora_rank = state_dict["model.depth_loras.0.down.weight"].shape[0] if num_loras > 0 else 0

    use_lti = "model.lti.a_logit" in state_dict
    use_act = "model.act.halt_proj.weight" in state_dict
    has_recurrent = "model.recurrent_block.attn.q_proj.weight" in state_dict
    depth = num_prelude + num_coda + (1 if has_recurrent else 0)

    return {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "depth": depth,
        "num_prelude_layers": num_prelude,
        "num_coda_layers": num_coda,
        "num_recurrent_steps": num_loras if num_loras > 0 else 8,
        "use_lti_injection": use_lti,
        "use_act_halting": use_act,
        "depth_lora_rank": lora_rank,
        "weight_sharing": "recurrent_depth" if has_recurrent else "none",
    }


def build_model(config_path, checkpoint_path, device, recurrent_steps=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    inferred = infer_config_from_checkpoint(state)
    print(f"Config inférée du checkpoint:")
    print(f"  embed_dim={inferred['embed_dim']}, hidden_dim={inferred['hidden_dim']}, "
          f"heads={inferred['heads']}, num_kv_heads={inferred['num_kv_heads']}")
    print(f"  prelude={inferred['num_prelude_layers']}, coda={inferred['num_coda_layers']}, "
          f"lora_rank={inferred['depth_lora_rank']}, recurrent_steps={inferred['num_recurrent_steps']}")
    print(f"  LTI={inferred['use_lti_injection']}, ACT={inferred['use_act_halting']}, "
          f"weight_sharing={inferred['weight_sharing']}")

    with open(config_path) as f:
        cfg = json.load(f)["model"]
    block_size = cfg.get("block_size", 512)

    tok = AutoTokenizer.from_pretrained("camembert-base", use_fast=True)

    model_config = MiniGPTConfig(
        vocab_size=inferred["vocab_size"],
        block_size=block_size,
        embed_dim=inferred["embed_dim"],
        depth=inferred["depth"],
        heads=inferred["heads"],
        num_kv_heads=inferred["num_kv_heads"],
        hidden_dim=inferred["hidden_dim"],
        weight_sharing=inferred["weight_sharing"],
        use_rope=cfg.get("use_rope", True),
        num_prelude_layers=inferred["num_prelude_layers"],
        num_coda_layers=inferred["num_coda_layers"],
        num_recurrent_steps=inferred["num_recurrent_steps"],
        use_lti_injection=inferred["use_lti_injection"],
        use_act_halting=inferred["use_act_halting"],
        act_halt_threshold=0.99,
        depth_lora_rank=inferred["depth_lora_rank"],
        dropout=0.0,
        use_gradient_checkpointing=False,
    )

    model = MiniGPTForCausalLM(model_config)
    model.load_state_dict(state, strict=False)
    model.lm_head.weight = model.model.token_emb.weight

    if recurrent_steps is not None:
        model.set_inference_recurrent_steps(recurrent_steps)
        print(f"Recurrent steps: {inferred['num_recurrent_steps']} (train) → {recurrent_steps} (inference)")

    model = model.to(device).eval()

    val_loss = ckpt.get("val_loss", "N/A")
    step = ckpt.get("global_step", "N/A")
    params = sum(p.numel() for p in model.parameters())
    print(f"Modèle chargé: {checkpoint_path} (val_loss={val_loss}, step={step})")
    print(f"Paramètres: {params:,} | Device: {device}\n")

    return model, tok, model_config


@torch.no_grad()
def generate(model, tokenizer, prompt, block_size, device,
             max_new_tokens=80, temperature=0.7, top_p=0.9,
             repetition_penalty=1.0, no_repeat_ngram_size=0):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen_tokens = []

    for _ in range(max_new_tokens):
        idx_cond = ids[:, -block_size:]
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(idx_cond).logits[:, -1, :]

        if repetition_penalty != 1.0:
            seen = set(ids[0].tolist())
            for token_id in seen:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        if no_repeat_ngram_size > 0 and ids.shape[1] >= no_repeat_ngram_size - 1:
            prefix = ids[0].tolist()
            banned = banned_ngram_tokens(prefix, no_repeat_ngram_size)
            if banned:
                logits[0, banned] = -float("inf")

        if temperature != 1.0:
            logits = logits / temperature

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        indices_to_remove = mask.scatter(1, sorted_indices, mask)
        logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        ids = torch.cat((ids, next_token), dim=1)
        gen_tokens.append(token_id)

    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def banned_ngram_tokens(tokens, ngram_size):
    """Return next-token ids that would repeat an existing n-gram."""
    if ngram_size <= 1 or len(tokens) < ngram_size - 1:
        return []

    ngrams = {}
    for i in range(len(tokens) - ngram_size + 1):
        prefix = tuple(tokens[i : i + ngram_size - 1])
        next_token = tokens[i + ngram_size - 1]
        ngrams.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(tokens[-(ngram_size - 1):])
    return list(ngrams.get(current_prefix, []))


def run_tests(model, tokenizer, config, device, temperature, top_p, max_tokens,
              repetition_penalty, no_repeat_ngram_size):
    block_size = config.block_size

    prompts = [
        "La France est un pays d'Europe occidentale dont",
        "Le président de la République française est élu",
        "En mathématiques, une fonction continue est",
        "En 1789, la Révolution française",
        "La tour Eiffel est un monument situé",
        "L'intelligence artificielle désigne un ensemble de techniques",
        "Dans l'univers observable, les galaxies",
        "La musique classique est un genre musical qui",
        "Au Moyen Âge, les villes françaises",
        "Le changement climatique provoque",
    ]

    print("=" * 60)
    print(
        f"  Test de génération (T={temperature}, top_p={top_p}, "
        f"rep_penalty={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size})"
    )
    print("=" * 60)

    for prompt in prompts:
        text = generate(model, tokenizer, prompt, block_size, device,
                        max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size)
        print(f"\n> {prompt}")
        print(f"  {text}")

    print("\n" + "=" * 60)


def interactive_mode(model, tokenizer, config, device, temperature, top_p, max_tokens,
                     repetition_penalty, no_repeat_ngram_size):
    block_size = config.block_size

    print("=" * 60)
    print("  Mode interactif — tapez un début de phrase")
    print("  'quit' pour quitter")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        text = generate(model, tokenizer, prompt, block_size, device,
                        max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size)
        print(f"  → {text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test du modèle pré-entraîné MiniGPT")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_miniGPT.pt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--recurrent-steps", type=int, default=None,
                        help="Override recurrent depth steps (default: training value)")
    parser.add_argument("--interactive", action="store_true",
                        help="Mode interactif pour tester des prompts personnalisés")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model, tokenizer, config = build_model(
        args.config, args.checkpoint, device, args.recurrent_steps
    )

    if args.interactive:
        interactive_mode(model, tokenizer, config, device,
                         args.temperature, args.top_p, args.max_tokens,
                         args.repetition_penalty, args.no_repeat_ngram_size)
    else:
        run_tests(model, tokenizer, config, device,
                  args.temperature, args.top_p, args.max_tokens,
                  args.repetition_penalty, args.no_repeat_ngram_size)
