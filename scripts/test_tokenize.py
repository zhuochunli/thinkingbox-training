"""Round-trip check: tokenize a rollout, decode the masked spans, compare to
the original assistant content. Run after demo_rollout.py.

Usage:
    PYTHONPATH=. python scripts/test_tokenize.py \
        --rollouts /tmp/tb_demo_rollouts.jsonl \
        --model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from train.tokenize_chat import load_tokenizer, render_with_assistant_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="/tmp/tb_demo_rollouts.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--limit", type=int, default=3, help="check first N rollouts")
    return p.parse_args()


def main():
    args = parse_args()
    tok = load_tokenizer(args.model)
    print(f"loaded tokenizer: vocab={tok.vocab_size}  pad={tok.pad_token}  eos={tok.eos_token}")

    n_ok = 0
    n_total = 0
    with open(args.rollouts) as f:
        for line_no, line in enumerate(f, 1):
            if line_no > args.limit:
                break
            r = json.loads(line)
            uid = r["uid"]
            raw_msgs = r["raw_messages"]
            if not raw_msgs:
                print(f"  [{line_no}] {uid}: no raw_messages, skip")
                continue

            try:
                tk = render_with_assistant_mask(raw_msgs, tok)
            except Exception as e:
                print(f"  [{line_no}] {uid}: RENDER FAIL: {e}")
                continue

            n_assistant_msgs = sum(1 for m in raw_msgs if m.get("role") == "assistant")
            mask_token_count = sum(tk.assistant_mask)
            print(
                f"\n[{line_no}] {uid}"
                f"\n  msgs={len(raw_msgs)}  assistant_msgs={n_assistant_msgs}"
                f"  total_tokens={len(tk.input_ids)}  masked_tokens={mask_token_count}"
                f"  spans={len(tk.assistant_spans)}"
            )

            # Decode each masked span and verify it contains the assistant content
            asst_msgs = [m for m in raw_msgs if m.get("role") == "assistant"]
            if len(asst_msgs) != len(tk.assistant_spans):
                print(
                    f"  ✗ span count {len(tk.assistant_spans)} != assistant msg count {len(asst_msgs)}"
                )
                continue

            all_good = True
            for j, ((s, e), msg) in enumerate(zip(tk.assistant_spans, asst_msgs)):
                decoded = tok.decode(tk.input_ids[s:e], skip_special_tokens=False)
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                # Content-only assistant message: decoded must contain content text + <|im_end|>
                if content and not tool_calls:
                    if content.strip() not in decoded:
                        all_good = False
                        print(f"    span[{j}]: content mismatch")
                        print(f"      decoded[:120] = {decoded[:120]!r}")
                        print(f"      content[:120] = {content[:120]!r}")
                if not decoded.rstrip().endswith("<|im_end|>"):
                    all_good = False
                    print(f"    span[{j}]: missing trailing <|im_end|>  decoded tail = {decoded[-60:]!r}")

            if all_good:
                n_ok += 1
                print("  ✓ round-trip ok")
            n_total += 1

    print(f"\n=== {n_ok}/{n_total} rollouts passed round-trip ===")
    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
