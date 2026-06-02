"""Standalone unit test for train.checkpoint — no vLLM / MCP / Qwen needed.

Uses a tiny synthetic model + LoRA to validate that save_train_state /
load_train_state round-trip correctly and that prune_old_state respects keep.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from train.checkpoint import (
    find_latest_state,
    load_train_state,
    lora_path_for,
    prune_old_state,
    resolve_resume_path,
    save_train_state,
    state_path,
)
from train.lora_sync import save_adapter


class _TinyCausalLM(nn.Module):
    """A 2-layer model with q_proj/v_proj that PEFT can target."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.out = nn.Linear(16, 4, bias=False)

    def forward(self, x):
        return self.out(self.v_proj(self.q_proj(x)))


def _make_peft():
    base = _TinyCausalLM()
    cfg = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(base, cfg)


def test_save_and_resume_roundtrip(tmp_path: Path):
    lora_dir = tmp_path / "lora"
    state_dir = tmp_path / "state"
    template = "policy_step_{step:05d}"

    # --- Train one model, take an optimizer step, save state. ---
    torch.manual_seed(123)
    random.seed(123)
    model_a = _make_peft()
    opt_a = torch.optim.AdamW(
        [p for p in model_a.parameters() if p.requires_grad], lr=1e-3,
    )
    # Synthetic gradient + optimizer step so AdamW has non-trivial state to save.
    for p in model_a.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    opt_a.step()

    step = 7
    save_adapter(model_a, lora_path_for(lora_dir, template, step))
    save_train_state(
        step=step, optimizer=opt_a, lora_name=template.format(step=step),
        world_size=1, out_path=state_path(state_dir, step),
    )

    # --- Fresh model + fresh optimizer; resume from state. ---
    torch.manual_seed(999)  # different seed → different fresh LoRA init
    random.seed(999)
    model_b = _make_peft()
    opt_b = torch.optim.AdamW(
        [p for p in model_b.parameters() if p.requires_grad], lr=1e-3,
    )
    payload = load_train_state(
        in_path=state_path(state_dir, step),
        optimizer=opt_b, peft_model=model_b,
        lora_save_dir=lora_dir, lora_name_template=template,
        world_size=1, map_location="cpu",
    )

    # --- Verify LoRA weights match. ---
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert na == nb
        if "lora_" in na:
            assert torch.equal(pa, pb), f"LoRA param {na} mismatch after resume"

    # --- Verify optimizer state matches. ---
    sa, sb = opt_a.state_dict(), opt_b.state_dict()
    assert sa["param_groups"][0]["lr"] == sb["param_groups"][0]["lr"]
    assert set(sa["state"]) == set(sb["state"]), "optimizer state keys differ"
    for k in sa["state"]:
        for tk in sa["state"][k]:
            va, vb = sa["state"][k][tk], sb["state"][k][tk]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), f"optimizer state mismatch at {k}/{tk}"
            else:
                assert va == vb

    # --- Verify metadata. ---
    assert payload["step"] == step
    assert payload["lora_name"] == template.format(step=step)
    print("[ok] roundtrip: LoRA + optimizer + metadata all match after resume")


def test_resolve_latest_and_prune(tmp_path: Path):
    lora_dir = tmp_path / "lora"
    state_dir = tmp_path / "state"
    template = "policy_step_{step:05d}"

    model = _make_peft()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    saved_steps = [0, 1, 2, 3, 5, 8]
    for s in saved_steps:
        save_adapter(model, lora_path_for(lora_dir, template, s))
        save_train_state(
            step=s, optimizer=opt, lora_name=template.format(step=s),
            world_size=1, out_path=state_path(state_dir, s),
        )

    # --- find_latest_state ---
    latest = find_latest_state(state_dir)
    assert latest is not None and latest.name == "state_step_00008.pt", latest

    # --- resolve_resume_path "latest" ---
    p = resolve_resume_path("latest", state_dir)
    assert p == latest

    # --- prune_old_state keep=2 → keep steps {8, 5} + step 0 (always); drop {1, 2, 3} ---
    prune_old_state(state_dir, lora_dir, template, keep=2)
    remaining_states = sorted(p.name for p in state_dir.iterdir())
    assert remaining_states == ["state_step_00000.pt", "state_step_00005.pt", "state_step_00008.pt"], remaining_states
    remaining_loras = sorted(p.name for p in lora_dir.iterdir())
    assert remaining_loras == ["policy_step_00000", "policy_step_00005", "policy_step_00008"], remaining_loras
    print("[ok] find_latest + resolve_resume + prune all behave correctly")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_save_and_resume_roundtrip(Path(td) / "a")
    with tempfile.TemporaryDirectory() as td:
        test_resolve_latest_and_prune(Path(td) / "b")
    print("all checkpoint tests passed.")
