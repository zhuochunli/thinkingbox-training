# Missing Artifacts

The following artifacts are **not** in any cache and must be re-created
or restored from elsewhere before the repo will run end-to-end.

## Configuration

* **`config_training.yaml`** — the master `ConfigFile` (orchestrator,
  agent_model, judge_model, MCP wiring, …). Referenced by `train/train.py`,
  `scripts/demo_rollout.py`, `scripts/smoke_test.sh`, and the W&B run.
  No textual copy survives in the transcript; it was always referenced by
  path, never inlined. Re-create from `~/zhuochun/thinkingbox`'s
  `config*.yaml` family, pointing `deployment` at the local vLLM
  (`http://127.0.0.1:8000/v1`, `Qwen/Qwen3.5-9B`).

## Training data lists

Referenced throughout the repo and W&B run; YAML lists of
`"filename:testname"` strings consumed by `train.data_pipeline.load_test_list`:

* `data/train_list.yaml`
* `data/train_list_airline.yaml`
* `data/train_list_origin.yaml`
* `data/train_list_in_scenario.yaml`

These were curated by the user; no copy exists in cache. Re-derive from
`$THINKINGBOX_DATA/dataset/**/*.yaml` using the same selection criteria
used originally (airline / origin / in-scenario partitions of the suite).

## Dataset

* **`/home/azureuser/zhuochun/AI.ThinkingBox.Data/dataset`** — the
  hydration source for every `HydratedTestCase`. This lives in a private
  data repo (`AI.ThinkingBox.Data`); it is **not** part of the training
  repo and is expected to be cloned alongside it on the training VM.
  The `THINKINGBOX_DATA` env var in `scripts/start_servers.sh` and
  `scripts/smoke_test.sh` must point at it.

## Sibling editable install

The training repo imports heavily from `thinkingbox.*` (see
`train/rollout.py`, `train/data_pipeline.py`, `train/patches.py`,
`train/eval_loop.py`):

* `thinkingbox.cli.infer.TBWorker`
* `thinkingbox.cli.common.load_yaml`
* `thinkingbox.common.config_types.{ConfigFile, HydratedTestCase}`
* `thinkingbox.common.chat_types.DecodeResult`
* `thinkingbox.common.hydrator.iter_cases_by_names`
* `thinkingbox.common.http_client.initialize_dns_cache`
* `thinkingbox.common.agent_session.AgentSession`

These come from the sibling `~/zhuochun/thinkingbox/` repo, installed
editably into the same venv. Without it, nothing in this training repo
imports.

## Runtime state (intentionally absent)

These are training-run outputs, not source artifacts; listed here only
so future readers don't go hunting:

* `checkpoints/lora/policy_step_*/`        — saved LoRA adapters
* `checkpoints/state/state_step_*.pt`      — optimizer / RNG / metadata
* `checkpoints/train.jsonl`                — per-step training log
* `checkpoints/eval.jsonl`                 — per-eval-step metrics log
* W&B run dirs under `wandb/`              — see cache `recovered_wandb_ext/`

## Tests directory marker

* **`train/tests/__init__.py`** is an empty file we added so pytest /
  `python -m` discovery works; no copy survives because the transcript
  never showed it being created. Likely existed in the original repo.

## Open questions for the original author

1. Confirm whether `data/train_list.yaml` is the union of the three
   sub-lists or a separate curated set.
2. Confirm the exact LoRA target modules for Qwen3.5-9B (the W&B config
   says `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
   — verify these match the actual `LoraConfig` in `config_training.yaml`).
3. Verify the post-Step-G `train.py` (which integrated `eval_loop`,
   `checkpoint`, `wandb_logger`, `patches`) was committed somewhere
   — the cached `train.py` predates that integration.
