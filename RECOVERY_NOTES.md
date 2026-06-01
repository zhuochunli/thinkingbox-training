# Recovery Notes — thinkingbox-training

This directory is a reconstruction of the `thinkingbox-training` Azure ML
repository, assembled from three caches after the working copy was lost:

1. **Recovered AML snapshot** — `~/recovered_aml/.../thinkingbox-training/`
   contained 10 substantial files (the verbatim ones below) plus a number of
   0-byte stubs left over from `git checkout` racing against tooling.
2. **Chat transcript** — `~/recovered_aml_transcripts/970dfb5c-87c7-4139-a1c4-825440db83ca.md`
   (the ≈1.5 MB design conversation) contained the full source for every
   file the assistant authored, embedded inline as it was created.
3. **W&B run metadata** — `~/recovered_wandb_ext/thon92gu/` had the exact CLI
   args and 32 hyperparameters used in the last successful run.

Files marked **VERBATIM** are byte-for-byte copies from cache 1 and need no
review. Files marked **RECONSTRUCTED** were extracted from cache 2; each
carries a header comment naming the transcript line range it came from. They
should be sanity-checked against memory before deployment, but they compile
cleanly and match the import surface declared in `train/train.py`.

## File inventory

### Verbatim from recovered AML (10 files)

| File                              | Bytes |
| --------------------------------- | ----- |
| `pyproject.toml`                  |  1238 |
| `scripts/start_vllm.sh`           |  1639 |
| `scripts/test_lora_sync.py`       |  5560 |
| `scripts/test_rl_loss.py`         | 10030 |
| `train/checkpoint.py`             |  7720 |
| `train/lora_sync.py`              |  4445 |
| `train/rl_loss.py`                |  8716 |
| `train/tokenize_chat.py`          |  6048 |
| `train/train.py`                  | 19502 |
| `train/wandb_logger.py`           |  4306 |

### Reconstructed from transcript (12 files)

| File                               | Source (transcript line range)   |
| ---------------------------------- | -------------------------------- |
| `train/__init__.py`                | 3191–3193                        |
| `train/data_pipeline.py`           | 4558–4621                        |
| `train/rollout.py`                 | 4624–4729                        |
| `train/rewards.py`                 | 7394–7436                        |
| `train/eval_loop.py`               | 13493–13583                      |
| `train/patches.py`                 | 10030–10098                      |
| `train/tests/__init__.py`          | (empty — package marker)         |
| `train/tests/test_checkpoint.py`   | 14293–14443 + edit 14478–14484   |
| `scripts/demo_rollout.py`          | 4732–4842                        |
| `scripts/test_tokenize.py`         | 5820–5914                        |
| `scripts/start_servers.sh`         | 3196–3234                        |
| `scripts/smoke_test.sh`            | 3287–3370                        |
| `scripts/start_train_ddp.sh`       | 13345–13375                      |

The edit at line 14478 of the transcript replaced an
`((model_a(x) - target) ** 2).mean()` forward-pass in
`test_save_and_resume_roundtrip` with a synthetic-gradient block, because
PEFT-wrapped models do not accept the `input_ids=...` kwarg the old test
implied. The reconstructed file applies that edit (see the `for p in
model_a.parameters(): if p.requires_grad: p.grad = torch.randn_like(p)`
loop).

## Caveat — `train/train.py` is a pre-Step-G snapshot

The verbatim `train/train.py` in cache 1 predates the W&B run captured in
cache 3. Specifically, it does **not** import any of:

* `train.eval_loop`     (`run_eval`, `log_eval`)
* `train.checkpoint`    (`save_train_state`, `resolve_resume_path`, …)
* `train.wandb_logger`  (`WandbRun`)
* `train.patches`       (`apply()`)

…and it does not parse the following CLI flags that the W&B run used:

```
--max-seq-len 16384  --eval-every 25  --eval-at-start
--save-every 25      --keep-checkpoints 4
--wandb              --wandb-run-name thinkingbox-grpo-50-0528-2118
```

All the *callees* of those missing wires are present (the four modules
listed above), so a small integration pass is required to re-wire them
into `train.py` before the run can be reproduced. The shape of that
wiring is implicit in the function signatures of each callee module and
in the W&B run config below.

## W&B run reference (`thon92gu` / `thinkingbox-grpo-50-0528-2118`)

Exact invocation captured in `recovered_wandb_ext/thon92gu/wandb-metadata.json`:

```
python -m train.train \
    --config config_training.yaml \
    --dataset /home/azureuser/zhuochun/AI.ThinkingBox.Data/dataset \
    --train-list data/train_list.yaml \
    --algo grpo \
    --max-steps 50 \
    --max-seq-len 16384 \
    --n-prompts 12 \
    --g 8 \
    --concurrency 24 \
    --eval-every 25 --eval-at-start \
    --save-every 25 --keep-checkpoints 4 \
    --wandb --wandb-run-name thinkingbox-grpo-50-0528-2118
```

Resolved hyperparameters (32 keys, from `config.json`):

| key                  | value                                                                  |
| -------------------- | ---------------------------------------------------------------------- |
| algo                 | grpo                                                                   |
| model                | Qwen/Qwen3.5-9B                                                        |
| lr                   | 1e-6                                                                   |
| max_steps            | 50                                                                     |
| max_seq_len          | 16384                                                                  |
| n_prompts            | 12                                                                     |
| g                    | 8                                                                      |
| concurrency          | 24                                                                     |
| micro_batch          | 1                                                                      |
| grad_accum           | 1                                                                      |
| lora_r               | 16                                                                     |
| lora_alpha           | 32                                                                     |
| lora_dropout         | 0.0                                                                    |
| lora_target_modules  | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj          |
| kl_coef              | 0.04                                                                   |
| clip_ratio           | 0.2                                                                    |
| entropy_coef         | 0.0                                                                    |
| eval_every           | 25                                                                     |
| eval_at_start        | true                                                                   |
| save_every           | 25                                                                     |
| keep_checkpoints     | 4                                                                      |
| reward_fn            | binary_test_result                                                     |
| dataset              | /home/azureuser/zhuochun/AI.ThinkingBox.Data/dataset                   |
| train_list           | data/train_list.yaml                                                   |
| config               | config_training.yaml                                                   |
| wandb_run_name       | thinkingbox-grpo-50-0528-2118                                          |
| world_size           | 6                                                                      |
| seed                 | 42                                                                     |
| eval_size            | 5                                                                      |
| lora_save_dir        | checkpoints/lora                                                       |
| state_save_dir       | checkpoints/state                                                      |
| lora_name_template   | policy_step_{step:05d}                                                 |

The 32nd key (`agent`) defaulted to `"think"`.

## Sanity checklist

1. `pip install -e .` in a venv that already has the sibling editable
   install of `~/zhuochun/thinkingbox` (this repo imports from it).
2. Place `config_training.yaml` and `data/train_list*.yaml` at the paths
   listed in MISSING.md.
3. Start the vLLM/MCP/typesense backplane: `scripts/start_vllm.sh` and
   `scripts/start_servers.sh` in separate shells.
4. Confirm the runtime contract: `python scripts/demo_rollout.py
   --n-prompts 2 --g 2 --concurrency 2 --out /tmp/demo.jsonl` should emit a
   JSONL with 4 lines and non-error rollouts.
5. Verify tokenization round-trips: `python scripts/test_tokenize.py
   --rollouts /tmp/demo.jsonl`.
6. Run the checkpoint unit test (no GPU required):
   `python train/tests/test_checkpoint.py`.
7. Only after the above pass, integrate the missing `eval_loop` /
   `checkpoint` / `wandb_logger` / `patches` wiring into `train/train.py`
   (see "pre-Step-G snapshot" caveat above) and launch the full run via
   `scripts/start_train_ddp.sh`.
