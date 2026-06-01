# thinkingbox-training

Fine-tuning / RL training scripts built on top of the [thinkingbox](https://github.com/microsoft/thinkingbox) framework.

## Reproducing the environment

The training stack pins exact versions of every dependency. The GPU stack
(PyTorch + vLLM + CUDA wheels) is installed separately because those wheels
come from custom indexes and are auto-selected for your local CUDA.

### Prerequisites

- Linux with an NVIDIA GPU and a recent CUDA driver
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or see uv docs)
- Sibling checkout of `thinkingbox` next to this repo:

  ```
  parent/
    thinkingbox/
    thinkingbox-training/   <-- you are here
  ```

### Steps

```bash
# 1. Create the venv
uv venv --python 3.12

# 2. Install GPU stack (torch, vllm, triton, CUDA wheels) — auto-detects CUDA
uv pip install vllm --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

# 3. Install the rest of the pinned dependencies
uv pip install -r requirements.txt

# 4. Install thinkingbox as an editable sibling
uv pip install -e ../thinkingbox
```

Activate with `source .venv/bin/activate`.

### Re-generating `requirements.txt`

After adding/removing tools in the venv:

```bash
uv pip freeze \
  | grep -v '^-e file://' \
  | grep -viE '^(torch|torchvision|torchaudio|triton|vllm|nvidia-|cuda-|flashinfer|xgrammar|tilelang|tokenspeed|quack-kernels|humming-kernels|apache-tvm)' \
  > requirements.txt
```

(The grep filters out the GPU-stack packages installed in step 2; those are
pulled in transitively by `vllm` and pinned by uv's torch backend resolver.)
