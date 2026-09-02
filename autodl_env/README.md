# AutoDL environment handoff — Cosmos Policy L21

This bundle belongs to the original Cosmos Policy repository:

```text
/data/lyc/projects/cosmos-policy
branch: exp/bilevel-dit-l21-v1
HEAD: a51caaf6e0b82ca084c7abb7bf75bb9ff41dd11d
```

It does **not** apply to `cosmos-predict2p5-sparse` or to any of the old
sparse-attention repositories.

## Runtime target

- Linux x86_64
- NVIDIA driver compatible with CUDA 12.8
- Python 3.10
- H100/A100-class GPU recommended for full Predict2-2B training

The source of truth for Python dependencies is the repository's `uv.lock`.
Do not replace it with a hand-written `requirements.txt`.

The verified local versions are:

```text
Python               3.10.14
torch                2.7.0+cu128
CUDA runtime         12.8
transformer-engine   2.2+cu128.torch27
flash-attn           2.7.3
```

## Setup on AutoDL

Clone the repository and select the exact experiment branch:

```bash
git clone git@github.com:Lily021117/sparse-cosmos-policy.git cosmos-policy
cd cosmos-policy
git switch exp/bilevel-dit-l21-v1
git rev-parse HEAD
```

The expected SHA is `a51caaf6e0b82ca084c7abb7bf75bb9ff41dd11d`.

Install host libraries if they are absent from the AutoDL image:

```bash
apt-get update
apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0
```

Install uv, then create the locked CUDA 12.8 + LIBERO environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
./autodl_env/setup_locked_env.sh
```

Run the no-training verification:

```bash
uv run --locked --extra cu128 --group libero --python 3.10 \
  python autodl_env/verify_environment.py
```

## Required non-code assets

These are not stored in Git and must be provided separately.

```text
LIBERO data root:
  $BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/success_only
  $BASE_DATASETS_DIR/LIBERO-Cosmos-Policy/all_episodes

Policy checkpoint:
  Cosmos-Policy-LIBERO-Predict2-2B.pt
  SHA256:
  8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2
```

Set the data root before training:

```bash
export BASE_DATASETS_DIR=/path/to/datasets
```

The project config also resolves the base Video2World checkpoint while it is
being imported. Ensure the AutoDL account has Hugging Face access/cache for:

```text
nvidia/Cosmos-Predict2-2B-Video2World
nvidia/Cosmos-Policy-LIBERO-Predict2-2B
```

Pre-download them if network access is unreliable:

```bash
uv run --locked --extra cu128 --group libero hf download \
  nvidia/Cosmos-Predict2-2B-Video2World
uv run --locked --extra cu128 --group libero hf download \
  nvidia/Cosmos-Policy-LIBERO-Predict2-2B
```

## Experiment constraints preserved by this branch

- `SharedTokenLinear` is identity initialized and frozen for the current
  single-level L21 experiments.
- Structured L2,1 supports SA input-channel, CA query-input-channel and MLP
  input-channel terms.
- The current intended Policy checkpoint is the SHA above, not the base
  Video2World checkpoint.

No dataset, checkpoint, cache, `__pycache__`, or temporary Phase3 scripts are
included by this environment bundle.
