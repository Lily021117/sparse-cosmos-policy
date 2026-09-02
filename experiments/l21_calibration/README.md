# Phase 3F/3H structured-L2,1 provenance

This directory preserves the small, reproducible artifacts needed to continue
structured-L2,1 calibration on a different machine.  It intentionally excludes
Phase 3F's 52 GiB DCP resume state and all optimizer/master-weight checkpoints.

## Git baseline

```text
a51caaf6e0b82ca084c7abb7bf75bb9ff41dd11d
```

## Phase 3F: validated dense Policy recipe

Phase 3F is complete and does **not** need to be rerun on AutoDL.

```text
seed                       = 1234
micro_batch_size           = 1
grad_accum_steps           = 32
base_lr                    = 1.6667e-6
clip_norm                  = 1.0
optimizer_steps            = 500
structured_l21_lambda      = 0
SA                          = false
CA                          = false
MLP                         = true, diagnostic only
SharedTokenLinear           = frozen
```

Validation:

```text
R_mlp                       = 65377.80078125
groups                      = 57344
group-norm mean             = 1.14009845
group-norm CV               = 0.25390932
optimizer steps completed   = 500
micro batches completed     = 16000
SharedTokenLinear final diff= 0
```

## Phase 3H: SA/CA gradient calibration

`phase3h_sa_ca_accum_gradient_audit.py` is read-only: it does not create an
optimizer or execute an optimizer step.

```text
seed                 = 1234
micro_batches        = 32
batch_size            = 1
loss_divisor          = 32
lambda                = 0
SA                    = true
CA                    = true
MLP                   = false
SharedTokenLinear     = frozen
```

Recorded result:

```text
SA: R = 53302.29296875, groups = 57344, lambda_1pct = 5.759493884e-06
CA: R = 28110.14453125, groups = 57344, lambda_1pct = 1.902407393e-07
```

## Required assets and invocation

Use the Policy checkpoint, not the base Video2World checkpoint:

```text
Cosmos-Policy-LIBERO-Predict2-2B.pt
SHA256 = 8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2
```

Set the dataset and checkpoint paths for the new machine, then run one process:

```bash
export BASE_DATASETS_DIR=/path/to/datasets
export POLICY_CHECKPOINT=/path/to/Cosmos-Policy-LIBERO-Predict2-2B.pt

torchrun --standalone --nproc_per_node=1 \
  experiments/l21_calibration/phase3h_sa_ca_accum_gradient_audit.py
```

Alternatively, pass `--checkpoint /path/to/Cosmos-Policy-LIBERO-Predict2-2B.pt`.
The script derives the repository root from its own location, or accepts
`--repo-root /path/to/cosmos-policy`.
