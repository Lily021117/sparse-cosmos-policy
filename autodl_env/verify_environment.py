"""No-training AutoDL environment verification for the original Cosmos Policy."""

from __future__ import annotations

import platform
import sys

import torch


def version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "installed"))


print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"transformer_engine={version('transformer_engine')}")
print(f"flash_attn={version('flash_attn')}")
try:
    import libero
except ImportError as exc:
    raise SystemExit(f"LIBERO import failed: {exc}") from exc
print(f"libero={getattr(libero, '__version__', 'installed (no __version__)')}")

# A minimal allocation catches an unusable CUDA runtime without training.
probe = torch.ones(16, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert bool(torch.isfinite(probe).all())
print("cuda_probe=PASS")
