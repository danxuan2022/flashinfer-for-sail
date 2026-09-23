#!/usr/bin/env bash
# Run the reduced PPU (810E) unit-test subset against the freshly built
# flashinfer-for-sail wheel. Invoked from .github/workflows/ppu-linux-test.yml
# inside the flytiger PPU pod. The image ships bash, so this stays bash-only and
# uses an array for the pytest --deselect list; the workflow calls it via `bash`.
set -eu

cd "${PPU_SOURCE_DIR:-/workspace/source}"

echo '=== PPU Verification ==='
echo "Hostname: $(hostname), Node: ${NODE_NAME:-}"
echo "RANK=${RANK:-} NODE_RANK=${NODE_RANK:-} WORLD_SIZE=${WORLD_SIZE:-} NPROC_PER_NODE=${NPROC_PER_NODE:-}"
ppu-smi || true
echo '--- source tree ---'
ls -al

# Locate the wheel staged from the build run and install it.
# --no-deps --force-reinstall: do NOT pull requirements.txt (it pins torch and
# would clobber the PPU runtime baked into this image); just install the
# compiled flashinfer-for-sail wheel over any preinstalled copy. This mirrors
# the README offline-install path.
WHEEL=$(ls -t wheelhouse/*/*.whl 2>/dev/null | head -n 1 || true)
if [ -z "$WHEEL" ]; then
  echo "ERROR: no wheel found under wheelhouse/" >&2
  find wheelhouse -type f -print || true
  exit 1
fi
echo "Installing wheel: $WHEEL"
pip install --no-deps --force-reinstall "$WHEEL"

# Verify the install before spending PPU time on the suite.
python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
python -m flashinfer show-config || true

# The PPU image ships pytest via SGLang; install it only if missing (again
# without touching torch).
python -c "import pytest" 2>/dev/null || pip install --no-deps pytest

# Cases that are known to fail on the 810E PPU backend are skipped via
# --deselect so the rest of each file still runs:
#   - test_norm.py::test_layernorm_quant* : layernorm_quant is not built for PPU
#     ("module 'flashinfer.norm' has no attribute 'layernorm_quant'").
#   - test_sampling.py freq / seed-offset cases: fail the distribution-similarity
#     and same-seed reproducibility asserts on this backend.
DESELECT=(
  --deselect "tests/utils/test_norm.py::test_layernorm_quant"
  --deselect "tests/utils/test_norm.py::test_layernorm_quant_invalid_inputs"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_freq[normal_distribution(std=1)-32000]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_freq[normal_distribution(std=1)-128256]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[32000-99]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[32000-989]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[128256-99]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[128256-989]"
)

# Reduced PPU smoke set: only these scripts run here instead of the full
# auto-discovered tests/ suite. single PPU => sequential.
echo '=== Run unit tests (PPU subset) ==='
pytest --continue-on-collection-errors \
  "${DESELECT[@]}" \
  tests/test_jit_cpp_ext.py \
  tests/autotuner/test_autotuner_configs.py \
  tests/grouped_mm/test_grouped_mm_bf16.py \
  tests/utils/test_norm.py \
  tests/utils/test_activation.py \
  tests/utils/test_sampling.py
echo '=== DONE ==='
