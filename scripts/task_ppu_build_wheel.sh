#!/bin/bash
# Build the flashinfer-for-sail wheel (AOT) inside the PPU build container.
# Expects the repository (with submodules) to be mounted at the current directory.

set -eo pipefail
set -x

: "${TORCH_CUDA_ARCH_LIST:=8.0}"
: "${FLASHINFER_CUDA_ARCH_LIST:=8.0}"
export TORCH_CUDA_ARCH_LIST
export FLASHINFER_CUDA_ARCH_LIST

echo "========================================"
echo "Build environment"
echo "========================================"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
nvcc --version || echo "nvcc not found"
echo "  - Available Memory: $(free -g | awk '/^Mem:/ {print $7}') GB"
echo "  - Number of Processors: $(nproc)"

echo ""
echo "========================================"
echo "Installing build dependencies"
echo "========================================"
pip install packaging setuptools build --force-reinstall --no-deps
pip install apache-tvm-ffi==0.1.6
pip install pynvml==13.0.1

echo ""
echo "========================================"
echo "Running AOT compilation"
echo "========================================"
python -m flashinfer.aot

echo ""
echo "========================================"
echo "Building wheel"
echo "========================================"
rm -rf dist
python -m build --no-isolation --wheel

WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
echo "Built wheel: ${WHEEL_FILE}"

echo ""
echo "========================================"
echo "Installing and verifying wheel"
echo "========================================"
pip install "${WHEEL_FILE}"
python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo ""
echo "========================================"
echo "Build succeeded: ${WHEEL_FILE}"
echo "========================================"
