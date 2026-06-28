#!/usr/bin/env bash
# Source this file before launching any training script.
# Sets up Vulkan and activates the specified venv.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Vulkan fix for Mila cluster
export LD_LIBRARY_PATH="$REPO_ROOT/vulkan_lib/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

# Name of the virtual environment (default: .venv)
VENV_NAME="${1:-.venv}"

# Activate venv
source "$REPO_ROOT/$VENV_NAME/bin/activate"

# LeRobot home — keep datasets inside the repo
export HF_LEROBOT_HOME="$REPO_ROOT/demos/lerobot"

echo "Environment set up from $REPO_ROOT"
echo "Activated virtual environment: $VENV_NAME"
