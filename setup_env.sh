#!/bin/bash
# Create virtualenv and install dependencies without uv.
# Run from the repo root: bash setup_env.sh

set -e

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install torch first (torch-sparse/scatter need it at build time)
pip install torch

# torch-sparse and torch-scatter require torch to be present before building
pip install torch-sparse torch-scatter --no-build-isolation

# Rest of the dependencies
pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
