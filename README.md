[![Docs](https://github.com/LargeAgentCollider/torch_june/actions/workflows/docs.yml/badge.svg)](https://github.com/LargeAgentCollider/torch_june/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/largeagentcollider/torch_june/branch/main/graph/badge.svg?token=ddIEG0Eest)](https://codecov.io/gh/largeagentcollider/torch_june)
[![Build and test package](https://github.com/LargeAgentCollider/torch_june/actions/workflows/ci.yml/badge.svg)](https://github.com/LargeAgentCollider/torch_june/actions/workflows/ci.yml)

# GradABM-JUNE
Implementation of the JUNE model using the GradABM framework.

# Setup 

Install requirements

```bash
pip install -r requirements.txt
```

and install PyTorch geometric, manually for now:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
```

Then install the GradABM-june package

```bash
pip install --no-deps -e .
```


# Usage

See the [docs](https://largeagentcollider.github.io/GradABM-JUNE/).
