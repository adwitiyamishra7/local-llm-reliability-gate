# Windows GPU Setup

The tested development environment for this repository is:
- Python 3.13.5
- PyTorch 2.12.0+cu130
- Transformers 5.19.0
- CUDA available
- NVIDIA GeForce GTX 1650

For a fresh environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gpu-cu130.txt
```
