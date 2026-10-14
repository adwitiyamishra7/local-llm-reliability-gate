"""Lets the unit tests import src.gate.* without torch/transformers installed (mocked tests only)."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except ImportError:  # sandbox / CI without GPU stack: stub only what import-time needs
    torch_stub = types.ModuleType("torch")
    torch_stub.device = type("device", (), {})
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.float16 = torch_stub.float32 = "dtype"
    sys.modules["torch"] = torch_stub

try:
    import transformers  # noqa: F401
except ImportError:
    tf_stub = types.ModuleType("transformers")
    tf_stub.AutoModelForCausalLM = tf_stub.AutoTokenizer = object
    sys.modules["transformers"] = tf_stub
