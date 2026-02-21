import torch
import functools

# PyTorch 2.6 changed torch.load default weights_only from False to True.
# Third-party checkpoints (e.g., DUSt3R) use argparse.Namespace and other
# objects that require weights_only=False. Monkey-patch globally so we don't
# need to modify third-party code.
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load