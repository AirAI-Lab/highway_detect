import os
import torch
from torchvision import models

TAG = "[BackboneLoader]"

# Structured status logs for programmatic consumption by tools/quick_cpu_sweep.py
# Each entry: {caller, variant, event, path, strict, exception}
STATUS_LOGS = []

def _append_status(entry: dict):
    try:
        STATUS_LOGS.append(entry)
    except Exception:
        pass

def get_and_clear_status_logs():
    """Return and clear internal status logs (thread-unsafe but sufficient here)."""
    logs = list(STATUS_LOGS)
    STATUS_LOGS.clear()
    return logs

def _normpath_for_msg(p: str) -> str:
    # show forward slashes for readability across platforms
    return os.path.normpath(p).replace('\\', '/')


def load_resnet34_backbone(pretrained: bool = True, caller: str = ""):
    """Back-compat wrapper kept for existing imports. Uses the generalized loader."""
    return load_resnet_backbone('resnet34', pretrained=pretrained, caller=caller)


def load_resnet_backbone(variant: str = 'resnet34', pretrained: bool = True, caller: str = ""):
    """
    Generalized local-only loader for ResNet backbones.
    Behavior:
      - Always construct torchvision model with weights=None (no downloads)
      - If pretrained=True, try to load a local file from experiments/weights/{variant}.pth
      - If file missing or load fails, fall back to random init and log the event
    Supported variants: resnet18, resnet34, resnet50, resnet101 (others will attempt ctor(weights=None)).
    """
    variant = (variant or 'resnet34').lower()
    ctor = getattr(models, variant)

    # Env-configurable flags
    env_strict = os.environ.get('BACKBONE_STRICT')
    env_map_location = os.environ.get('BACKBONE_MAP_LOCATION')  # e.g., 'cpu', 'cuda', 'cuda:0'
    env_move_to_device = os.environ.get('BACKBONE_MOVE_TO_DEVICE')  # '1' to move model before return
    strict = str(env_strict).strip().lower() in {"1", "true", "yes"} if env_strict is not None else False
    map_location = env_map_location if env_map_location else 'cpu'
    move_to_device = str(env_move_to_device).strip().lower() in {"1", "true", "yes"}

    # Construct uninitialized model (no downloads)
    try:
        model = ctor(weights=None)
    except TypeError:  # very old torchvision API
        model = ctor(pretrained=False)

    if not pretrained:
        print(f"{TAG} {caller}: pretrained=False -> using random-init {variant}")
        _append_status({
            'caller': caller,
            'variant': variant,
            'event': 'pretrained_false',
            'path': None,
            'strict': strict,
            'exception': None,
        })
        return model

    # Resolve local weights path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, '..', 'weights', f'{variant}.pth')
    weights_path_msg = _normpath_for_msg(weights_path)

    if not os.path.exists(weights_path):
        print(f"{TAG} {caller}: weights not found at {weights_path_msg} -> fallback to random-init {variant}")
        _append_status({
            'caller': caller,
            'variant': variant,
            'event': 'not_found',
            'path': weights_path_msg,
            'strict': strict,
            'exception': None,
        })
        return model

    try:
        # torch.load may have weights_only default in newer PyTorch; set explicitly for legacy files
        state = torch.load(weights_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state, strict=strict)
        if move_to_device and map_location and map_location != 'cpu':
            try:
                model = model.to(map_location)
            except Exception:
                pass
        print(f"{TAG} {caller}: loaded {variant} weights from {weights_path_msg} (strict={strict})")
        _append_status({
            'caller': caller,
            'variant': variant,
            'event': 'loaded',
            'path': weights_path_msg,
            'strict': strict,
            'exception': None,
        })
    except Exception as e:
        print(f"{TAG} {caller}: failed to load weights from {weights_path_msg} ({e!r}); fallback to random-init {variant}")
        _append_status({
            'caller': caller,
            'variant': variant,
            'event': 'load_failed',
            'path': weights_path_msg,
            'strict': strict,
            'exception': repr(e),
        })
    return model
