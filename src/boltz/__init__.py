from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("boltz")
except PackageNotFoundError:
    # package is not installed
    pass

# Apply SafeLayerNorm monkey-patch for PyTorch CUDA uint32 overflow bug.
# https://github.com/pytorch/pytorch/issues/149301
import boltz.model.layers.safe_layer_norm  # noqa: E402, F401
