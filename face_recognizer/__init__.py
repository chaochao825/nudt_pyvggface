__all__ = []

try:
    from .pyvggface_model import PyVGGFaceModel

    __all__.append("PyVGGFaceModel")
except Exception:
    PyVGGFaceModel = None
