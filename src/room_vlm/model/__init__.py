"""Model package exports."""

from room_vlm.model.loader import LoadedModel, attach_adapter, load_model_and_processor, print_oom_remediation

__all__ = [
    "LoadedModel",
    "attach_adapter",
    "load_model_and_processor",
    "print_oom_remediation",
]
