from . import lnl, spec_guidance, generic_guidance

NODE_CLASS_MAPPINGS = (
    lnl.NODE_CLASS_MAPPINGS
    | spec_guidance.NODE_CLASS_MAPPINGS
    | generic_guidance.NODE_CLASS_MAPPINGS
)
NODE_DISPLAY_NAME_MAPPINGS = (
    lnl.NODE_DISPLAY_NAME_MAPPINGS
    | spec_guidance.NODE_DISPLAY_NAME_MAPPINGS
    | generic_guidance.NODE_DISPLAY_NAME_MAPPINGS
)
