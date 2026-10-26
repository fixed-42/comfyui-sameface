"""SameFace: measure and repair face drift after an image edit."""
# Loaded by ComfyUI as a package. Outside one - pytest collecting the repository
# root - there are no nodes to register, and the tests import `sameface` directly.
if __package__:
    from .nodes import comfy_entrypoint

    # SameFace Simple's figures box on the node.
    WEB_DIRECTORY = "./web"
    __all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
