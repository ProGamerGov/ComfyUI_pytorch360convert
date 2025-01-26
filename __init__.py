from .nodes import (
    C2EMaskedDiffNode,
    C2ENode,
    CropImageWithCoordsNode,
    E2CNode,
    E2ENode,
    E2PNode,
    PasteImageWithCoordsNode,
    RollImageNode,
    SplitFacesNode,
    StackFacesNode,
)

NODE_CLASS_MAPPINGS = {
    "Cubemap to Equirectangular": C2ENode,
    "Equirectangular to Cubemap": E2CNode,
    "Equirectangular to Perspective": E2PNode,
    "Equirectangular Rotation": E2ENode,
    "Split Cubemap Faces": SplitFacesNode,
    "Stack Cubemap Faces": StackFacesNode,
    "Roll Image Axes": RollImageNode,
    "Masked Diff C2E": C2EMaskedDiffNode,
    "Crop Image with Coords": CropImageWithCoordsNode,
    "Paste Image with Coords": PasteImageWithCoordsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cubemap to Equirectangular": "Cubemap to Equirectangular",
    "Equirectangular to Cubemap": "Equirectangular to Cubemap",
    "Equirectangular to Perspective": "Equirectangular to Perspective",
    "Equirectangular Rotation": "Equirectangular Rotation",
    "Split Cubemap Faces": "Split Cubemap Faces",
    "Stack Cubemap Faces": "Stack Cubemap Faces",
    "Roll Image Axes": "Roll Image Axes",
    "Masked Diff C2E": "Masked Diff C2E",
    "Crop Image with Coords": "Crop Image with Coords",
    "Paste Image with Coords": "Paste Image with Coords",
}
