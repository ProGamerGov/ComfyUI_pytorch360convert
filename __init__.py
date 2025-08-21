from .nodes import (
    ApplyCircularConvPaddingModel,
    ApplyCircularConvPaddingVAE,
    C2EMaskedDiffNode,
    C2ENode,
    CreatePoleMask,
    CreateSeamMask,
    Crop360To180Node,
    CropImageWithCoordsNode,
    E2CNode,
    E2ENode,
    E2Face,
    E2PNode,
    EMask2Face,
    Face2E,
    FaceMask2E,
    MaskE2ENode,
    MonoScopicToStereoNode,
    Pad180To360Node,
    PasteImageWithCoordsNode,
    RollImageNode,
    RollMaskNode,
    SplitFacesNode,
    StackFacesNode,
    StereoToMonoScopicNode,
)

NODE_CLASS_MAPPINGS = {
    # Equirectangular Single Face
    "Equirectangular to Face": E2Face,
    "Equirectangular Mask to Face": EMask2Face,
    "Face to Equirectangular": Face2E,
    "Face Mask to Equirectangular": Face2EMask,
    # Equirectangular Full
    "Cubemap to Equirectangular": C2ENode,
    "Equirectangular to Cubemap": E2CNode,
    "Equirectangular to Perspective": E2PNode,
    "Equirectangular Rotation": E2ENode,
    "Mask Equirectangular Rotation": MaskE2ENode,
    "Pad 180 to 360 Equirectangular": Pad180To360Node,
    "Crop 360 to 180 Equirectangular": Crop360To180Node,
    # Cubemap
    "Split Cubemap Faces": SplitFacesNode,
    "Stack Cubemap Faces": StackFacesNode,
    # Miscellaneous
    "Roll Image Axes": RollImageNode,
    "Roll Mask Axes": RollMaskNode,
    "Masked Diff C2E": C2EMaskedDiffNode,
    "Crop Image with Coords": CropImageWithCoordsNode,
    "Paste Image with Coords": PasteImageWithCoordsNode,
    # Stereo
    "Crop Stereo to Monoscopic": StereoToMonoScopicNode,
    "Merge Monoscopic into Stereo": StereoToMonoScopicNode,
    # Models
    "Apply Circular Padding VAE": ApplyCircularConvPaddingVAE,
    "Apply Circular Padding Model": ApplyCircularConvPaddingModel,
    # Masks
    "Create Seam Mask": CreateSeamMask,
    "Create Pole Mask": CreatePoleMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Equirectangular Single Face
    "Equirectangular to Face": "Equirectangular to Face",
    "Equirectangular Mask to Face": "Equirectangular Mask to Face",
    "Face to Equirectangular": "Face to Equirectangular",
    "Face Mask to Equirectangular": "Face Mask to Equirectangular",
    # Equirectangular Full
    "Cubemap to Equirectangular": "Cubemap to Equirectangular",
    "Equirectangular to Cubemap": "Equirectangular to Cubemap",
    "Equirectangular to Perspective": "Equirectangular to Perspective",
    "Equirectangular Rotation": "Equirectangular Rotation",
    "Mask Equirectangular Rotation": "Mask Equirectangular Rotation",
    "Pad 180 to 360 Equirectangular": "Pad 180 to 360 Equirectangular",
    "Crop 360 to 180 Equirectangular": "Crop 360 to 180 Equirectangular",
    # Cubemap
    "Split Cubemap Faces": "Split Cubemap Faces",
    "Stack Cubemap Faces": "Stack Cubemap Faces",
    # Miscellaneous
    "Roll Image Axes": "Roll Image Axes",
    "Roll Mask Axes": "Roll Mask Axes",
    "Masked Diff C2E": "Masked Diff C2E",
    "Crop Image with Coords": "Crop Image with Coords",
    "Paste Image with Coords": "Paste Image with Coords",
    # Stereo
    "Crop Stereo to Monoscopic": "Crop Stereo to Monoscopic",
    "Merge Monoscopic into Stereo": "Merge Monoscopic into Stereo",
    # Models
    "Apply Circular Padding VAE": "Apply Circular Padding VAE",
    "Apply Circular Padding Model": "Apply Circular Padding Model",
    # Masks
    "Create Seam Mask": "Create Seam Mask",
    "Create Pole Mask": "Create Pole Mask",
}
