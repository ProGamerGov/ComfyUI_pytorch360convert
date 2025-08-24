import copy
from typing import Dict, List, Tuple

import torch
from pytorch360convert import c2e, e2c, e2e, e2p


class C2ENode:
    """
    Cubemap To Equirectangular Node
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "e_img": ("IMAGE", {"default": None}),
                "h": ("INT", {"default": -1}),
                "w": ("INT", {"default": -1}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "cube_format": (
                    ["stack", "dice", "horizon", "list", "dict"],
                    {"default": "stack"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Equirectangular Image",)

    FUNCTION = "c2e"

    CATEGORY = "pytorch360convert/equirectangular"

    def c2e(
        self,
        e_img: torch.Tensor,
        h=None,
        w=None,
        padding_mode: str = "bilinear",
        cube_format: str = "stack",
    ) -> Tuple[torch.Tensor]:
        assert e_img.shape[0] == 6, f"Input should have 6 faces, got {e_img.shape[0]}"
        h = None if h < 1 else h
        w = None if w < 1 else w
        return (
            c2e(
                e_img.squeeze(0),
                h=h,
                w=w,
                cube_format=cube_format,
                mode=padding_mode,
                channels_first=False,
            ).unsqueeze(0),
        )


class E2CNode:
    """
    Equirectangular to Cubemap Node
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "e_img": ("IMAGE", {"default": None}),
                "face_width": ("INT", {"default": -1}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "cube_format": (
                    ["stack", "dice", "horizon", "list", "dict"],
                    {"default": "stack"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cubemap Image",)

    FUNCTION = "e2c"

    CATEGORY = "pytorch360convert/equirectangular"

    def e2c(
        self,
        e_img: torch.Tensor,
        face_width: int = -1,
        padding_mode: str = "bilinear",
        cube_format: str = "stack",
    ) -> Tuple[torch.Tensor]:
        assert e_img.shape[0] == 1, (
            "Only a batch size of 1 is currently" + f"supported, got {e_img.shape[0]}"
        )
        face_width = e_img.shape[1] // 2 if face_width < 1 else face_width
        output = e2c(
            e_img.squeeze(0),
            face_w=face_width,
            mode=padding_mode,
            cube_format=cube_format,
            channels_first=False,
        )
        output = output.unsqueeze(0) if output.dim() == 3 else output
        return (output,)


class E2PNode:
    """
    Equirectangular to Perspective Node
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "e_img": ("IMAGE", {"default": None}),
                "fov_deg_h": ("FLOAT", {"default": 90.0}),
                "fov_deg_v": ("FLOAT", {"default": 90.0}),
                "h_deg": ("FLOAT", {"default": 0.0}),
                "v_deg": ("FLOAT", {"default": 0.0}),
                "out_h": ("INT", {"default": 512}),
                "out_w": ("INT", {"default": 512}),
                "in_rot_deg": ("FLOAT", {"default": 0.0}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Perspective Image",)

    FUNCTION = "e2p"

    CATEGORY = "pytorch360convert/equirectangular"

    def e2p(
        self,
        e_img: torch.Tensor,
        fov_deg_h: float = 90.0,
        fov_deg_v: float = 90.0,
        h_deg: float = 0.0,
        v_deg: float = 0.0,
        out_h: int = 512,
        out_w: int = 512,
        in_rot_deg: float = 0.0,
        padding_mode: str = "bilinear",
    ) -> Tuple[torch.Tensor]:
        assert e_img.dim() == 4, f"e_img should have 4 dimensions, got {e_img.dim()}"
        return (
            e2p(
                e_img,
                fov_deg=(fov_deg_h, fov_deg_v),
                h_deg=h_deg,
                v_deg=v_deg,
                out_hw=(out_h, out_w),
                in_rot_deg=in_rot_deg,
                mode=padding_mode,
                channels_first=False,
            ),
        )


class E2ENode:
    """
    Equirectangular Rotation Node
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "e_img": ("IMAGE", {"default": None}),
                "roll": ("FLOAT", {"default": 0.0}),
                "h_deg": ("FLOAT", {"default": 0.0}),
                "v_deg": ("FLOAT", {"default": 0.0}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Rotated Image",)

    FUNCTION = "e2e"

    CATEGORY = "pytorch360convert/equirectangular"

    def e2e(
        self,
        e_img: torch.Tensor,
        roll: float = 0.0,
        h_deg: float = 0.0,
        v_deg: float = 0.0,
        padding_mode: str = "bilinear",
    ) -> Tuple[torch.Tensor]:
        assert e_img.dim() == 4, f"e_img should have 4 dimensions, got {e_img.dim()}"
        return (
            e2e(
                e_img=e_img,
                h_deg=h_deg,
                v_deg=v_deg,
                roll=roll,
                mode=padding_mode,
                channels_first=False,
            ),
        )


class SplitFacesNode:
    """
    Split a stack of cube faces for easy manipulation.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "face_stack": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Front", "Right", "Back", "Left", "Up", "Down")

    FUNCTION = "split_faces"

    CATEGORY = "pytorch360convert/miscellaneous"

    def split_faces(self, face_stack: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        assert (
            face_stack.dim() == 4
        ), f"face_stack should have 4 dimensions, got {face_stack.dim()}"
        assert (
            face_stack.shape[0] == 6
        ), f"face_stack should have 6 faces, got {face_stack.shape[0]}"
        outputs = [face_stack[i].unsqueeze(0) for i in range(face_stack.shape[0])]
        return tuple(outputs)


class StackFacesNode:
    """
    Combine multiple cube faces into a stack.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "Front": ("IMAGE", {"default": None}),
                "Right": ("IMAGE", {"default": None}),
                "Back": ("IMAGE", {"default": None}),
                "Left": ("IMAGE", {"default": None}),
                "Up": ("IMAGE", {"default": None}),
                "Down": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cubemap stack",)

    FUNCTION = "stack_faces"

    CATEGORY = "pytorch360convert/miscellaneous"

    def stack_faces(
        self,
        Front: torch.Tensor,
        Right: torch.Tensor,
        Back: torch.Tensor,
        Left: torch.Tensor,
        Up: torch.Tensor,
        Down: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        assert (
            Front.dim() == 4
            and Right.dim() == 4
            and Back.dim() == 4
            and Left.dim() == 4
            and Up.dim() == 4
            and Down.dim() == 4
        )
        return (torch.cat([Front, Right, Back, Left, Up, Down], 0),)


class RollImageNode:
    """
    Roll an image to make face seams easier to remove or access.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "roll_x": ("INT", {"default": 0}),
                "roll_y": ("INT", {"default": 0}),
                "roll_x_by_50_percent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Ignores roll_x and roll_y. Shifts image horizontally by 50%.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Rolled Image",)

    FUNCTION = "roll_image"

    CATEGORY = "pytorch360convert/miscellaneous"

    def roll_image(
        self,
        image: torch.Tensor,
        roll_x: int = 0,
        roll_y: int = 0,
        roll_x_by_50_percent: bool = False,
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, f"image should have 4 dimensions, got {image.dim()}"
        if roll_x_by_50_percent:
            _, H, W, _ = image.shape
            px_half = W // 2
            roll_y = 0
            roll_x = px_half
        return (torch.roll(image, shifts=(roll_y, roll_x), dims=(1, 2)),)


class C2EMaskedDiffNode:
    """
    Cubemap To Equirectangular with Masked Diff Node
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "original_faces": ("IMAGE", {"default": None}),
                "modified_faces": ("IMAGE", {"default": None}),
                "original_equi": ("IMAGE", {"default": None}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "cube_format": (
                    ["stack", "dice", "horizon", "list", "dict"],
                    {"default": "stack"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Equirectangular Image",)

    FUNCTION = "c2e_masked_diff"

    CATEGORY = "pytorch360convert/mask"

    def c2e_masked_diff(
        self,
        original_faces: torch.Tensor,
        modified_faces: torch.Tensor,
        original_equi: torch.Tensor,
        padding_mode: str = "bilinear",
        cube_format: str = "stack",
    ) -> Tuple[torch.Tensor]:
        assert original_equi.shape[0] == 1, (
            "Only a batch size of 1 is currently supported"
            + f"for original_equi, got {original_equi.shape[0]}"
        )
        assert (
            original_faces.shape[0] == 6
        ), f"original_faces should have 6 faces, got {original_faces.shape[0]}"
        assert (
            modified_faces.shape[0] == 6
        ), f"modified_faces should have 6 faces, got {modified_faces.shape[0]}"
        new_equi = c2e(modified_faces, cube_format=cube_format, channels_first=False)

        faces_mask = original_faces != modified_faces
        mask_equi = c2e(
            faces_mask.float(), cube_format=cube_format, channels_first=False
        ).bool()
        return (torch.where(mask_equi, new_equi, original_equi),)


class CropImageWithCoordsNode:
    """
    Crop an section of an image for manipulation
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "crop_h": ("INT", {"default": 0}),
                "crop_w": ("INT", {"default": 0}),
                "crop_h2": ("INT", {"default": -1}),
                "crop_w2": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "LIST",
    )
    RETURN_NAMES = (
        "Cropped Image",
        "Coords",
    )

    FUNCTION = "crop_image"

    CATEGORY = "pytorch360convert/miscellaneous"

    def crop_image(
        self,
        image: torch.Tensor,
        crop_h: int = 0,
        crop_w: int = 0,
        crop_h2: int = -1,
        crop_w2: int = -1,
    ) -> Tuple[torch.Tensor, List[int]]:
        assert image.dim() == 4, f"image should have 4 dimensions, got {image.dim()}"

        _, H, W, _ = image.shape

        # Calculate the center crop indices
        if crop_h2 < 1:
            start_h = (H - crop_h) // 2
            end_h = start_h + crop_h
        else:
            start_h = crop_h
            end_h = crop_h2

        if crop_w2 < 1:
            start_w = (W - crop_w) // 2
            end_w = start_w + crop_w
        else:
            start_w = crop_w
            end_w = crop_w2

        cropped_tensor = image[:, start_h:end_h, start_w:end_w, :]

        coords = [start_h, end_h, start_w, end_w]
        return (
            cropped_tensor,
            coords,
        )


class PasteImageWithCoordsNode:
    """
    Add a cropped section back to the original image.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "full_image": ("IMAGE", {"default": None}),
                "cropped_image": ("IMAGE", {"default": None}),
                "coords": ("LIST", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Pasted Image",)

    FUNCTION = "paste_image"

    CATEGORY = "pytorch360convert/miscellaneous"

    def paste_image(
        self, full_image: torch.Tensor, cropped_image: torch.Tensor, coords: List[int]
    ) -> Tuple[torch.Tensor]:
        assert (
            full_image.dim() == 4
        ), f"full_image should have 4 dimensions, got {full_image.dim()}"
        assert (
            cropped_image.dim() == 4
        ), f"cropped_image should have 4 dimensions, got {cropped_image.dim()}"
        assert len(coords) == 4
        start_h, end_h, start_w, end_w = coords
        full_image[:, start_h:end_h, start_w:end_w, :] = cropped_image
        return (full_image,)


class Pad180To360Node:
    """
    Apply padding to a 180 degree equirectangular image, to make it 360 degrees.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "fill_value": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Padded 360 Image",)

    FUNCTION = "pad_180_to_360_image"

    CATEGORY = "pytorch360convert/equirectangular"

    def pad_180_to_360_image(
        self, image: torch.Tensor, fill_value: float = 0.0
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, f"image should have 4 dimensions, got {image.dim()}"
        image = image.permute(0, 3, 1, 2)
        H, W = image.shape[2:]
        pad_left = W // 2
        pad_right = W - pad_left

        image_padded = torch.nn.functional.pad(
            image, (pad_left, pad_right), mode="constant", value=fill_value
        )

        image_padded = image_padded.permute(0, 2, 3, 1)
        return (image_padded,)


class Crop360To180Node:
    """
    Crop a 360 degree equirectangular image to the central 180 degree part.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped 180 Image",)

    FUNCTION = "crop_360_to_180_image"

    CATEGORY = "pytorch360convert/equirectangular"

    def crop_360_to_180_image(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Crop a 360-degree equirectangular image to the central 180-degree part.

        Args:
            image (torch.Tensor): The 360-degree equirectangular image. Shape should
                be: (B, H, W, C).

        Returns:
            torch.Tensor: The cropped 180-degree equirectangular image. Shape will be:
                (B, H, W//2, C) where the width is halved.

        Raises:
            ValueError: If the input image width is less than or equal to 1.
        """
        assert image.dim() == 4, f"image should have 4 dimensions, got {image.dim()}"
        _, _, width, _ = image.shape

        # Crop the central 180-degree part by slicing the image.
        crop_start = width // 4
        crop_end = 3 * width // 4

        # Crop along the width dimension (left and right 180 degrees)
        cropped_img = image[:, :, crop_start:crop_end, :]

        return (cropped_img,)


class StereoToMonoScopicNode:
    """
    Split a stereo image into 2 monoscopic images.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "split_direction": (
                    ["horizontal", "vertical"],
                    {"default": "horizontal"},
                ),
                "larger_side": (["first", "second"], {"default": "first"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "First Image",
        "Second Image",
    )

    FUNCTION = "split_stereo_image"

    CATEGORY = "pytorch360convert/stereo"

    def split_stereo_image(
        self,
        image: torch.Tensor,
        split_direction: str = "horizontal",
        larger_side: str = "first",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits a batch of tensor images (BHWC format) into two halves either vertically
        or horizontally,  splitting exactly down the middle. In case of odd dimensions,
        one side gets an extra pixel.

        Args:
            image (torch.Tensor): The tensor image batch to split. Shape should be:
                (B, H, W, C).
            split_direction (str, optional): The direction to split the image. Either
                "horizontal" or "vertical". Default: "horizontal"
            larger_side (str, optional): Specifies which side gets the extra pixel when
                dimensions are odd. "first" gives the extra pixel to the first side
                (top/left), "second" gives it to the second side (bottom/right).
                Default: "first"

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the first and second
                halves of the split images. The shape of each tensor is:
                - For horizontal split: (B, H/2, W, C)
                - For vertical split: (B, H, W/2, C)

        Raises:
            ValueError: If `split_direction` is not "horizontal" or "vertical".
        """
        _, height, width, channels = image.shape

        if split_direction == "vertical":
            # Split horizontally (by height)
            mid_point = height // 2
            if height % 2 != 0:  # If height is odd
                if larger_side == "first":
                    first_half = image[:, : mid_point + 1, :, :]
                    second_half = image[:, mid_point + 1 :, :, :]
                else:  # "second"
                    first_half = image[:, :mid_point, :, :]
                    second_half = image[:, mid_point:, :]
            else:
                # Even height, just split in the middle
                first_half = image[:, :mid_point, :, :]
                second_half = image[:, mid_point:, :]
        elif split_direction == "horizontal":
            # Split vertically (by width)
            mid_point = width // 2
            if width % 2 != 0:  # If width is odd
                if larger_side == "first":
                    first_half = image[:, :, : mid_point + 1, :]
                    second_half = image[:, :, mid_point + 1 :, :]
                else:  # "second"
                    first_half = image[:, :, :mid_point, :]
                    second_half = image[:, :, mid_point:, :]
            else:
                # Even width, just split in the middle
                first_half = image[:, :, :mid_point, :]
                second_half = image[:, :, mid_point:, :]
        else:
            raise ValueError(
                "Invalid split direction. Please choose"
                + " 'horizontal' or 'vertical'. "
                + f"Got {split_direction}"
            )

        return (first_half, second_half)


class MonoScopicToStereoNode:
    """
    Merge two monoscopic images into a stereo image.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "first_image": ("IMAGE", {"default": None}),
                "second_image": ("IMAGE", {"default": None}),
                "merge_direction": (
                    ["horizontal", "vertical"],
                    {"default": "horizontal"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Stereo Image",)

    FUNCTION = "merge_monoscopic_to_stereo"

    CATEGORY = "pytorch360convert/stereo"

    def merge_monoscopic_to_stereo(
        self,
        first_image: torch.Tensor,
        second_image: torch.Tensor,
        merge_direction: str = "horizontal",
    ) -> Tuple[torch.Tensor]:
        """
        Merges two monoscopic images into a single stereo image by concatenating
        the two images either horizontally or vertically.

        Args:
            first_image (torch.Tensor): The left monoscopic image. Shape should
                be: (B, H, W, C).
            second_image (torch.Tensor): The right monoscopic image. Shape should
                be: (B, H, W, C).
            merge_direction (str, optional): The direction to merge the images. Either
                "horizontal" or "vertical". Default: "horizontal"

        Returns:
            torch.Tensor: The merged stereo image.

        Raises:
            ValueError: If `merge_direction` is not "horizontal" or "vertical".
        """

        f_batch_size, f_height, f_width, f_channels = first_image.shape
        s_batch_size, s_height, s_width, s_channels = second_image.shape

        if merge_direction == "horizontal":
            assert f_batch_size == s_batch_size, (
                "Batch size must match: " + f"{f_batch_size} vs {s_batch_size}"
            )
            assert f_height == s_height, (
                "Height must match: " + f"{f_height} vs {s_height}"
            )
            assert f_channels == s_channels, (
                "Channels must match: " + f"{f_channels} vs {s_channels}"
            )

            # Concatenate horizontally (by width)
            stereo_image = torch.cat((first_image, second_image), dim=2)

        elif merge_direction == "vertical":
            assert f_batch_size == s_batch_size, (
                "Batch size must match: " + f"{f_batch_size} vs {s_batch_size}"
            )
            assert f_width == s_width, "Width must match: " + f"{f_width} vs {s_width}"
            assert f_channels == s_channels, (
                "Channels must match: " + f"{f_channels} vs {s_channels}"
            )

            # Concatenate vertically (by height)
            stereo_image = torch.cat((first_image, second_image), dim=1)

        else:
            raise ValueError(
                "Invalid split direction. Please choose"
                + " 'horizontal' or 'vertical'. "
                + f"Got {merge_direction}"
            )

        return (stereo_image,)


def _conv_forward(
    self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    x = torch.nn.functional.pad(x, self.padding_values_x, mode="circular")
    x = torch.nn.functional.pad(x, self.padding_values_y, mode="constant")
    return torch.nn.functional.conv2d(
        x, weight, bias, self.stride, (0, 0), self.dilation, self.groups
    )


def _apply_circular_conv2d_padding(
    model: torch.nn.Module, is_vae: bool = False, x_axis_only: bool = True
) -> torch.nn.Module:
    for layer in model.first_stage_model.modules() if is_vae else model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            if x_axis_only:
                layer.padding_values_x = (
                    layer._reversed_padding_repeated_twice[0],
                    layer._reversed_padding_repeated_twice[1],
                    0,
                    0,
                )
                layer.padding_values_y = (
                    0,
                    0,
                    layer._reversed_padding_repeated_twice[2],
                    layer._reversed_padding_repeated_twice[3],
                )
                layer._conv_forward = _conv_forward.__get__(layer, torch.nn.Conv2d)
            else:
                layer.padding_mode = "circular"
    return model


class ApplyCircularConvPaddingModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "Model to add circular x-axis conv2d padding to."},
                ),
                "inplace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Modify "
                        + "the already loaded model (True) or a copy of the model (False). "
                        + "If True, model will have to be reloaded to restore padding to "
                        + "the original values. Modifying inplace will use less memory.",
                    },
                ),
                "x_axis_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply"
                        + " circular padding only to the x-axis or to both the x and y axes.",
                    },
                ),
            },
        }

    CATEGORY = "pytorch360convert/models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(
        self, model: torch.nn.Module, inplace: bool = True, x_axis_only: bool = True
    ) -> Tuple[torch.nn.Module]:
        if inplace:
            use_model = model
        else:
            use_model = copy.deepcopy(model)
        _apply_circular_conv2d_padding(use_model.model, False, x_axis_only)
        return (use_model,)


class ApplyCircularConvPaddingVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (
                    "VAE",
                    {"tooltip": "VAE to add circular x-axis conv2d padding to."},
                ),
                "inplace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Modify "
                        + "the already loaded VAE (True) or a copy of the VAE (False). "
                        + "If True, VAE will have to be reloaded to restore padding to "
                        + "the original values. Modifying inplace will use less memory.",
                    },
                ),
                "x_axis_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply"
                        + " circular padding only to the x-axis or to both the x and y axes.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "run"
    CATEGORY = "pytorch360convert/models"

    def run(
        self, vae: torch.nn.Module, inplace: bool = True, x_axis_only: bool = True
    ) -> Tuple[torch.nn.Module]:
        if inplace:
            use_vae = vae
        else:
            use_vae = copy.deepcopy(vae)
        _apply_circular_conv2d_padding(use_vae, True, x_axis_only)
        return (use_vae,)


def _create_center_seam_mask(
    x: torch.Tensor, frac_width: float = 0.10, feather: int = 0
) -> torch.Tensor:
    """
    For a ComfyUI-style mask: shape [B, H, W], values 0 or 1, with optional feathering.

    Args:
        x (torch.Tensor): input tensor with shape [B, H, W, C].
        frac_width (float, optional): fraction of input width for the vertical strip.
        feather (int, optional): pixel size of feathering on both sides of the mask.

    Returns:
        mask: torch.Tensor of shape [B, H, W] with float values 0.0 to 1.0.
    """
    # Extract batch, height, and width from x
    B, H, W, *_ = x.shape

    strip = max(1, int(W * frac_width))
    x0 = (W - strip) // 2
    x1 = x0 + strip

    # Create the mask with zeros
    mask = torch.zeros((B, H, W), dtype=x.dtype, device=x.device)

    if feather <= 0:
        mask[:, :, x0:x1] = 1.0
    else:
        # Create feathered mask
        # Left feather region
        left_feather_start = max(0, x0 - feather)
        left_feather_end = x0
        if left_feather_end > left_feather_start:
            feather_steps = torch.linspace(
                0.0,
                1.0,
                left_feather_end - left_feather_start,
                dtype=x.dtype,
                device=x.device,
            )
            mask[:, :, left_feather_start:left_feather_end] = feather_steps[
                None, None, :
            ]

        # Center region (full mask)
        mask[:, :, x0:x1] = 1.0

        # Right feather region
        right_feather_start = x1
        right_feather_end = min(W, x1 + feather)
        if right_feather_end > right_feather_start:
            feather_steps = torch.linspace(
                1.0,
                0.0,
                right_feather_end - right_feather_start,
                dtype=x.dtype,
                device=x.device,
            )
            mask[:, :, right_feather_start:right_feather_end] = feather_steps[
                None, None, :
            ]

    return mask


class CreateSeamMask:
    """
    Create a seam mask for inpainting on equirectangular images.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "seam_mask_width": ("FLOAT", {"default": 0.10}),
                "feather": (
                    "INT",
                    {"default": 0, "tooltip": "Pixel size of the feathering."},
                ),
                "roll_x_by_50_percent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Shifts the output mask horizontally by 50%."
                        + " Equivalent to a 180 degree rotation on an equirectangular image.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Seam Mask",)

    FUNCTION = "run"

    CATEGORY = "pytorch360convert/mask"

    def run(
        self,
        image: torch.Tensor,
        seam_mask_width: float = 0.10,
        feather: int = 0,
        roll_x_by_50_percent: bool = False,
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, "Image should have 4 dimensions"
        _, H, W, _ = image.shape
        px_half = W // 2
        seam_mask = _create_center_seam_mask(
            image, frac_width=seam_mask_width, feather=feather
        )

        if roll_x_by_50_percent:
            seam_mask = torch.roll(seam_mask, shifts=(0, px_half), dims=(1, 2))
        return (seam_mask,)


class E2Face:
    """
    Equirectangular to Face
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "e_img": ("IMAGE", {"default": None}),
                "face_width": ("INT", {"default": -1}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "cube_face": (
                    ["Up", "Down", "Right", "Left", "Front", "Back"],
                    {"default": "Front"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Face Image",)

    FUNCTION = "run_e2face"

    CATEGORY = "pytorch360convert/equirectangular"

    def run_e2face(
        self,
        e_img: torch.Tensor,
        face_width: int = -1,
        padding_mode: str = "bilinear",
        cube_face: str = "Front",
    ) -> Tuple[torch.Tensor]:

        B, H, W, C = e_img.shape
        outputs = []

        for i in range(B):
            single_img = e_img[i]  # [H,W,C]
            # Determine face width
            face_w = H // 2 if face_width < 1 else face_width

            # Convert single equirectangular image to cubemap dict
            cubemap = e2c(
                single_img,
                face_w=face_w,
                mode=padding_mode,
                cube_format="dict",
                channels_first=False,
            )

            # Pick requested face
            face_tensor = cubemap[cube_face]  # [face_w, face_w, C]

            outputs.append(face_tensor.unsqueeze(0))  # [1,H,W,C]

        # Concatenate into [B,H,W,C]
        output_batch = torch.cat(outputs, dim=0)
        return (output_batch,)


def _create_centered_circle_mask(
    x: torch.Tensor, circle_radius: float, feather: int = 0
) -> torch.Tensor:
    """
    Create a centered circle mask with optional feathering.

    Args:
        x (torch.Tensor): Reference tensor with shape (B, C, H, W).
                          The mask will copy device, dtype, and batch size from x.
        circle_radius (float): Fraction (0.0–1.0) of max possible radius (min(H,W)/2).
                               1.0 = circle touches edges.
        feather (int): Feather width in pixels, extending outward from circle_radius.
                       0 = hard edge, >0 = smooth falloff.

    Returns:
        torch.Tensor: Mask tensor of shape (1, C, H, W), values in [0.0, 1.0].
    """
    _, C, H, W = x.shape
    size = min(H, W)

    # Circle radius in pixels
    max_radius = size / 2.0
    inner_radius = circle_radius * max_radius
    outer_radius = inner_radius + feather

    # Coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(size, device=x.device),
        torch.arange(size, device=x.device),
        indexing="ij",
    )
    center = size // 2
    dist = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)

    # Base mask
    mask = torch.zeros_like(dist, dtype=x.dtype, device=x.device)

    # Inside circle = 1.0
    mask[dist <= inner_radius] = 1.0

    # Feather transition
    if feather > 0:
        transition_zone = (dist > inner_radius) & (dist <= outer_radius)
        mask[transition_zone] = 1.0 - (dist[transition_zone] - inner_radius) / feather

    # Expand to BCHW and center in full HxW
    mask_full = torch.zeros((1, C, H, W), dtype=x.dtype, device=x.device)
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    mask_full[:, :, y0 : y0 + size, x0 : x0 + size] = mask

    return mask_full


class CreatePoleMask:
    """
    Create a pole mask for inpainting on equirectangular images.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "circle_radius": ("FLOAT", {"default": 0.10}),
                "feather": ("INT", {"default": 0}),
                "mode": (["face", "equirectangular"], {"default": "face"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Seam Mask",)

    FUNCTION = "run"

    CATEGORY = "pytorch360convert/mask"

    def run(
        self,
        image: torch.Tensor,
        circle_radius: float = 0.10,
        feather: int = 0,
        mode: str = "face",
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, f"Image should have 4 dimensions, got {image.shape}"

        if mode == "face":
            image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
            output_mask = _create_centered_circle_mask(image, circle_radius, feather)
            output_mask = output_mask
        else:
            output_mask = []
            for im in image:  # im: [H,W,C]
                # Convert single equirectangular image to cubemap dict
                cubemap = e2c(
                    torch.zeros_like(im), cube_format="dict", channels_first=False
                )

                # Apply circle mask on Up and Down faces
                for face in ["Up", "Down"]:
                    face_tensor = cubemap[face]  # [H_face, W_face, C]
                    face_tensor = face_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                    face_tensor = _create_centered_circle_mask(
                        face_tensor, circle_radius, feather
                    )
                    cubemap[face] = face_tensor.permute(0, 2, 3, 1).squeeze(0)

                # Convert back to equirectangular
                new_equi = c2e(
                    cubemap, cube_format="dict", channels_first=False
                ).unsqueeze(
                    0
                )  # add batch dim
                output_mask.append(new_equi)

            # Stack all results into a single tensor [B,H,W,C]
            output_mask = torch.cat(output_mask).permute(0, 3, 1, 2)

        return (output_mask[:, 0, ...],)


class Face2E:
    """
    Face To Equirectangular Node
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "face": (
                    ["Up", "Down", "Front", "Right", "Left", "Back"],
                    {"default": "Down"},
                ),
                "base_equi_color": ("FLOAT", {"default": 0.0}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Equirectangular Image",)

    FUNCTION = "run_face2e"

    CATEGORY = "pytorch360convert/equirectangular"

    def run_face2e(
        self,
        image: torch.Tensor,
        face: str = "Down",
        base_equi_color: float = 0.0,
        padding_mode: str = "bilinear",
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, f"Image should have 4 dimensions, got {image.shape}"

        output_image = []
        for f_image in image:

            cubemap_dict = {}
            for face_name in ["Front", "Right", "Back", "Left", "Up", "Down"]:
                if face_name != face:
                    cubemap_dict[face_name] = torch.ones_like(f_image) * base_equi_color
                else:
                    cubemap_dict[face_name] = f_image
            output_image += [
                c2e(
                    cubemap=cubemap_dict,
                    cube_format="dict",
                    mode=padding_mode,
                    channels_first=False,
                )
            ]
        return (torch.stack(output_image),)


class RollMaskNode:
    """
    Roll a mask to make face seams easier to remove or access.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "mask": ("MASK", {"default": None}),
                "roll_x": ("INT", {"default": 0}),
                "roll_y": ("INT", {"default": 0}),
                "roll_x_by_50_percent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Shifts the input mask horizontally by 50%."
                        + " Equivalent to a 180 degree rotation on an equirectangular image.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Rolled Mask",)

    FUNCTION = "roll_mask"

    CATEGORY = "pytorch360convert/mask"

    def roll_mask(
        self,
        mask: torch.Tensor,
        roll_x: int = 0,
        roll_y: int = 0,
        roll_x_by_50_percent: bool = False,
    ) -> Tuple[torch.Tensor]:
        assert mask.dim() == 3, f"mask should have 3 dimensions, got {mask.dim()}"
        if roll_x_by_50_percent:
            _, H, W = mask.shape
            px_half = W // 2
            roll_y = 0
            roll_x = px_half
        return (torch.roll(mask, shifts=(roll_y, roll_x), dims=(1, 2)),)


class FaceMask2E:
    """
    Face Mask To Equirectangular
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "mask": ("MASK", {"default": None}),
                "face": (
                    ["Up", "Down", "Front", "Right", "Left", "Back"],
                    {"default": "Down"},
                ),
                "base_equi_color": ("FLOAT", {"default": 0.0}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Equirectangular Mask",)

    FUNCTION = "run_facemask2e"

    CATEGORY = "pytorch360convert/mask"

    def run_facemask2e(
        self,
        mask: torch.Tensor,
        face: str = "Down",
        base_equi_color: float = 0.0,
        padding_mode: str = "bilinear",
    ) -> Tuple[torch.Tensor]:
        assert mask.dim() == 3, f"Image should have 3 dimensions, got {mask.shape}"

        output_mask = []
        for f_mask in mask:
            f_mask = f_mask = f_mask[None, ...]
            cubemap_dict = {}
            for face_name in ["Front", "Right", "Back", "Left", "Up", "Down"]:
                if face_name != face:
                    cubemap_dict[face_name] = torch.ones_like(f_mask) * base_equi_color
                else:
                    cubemap_dict[face_name] = mask.reshape(1, *mask.shape[1:])
            output_mask += [
                c2e(
                    cubemap=cubemap_dict,
                    cube_format="dict",
                    mode=padding_mode,
                    channels_first=True,
                )
            ]
        return (torch.cat(output_mask),)


class EMask2Face:
    """
    Equirectangular Mask to Face
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "mask": ("MASK", {"default": None}),
                "face_width": ("INT", {"default": -1}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
                "cube_face": (
                    ["Up", "Down", "Right", "Left", "Front", "Back"],
                    {"default": "Front"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Face MASK",)

    FUNCTION = "run_emask2face"

    CATEGORY = "pytorch360convert/mask"

    def run_emask2face(
        self,
        mask: torch.Tensor,
        face_width: int = -1,
        padding_mode: str = "bilinear",
        cube_face: str = "Front",
    ) -> Tuple[torch.Tensor]:

        B, H, W = mask.shape
        outputs = []

        for i in range(B):
            singlmask = mask[i : i + 1][..., None]  # [H,W,C]
            # Determine face width
            face_w = H // 2 if face_width < 1 else face_width

            # Convert single equirectangular mask to cubemap dict
            cubemap = e2c(
                singlmask,
                face_w=face_w,
                mode=padding_mode,
                cube_format="dict",
                channels_first=False,
            )

            # Pick requested face
            face_tensor = cubemap[cube_face]  # [face_w, face_w, C]

            outputs.append(face_tensor)  # [1,H,W,C]

        # Concatenate into [B,H,W,C]
        output_batch = torch.cat(outputs, dim=0)[..., 0]
        return (output_batch,)


class MaskE2ENode:
    """
    Mask Equirectangular Rotation
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "mask": ("MASK", {"default": None}),
                "roll": ("FLOAT", {"default": 0.0}),
                "h_deg": ("FLOAT", {"default": 0.0}),
                "v_deg": ("FLOAT", {"default": 0.0}),
                "padding_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {"default": "bilinear"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Rotated Mask",)

    FUNCTION = "mask_e2e"

    CATEGORY = "pytorch360convert/mask"

    def mask_e2e(
        self,
        mask: torch.Tensor,
        roll: float = 0.0,
        h_deg: float = 0.0,
        v_deg: float = 0.0,
        padding_mode: str = "bilinear",
    ) -> Tuple[torch.Tensor]:
        assert mask.dim() == 3, f"mask should have 3 dimensions, got {mask.dim()}"
        return (
            e2e(
                e_img=mask[..., None],
                h_deg=h_deg,
                v_deg=v_deg,
                roll=roll,
                mode=padding_mode,
                channels_first=False,
            )[..., 0],
        )


class Create180To360Mask:
    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "input_mode": (
                    ["360", "180"],
                    {"default": "180"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Equirectangular Mask",)

    FUNCTION = "mask_180_to_360"

    CATEGORY = "pytorch360convert/mask"

    def mask_180_to_360(
        self, image: torch.Tensor, input_mode: str = "180"
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, f"image should have 4 dimensions, got {image.dim()}"
        _, H, W, _ = image.shape
        if input_mode == "360":
            W = W // 2
        pad_left = W // 2
        pad_right = W - pad_left

        mask = torch.ones(1, 1, H, W, dtype=image.dtype, device=image.device)
        mask_padded = torch.nn.functional.pad(
            mask, (pad_left, pad_right), mode="constant", value=0.0
        )
        return (mask_padded[:, 0, ...],)
