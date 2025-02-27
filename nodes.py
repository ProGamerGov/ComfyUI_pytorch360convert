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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Rolled Image",)

    FUNCTION = "roll_image"

    CATEGORY = "pytorch360convert"

    def roll_image(
        self, image: torch.Tensor, roll_x: int = 0, roll_y: int = 0
    ) -> Tuple[torch.Tensor]:
        assert image.dim() == 4, f"image should have 4 dimensions, got {image.dim()}"
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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

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

    CATEGORY = "pytorch360convert"

    def pad_180_to_360_image(self, e_img: torch.Tensor, fill_value: float = 0.0) -> Tuple[torch.Tensor]:
        assert e_img.dim() == 4, f"image should have 4 dimensions, got {e_img.dim()}"
        e_img = e_img.permute(0, 3, 1, 2)
        H, W = e_img.shape[2:]
        pad_left = W // 2
        pad_right = W - pad_left

        e_img_padded = torch.nn.functional.pad(e_img, (pad_left, pad_right), mode="constant", value=fill_value)

        e_img_padded = e_img_padded.permute(0, 2, 3, 1)
        return (e_img_padded, )


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

    CATEGORY = "pytorch360convert"

    def crop_360_to_180_image(self, e_img: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Crop a 360-degree equirectangular image to the central 180-degree part.
        
        Args:
            e_img (torch.Tensor): The 360-degree equirectangular image. Shape should
                be: (B, H, W, C).
        
        Returns:
            torch.Tensor: The cropped 180-degree equirectangular image. Shape will be:
                (B, H, W//2, C) where the width is halved.
        
        Raises:
            ValueError: If the input image width is less than or equal to 1.
        """
        assert e_img.dim() == 4, f"image should have 4 dimensions, got {e_img.dim()}"
        _, _, width, _ = e_img.shape

        # Crop the central 180-degree part by slicing the image.
        crop_start = width // 4
        crop_end = 3 * width // 4

        # Crop along the width dimension (left and right 180 degrees)
        cropped_img = e_img[:, :, crop_start:crop_end, :]

        return (cropped_img,)


class StereoscopicToMonoScopic:
    """
    Split a stereoscopic image into 2 monoscopic images.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "split_direction": (["horizontal", "vertical"], {"default": "horizontal"}),
                "larger_side": (["first", "second"], {"default": "first"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "Image",
        "Image",
    )

    FUNCTION = "split_stereoscopic_image"  # Set the function name here

    CATEGORY = "pytorch360convert"

    def split_stereoscopic_image(
        self, image: torch.Tensor, split_direction: str = "horizontal", larger_side: str = "first"
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

        if split_direction == "horizontal":
            # Split horizontally (by height)
            mid_point = height // 2
            if height % 2 != 0:  # If height is odd
                if larger_side == "first":
                    first_half = image[:, :mid_point + 1, :, :]
                    second_half = image[:, mid_point + 1:, :, :]
                else:  # "second"
                    first_half = image[:, :mid_point, :, :]
                    second_half = image[:, mid_point:, :]
            else:
                # Even height, just split in the middle
                first_half = image[:, :mid_point, :, :]
                second_half = image[:, mid_point:, :]
        elif split_direction == "vertical":
            # Split vertically (by width)
            mid_point = width // 2
            if width % 2 != 0:  # If width is odd
                if larger_side == "first":
                    first_half = image[:, :, :mid_point + 1, :]
                    second_half = image[:, :, mid_point + 1:, :]
                else:  # "second"
                    first_half = image[:, :, :mid_point, :]
                    second_half = image[:, :, mid_point:, :]
            else:
                # Even width, just split in the middle
                first_half = image[:, :, :mid_point, :]
                second_half = image[:, :, mid_point:, :]
        else:
            raise ValueError("Invalid split direction. Please choose" + " 'horizontal' or 'vertical'. " + f"Got {split_direction}")

        return first_half, second_half

class MonoScopicToStereoscopic:
    """
    Merge two monoscopic images into a stereoscopic image.
    """

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "first_image": ("IMAGE", {"default": None}),
                "second_image": ("IMAGE", {"default": None}),
                "merge_direction": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "Stereoscopic Image",
    )

    FUNCTION = "merge_monoscopic_to_stereoscopic"

    CATEGORY = "pytorch360convert"

    def merge_monoscopic_to_stereoscopic(
        self, first_image: torch.Tensor, second_image: torch.Tensor, merge_direction: str = "horizontal"
    ) -> torch.Tensor:
        """
        Merges two monoscopic images into a single stereoscopic image by concatenating 
        the two images either horizontally or vertically.

        Args:
            first_image (torch.Tensor): The left monoscopic image. Shape should
                be: (B, H, W, C).
            second_image (torch.Tensor): The right monoscopic image. Shape should
                be: (B, H, W, C).
            merge_direction (str, optional): The direction to merge the images. Either
                "horizontal" or "vertical". Default: "horizontal"

        Returns:
            torch.Tensor: The merged stereoscopic image.

        Raises:
            ValueError: If `merge_direction` is not "horizontal" or "vertical".
        """

        f_batch_size, f_height, f_width, f_channels = first_image.shape
        s_batch_size, s_height, s_width, s_channels = second_image.shape

        if merge_direction == "horizontal":
            assert f_batch_size == s_batch_size, "Batch size must match: " + f"{f_batch_size} vs {s_batch_size}"
            assert f_height == s_height, "Height must match: " + f"{f_height} vs {s_height}"
            assert f_channels == s_channels, "Channels must match: " + f"{f_channels} vs {s_channels}"
            
            # Concatenate horizontally (by width)
            stereoscopic_image = torch.cat((first_image, second_image), dim=2)
            
        elif merge_direction == "vertical":
            assert f_batch_size == s_batch_size, "Batch size must match: " + f"{f_batch_size} vs {s_batch_size}"
            assert f_width == s_width, "Width must match: " + f"{f_width} vs {s_width}"
            assert f_channels == s_channels, "Channels must match: " + f"{f_channels} vs {s_channels}"
            
            # Concatenate vertically (by height)
            stereoscopic_image = torch.cat((first_image, second_image), dim=1)
            
        else:
            raise ValueError("Invalid split direction. Please choose" + " 'horizontal' or 'vertical'. " + f"Got {merge_direction}")

        return (stereoscopic_image,)
