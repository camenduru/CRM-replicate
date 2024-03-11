import os
from cog import BasePredictor, Input, Path
from typing import List
import sys, shutil
sys.path.append('/content/CRM-hf')
os.chdir('/content/CRM-hf')

import numpy as np
from omegaconf import OmegaConf
import torch
from PIL import Image
import PIL
from pipelines import TwoStagePipeline
import rembg
from typing import Any
import json

from model import CRM
from inference import generate3d

rembg_session = rembg.new_session()

def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def check_input_image(input_image):
    if input_image is None:
        print("No image uploaded!")

def remove_background(
    image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        # explain why current do not rm bg
        print("alhpa channl not enpty, skip remove background, using alpha channel as mask")
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)

def preprocess_image(image, background_choice, foreground_ratio, backgroud_color):
    """
    input image is a pil image in RGBA, return RGB image
    """
    print(background_choice)
    if background_choice == "Alpha as mask":
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image = Image.alpha_composite(background, image)
    else:
        image = remove_background(image, rembg_session, force=True)
    image = do_resize_content(image, foreground_ratio)
    image = expand_to_square(image)
    image = add_background(image, backgroud_color)
    return image.convert("RGB")

def gen_image(input_image, seed, scale, step, pipeline, model):
    pipeline.set_seed(seed)
    rt_dict = pipeline(input_image, scale=scale, step=step)
    stage1_images = rt_dict["stage1_images"]
    stage2_images = rt_dict["stage2_images"]
    np_imgs = np.concatenate(stage1_images, 1)
    np_xyzs = np.concatenate(stage2_images, 1)
    glb_path = generate3d(model, np_imgs, np_xyzs, "cuda:0")
    return Image.fromarray(np_imgs), Image.fromarray(np_xyzs), glb_path#, obj_path

class Predictor(BasePredictor):
    def setup(self) -> None:
        crm_path = "/content/models/CRM.pth"
        specs = json.load(open("configs/specs_objaverse_total.json"))
        self.model = CRM(specs)
        self.model.load_state_dict(torch.load(crm_path, map_location="cpu"), strict=False)
        self.model = self.model.to("cuda:0")

        stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
        stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
        stage2_sampler_config = stage2_config.sampler
        stage1_sampler_config = stage1_config.sampler
        stage1_model_config = stage1_config.models
        stage2_model_config = stage2_config.models
        xyz_path = "/content/models/ccm-diffusion.pth"
        pixel_path = "/content/models/pixel-diffusion.pth"
        stage1_model_config.resume = pixel_path
        stage2_model_config.resume = xyz_path

        self.pipeline = TwoStagePipeline(
            stage1_model_config,
            stage2_model_config,
            stage1_sampler_config,
            stage2_sampler_config,
            device="cuda:0",
            dtype=torch.float32
        )
    def predict(
        self,
        image_path: Path = Input(description="Input Image"),
        foreground_ratio: float = Input(default=1.0, ge=0.5, le=1.0),
        back_groud_color: str = Input(default="#7F7F7F"),
        background_choice: str = Input(choices=["Alpha as mask", "Auto Remove background"], default="Auto Remove background"),
        seed: int = Input(default=1234),
        steps: int = Input(default=30),
        scale: float = Input(default=5.5),
    ) -> Path:
        image = Image.open(image_path)
        processed_image = preprocess_image(image, background_choice, foreground_ratio, back_groud_color)
        output_model = gen_image(processed_image, seed, scale, steps, self.pipeline, self.model)
        return Path(output_model[2])
