
import os
import argparse
import numpy as np
import random
import torch
import threestudio
import gc
from tqdm import tqdm

from glob import glob
import cv2
from torchvision.utils import save_image
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="ckpt/isotropic3d.ckpt", help="load pre-trained model")
    parser.add_argument("--pretrained_config", type=str, default="isotropic/configs/mvd_infer.yaml", help="load model from local config (override model_name)")
    parser.add_argument("--image_path", type=str, default="", help="image path")
    parser.add_argument("--save_path", type=str, default="output.png", help="save path")
    args = parser.parse_args()

    seed_everything(22)
    config = {
        'guidance_type': "multiview-diffusion-guidance",
        'guidance': {
            'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
            'pretrained_config': args.pretrained_config,
            'vram_O': False,
            "cond_image_path": args.image_path,
            'guidance_scale': 10.,
        },
    }

    guidance = threestudio.find(config['guidance_type'])(config['guidance'])

    image = guidance.generate(args.image_path)
    save_image(image, args.save_path)

    print("done!")