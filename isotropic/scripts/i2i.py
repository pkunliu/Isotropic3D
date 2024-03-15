import os
import sys
import random
import argparse
import numpy as np
from omegaconf import OmegaConf
import torch 

from camera_utils import get_camera
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange
from skimage.io import imread

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_im(path):
    img = imread(path)
    img = img.astype(np.float32) / 255.0
    mask = img[:,:,3:]
    img[:,:,:3] = img[:,:,:3] * mask + (1 - mask) # white background
    img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))

    im = img.convert("RGB")
    img = im.resize((256, 256), resample=Image.Resampling.BICUBIC)
    return img

def process_im(im):
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    tf = transforms.Compose(image_transforms)
    return tf(im)

def i2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):


    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device) #
        # c_concat = model.get_first_stage_encoding(model.encode_first_stage(prompt)).detach().repeat(batch_size,1,1,1)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default="configs/mvd_infer.yaml", help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default="", help="path to local checkpoint")
    parser.add_argument("--prompt", type=str, default="xhr_rgba.png")
    parser.add_argument("--save_name", type=str, default="images")
    parser.add_argument("--random_bg", action="store_true")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=5)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)

    print("load i2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.device = device
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    print("load i2i model done . ")

    # pre-compute camera matrices
    camera = get_camera(args.num_frames, elevation=args.camera_elev).to(device)
    set_seed(args.seed)
    assert args.prompt.endswith(".png") or args.prompt.endswith(".jpg"), "image should be png or jpg"

    input_image = load_im(args.prompt)

    c = process_im(input_image).unsqueeze(0).permute(0,3,1,2).to(device)
    img = i2i(model, args.size, c, uc, sampler, step=50, scale=10., batch_size=batch_size, ddim_eta=1.0, 
            dtype=dtype, device=device, camera=camera, num_frames=args.num_frames)
        
    images = np.concatenate(img, 1)
    Image.fromarray(images).save(f"{args.save_name}.png")