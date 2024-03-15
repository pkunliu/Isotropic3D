import importlib
import os
import cv2
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from isotropic.camera_utils import convert_opengl_to_blender, normalize_camera, create_camera_to_world_matrix,get_camera

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=False, verbose=False):
    sd = torch.load(ckpt, map_location="cpu")

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()
    model.eval().to(device)

    return model

@threestudio.register("multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        pretrained_config: str = "isotropic/configs/mvd_infer.yaml"
        vram_O: bool = False
        cond_image_path: str = ""
        guidance_scale: float = 10.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = False
        recon_std_rescale: float = 0.5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")
        self.config = OmegaConf.load(self.cfg.pretrained_config)
        self.weights_dtype = torch.float32
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )
        
        # # timesteps: use diffuser for convenience... hope it's alright.
        # self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )


        self.alphas: Float[Tensor, "..."] = self.model.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.prepare_embeddings(self.cfg.cond_image_path)

        threestudio.info(f"Loaded Multiview Diffusion!")

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def prepare_embeddings(self, image_path: str) -> None:
        # load cond image for zero123
        assert os.path.exists(image_path)
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        self.rgb_256: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        self.c_crossattn = self.get_img_embeds(self.rgb_256)

    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype))
        # c_concat = self.model.get_first_stage_encoding(self.model.encode_first_stage(img.to(self.weights_dtype)))#no mode
        return c_crossattn


    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def get_camera_cond(self, 
            elevation: Float[Tensor, "B"],
            azimuth: Float[Tensor, "B"],
    ):
        camera = []
        for e, a in zip(elevation, azimuth):
            e = np.deg2rad(e.cpu().numpy())
            a = np.deg2rad(a.cpu().numpy())
            camera_matrix = create_camera_to_world_matrix(e, a)
            camera_matrix = convert_opengl_to_blender(camera_matrix)
            camera.append(camera_matrix.flatten())
        camera = torch.tensor(np.stack(camera, 0)).float().to(self.device)
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents  # [B, 4, 32, 32] Latent space image

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy = None,
        timestep=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        c_crossattn = self.c_crossattn.repeat(batch_size, 1, 1)
        c_crossattn = torch.cat([torch.zeros_like(c_crossattn).to(self.device), c_crossattn], dim=0)

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(rgb_BCHW, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(rgb_BCHW, (256, 256), mode='bilinear', align_corners=False)
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
            
        t = t.repeat(latents.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_expand = torch.cat([t] * 2)
            # save input tensors for UNet
            camera = self.get_camera_cond(elevation, azimuth)
            camera = camera.repeat(2,1).to(c_crossattn)
            context = {"context": c_crossattn, "camera": camera, "num_frames": self.cfg.n_view}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_cond)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.cfg.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.cfg.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = self.cfg.recon_std_rescale * latents_recon_adjust + (1-self.cfg.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int( self.num_train_timesteps * min_step_percent )
        self.max_step = int( self.num_train_timesteps * max_step_percent )

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def generate(
        self,
        image_path,  # image tensor [1, 3, H, W] in [0, 1]
        elevation=0,
        azimuth=0,
        camera_distances=0,  # new view params
        c_crossattn=None,
        scale=10,
        ddim_steps=50,
        post_process=False,
        ddim_eta=1,
    ):
        if c_crossattn is None:
            c_crossattn = self.c_crossattn
        
        c_crossattn = c_crossattn.repeat(self.cfg.n_view, 1, 1)
        c_crossattn = torch.cat([torch.zeros_like(c_crossattn).to(self.device), c_crossattn], dim=0)

        camera = get_camera(self.cfg.n_view)
        
        cond = {
            'context': c_crossattn,
            'camera': camera.repeat(2,1).to(c_crossattn),
            'num_frames': self.cfg.n_view,
        }

        imgs = self.gen_from_cond(cond, scale, ddim_steps, post_process, ddim_eta)

        return imgs

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def gen_from_cond(
        self,
        cond,
        scale=10,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1.,
    ):
        # produce latents loop
        B = cond["context"].shape[0] // 2 
        latents = torch.randn((B, 4, 32, 32), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)

        for t in self.scheduler.timesteps:
            t_in = t.reshape(1).repeat(B).to(self.device)

            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t_in] * 2)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)[
                "prev_sample"
            ]

        imgs = self.decode_latents(latents)
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs

        return imgs
