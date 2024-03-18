# Isotropic3D
Official implementation for Isotropic3D: Image-to-3D Generation Based on a Single CLIP Embedding

## | [Project Page](https://isotropic3d.github.io/) | [Paper](https://arxiv.org/abs/2403.10395) | [Weights](https://huggingface.co/pkunliu/Isotropic3D)


https://github.com/pkunliu/Isotropic3D/assets/48075709/412a040e-386e-4520-82b6-0b0a1a217a1c




## Installation

```sh
# torch2.0.1+cu118
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install -r requirements.txt
```


## Quickstart

We use the configuration with soft-shading. It would need an A100 GPU in most cases to compute normal.
```sh
export PYTHONPATH=$PYTHONPATH:./isotropic
python launch.py \
    --config configs/isotropic3d-shading.yaml \
    --train \
    --gpu 0 \
    system.guidance.cond_image_path=assets/wolf_rgba.png
```

### Resume from checkpoints

If you want to resume from a checkpoint, do:

```sh
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
# if the training has completed, you can still continue training for a longer time by setting trainer.max_steps
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt trainer.max_steps=20000
```

If you want to test from a checkpoint, do:
```sh
# you can also perform testing using resumed checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --test --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
# note that the above commands use parsed configuration files from previous trials
# which will continue using the same trial directory
# if you want to save to a new trial directory, replace parsed.yaml with raw.yaml in the command

# only load weights from saved checkpoint but dont resume training (i.e. dont load optimizer state):
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 system.weights=path/to/trial/dir/ckpts/last.ckpt
```


## Acknowledgement
- [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio)
- [threestudio](https://github.com/threestudio-project/threestudio)

Thanks to the maintainers for their contribution to the community!


## Citing

If you find Isotropic3D helpful, please consider citing:

```
@article{liu2024isotropic3d,
  title={Isotropic3D: Image-to-3D Generation Based on a Single CLIP Embedding}, 
  author={Liu, Pengkun and Wang, Yikai and Sun, Fuchun and Li, Jiafang and Xiao, Hang and Xue, Hongxiang and Wang, Xinzhou},
  journal={arXiv preprint arXiv:2403.10395},
  year={2024}
}
```
