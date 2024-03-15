export PYTHONPATH=$PYTHONPATH:./isotropic

CUDA_VISIBLE_DEVICES=2 python 2dgen.py --image_path assets/mario_rgba.png --save_path mario.png
