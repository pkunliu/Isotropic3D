
export PYTHONPATH=$PYTHONPATH:./

CUDA_VISIBLE_DEVICES=2 python3 scripts/i2i.py  \
    --prompt "../assets/wolf_rgba.png" \
    --config_path "configs/mvd_infer.yaml" \
    --ckpt_path "../ckpt/isotropic3d.ckpt" \
    --num_frames 4 \
    --camera_elev 15 \
    --save_name "wolf"