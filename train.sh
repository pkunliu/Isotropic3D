
export PYTHONPATH=$PYTHONPATH:./isotropic
python launch.py \
    --config configs/isotropic3d-shading.yaml \
    --train \
    --gpu 0 \
    system.guidance.cond_image_path=assets/wolf_rgba.png \
    tag=wolf