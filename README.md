# NeRFBlendShape Unofficial Implementation

This is an ***UNOFFICIAL*** implementation of the paper "Reconstructing Personalized Semantic Facial NeRF Models From Monocular Video". The authors released the inference code [here](https://github.com/USTC3DV/NeRFBlendShape-code), while we implements the training part according to the paper. Besides, we train a ***torso net*** based on [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF). The performance of this implementation may differ from the original paper. An example test result can be found from data/example.mp4


## Usage notice

This implementation relies on a private blendshape extractor which can not be open-source, so you could try using the FaceWarehouse from the original paper or using other blendshape basis like BFM2009. 


## Data preparation
0. Prepare some model weights according to [here](https://github.com/ashawkey/RAD-NeRF) (Data pre-processing/preparation part).

1. Put the video under {data/vids} and run `python data_utils/process.py VIDEO_NAME`

2. We extract blendshapes and saving as files to `data/CORRESPONDING_DATASET_NAME/`, including `expr.npy`, `expr_max.npy`, `expr_min.npy`. The first one contains an array whose shape is [image_num_from_dataset, expression_blendshape_dim]. And the latter two contrain the maximum and the minimum expression caculated from the the first one. You could extract blendshapes using your extractor and saving as the similar format.


## Training

```
# train head part
python main_nerf.py data/DATASET_NAME --workspace EXPERIMENT_FILE_PATH --test_fps 25 --network blend4_noamb --use_patch_loss --patch_size 32 --train_epoch 15 --downscale 1 --cuda_ray --preload 2 --test_num 500 --expr_dim {expression_blendshape_dim} --update_extra_interval 1000 --fp16

# train torso part
python main_nerf.py data/DATASET_NAME --workspace EXPERIMENT_FILE_PATH --test_fps 25 --network blend4_noamb --use_patch_loss --patch_size 32 --train_epoch 15 --downscale 1 --cuda_ray --preload 2 --test_num 500 --expr_dim {expression_blendshape_dim}  --update_extra_interval 1000 --fp16 --torso --head_ckpt OUTPUT_PTH_FILE_PATH_FROM_FIRST_STEP

```

## Testing

```
python main_nerf.py data/DATASET_NAME --workspace EXPERIMENT_FILE_PATH --test_fps 25 --network blend4_noamb --use_patch_loss --patch_size 32 --train_epoch 15 --downscale 1 --cuda_ray --preload 2 --test_num 500 --expr_dim {expression_blendshape_dim} --update_extra_interval 1000 --fp16 --torso --head_ckpt OUTPUT_PTH_FILE_PATH_FROM_FIRST_STEP --test
```

## Acknowledgement

- [NeRFBlendshape](https://ustc3dv.github.io/NeRFBlendShape/)
- [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF)
- [torch-ngp](https://github.com/ashawkey/torch-ngp)

## Else

Give this repo a star if it helps you!
