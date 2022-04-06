### Using RGB images only. Train with U-Net.
python train.py --name brainmetshare_t1pre_t2flair --gpu_ids 2,3 --batchSize 32 --dataroot ./datasets/stanford_release_brainmask/axial/t1pre_t2flair --netG unet --label_nc 0 --resize_or_crop none
