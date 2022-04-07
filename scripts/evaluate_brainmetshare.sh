### Using RGB images only. Test with U-Net.
python evaluate.py --name brainmetshare_t1pre_t1post --gpu_ids 0 --dataroot ./datasets/stanford_release_brainmask/axial/t1pre_t1post --netG unet --label_nc 0 --no_segmentation --resize_or_crop none --which_epoch latest
