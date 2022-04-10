### Using RGB images only. Test with U-Net.
python evaluate.py --name brainmetshare_seg_t1pre_t1post --gpu_ids 0 --dataroot ./datasets/stanford_release_brainmask/axial/t1pre_t1post --netG unet --label_nc 0 --class_labels 0 255 --resize_or_crop none --which_epoch latest
