### Using RGB images only. Predict with U-Net.
python predict_test_brats.py --name brats_seg_t1_t1ce --gpu_ids 0 --dataroot ./datasets/BraTS_2019/axial/t1_t1ce --netG unet --label_nc 0 --no_segmentation --no_instance --resize_or_crop none --which_epoch latest
