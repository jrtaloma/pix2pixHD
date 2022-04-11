### Using RGB images only. Train with U-Net.
python train.py --name brats_seg_t1_t2 --gpu_ids 2,3 --batchSize 32 --dataroot ./datasets/BraTS_2019/axial/t1_t2 --netG unet --label_nc 0 --class_labels 0 64 128 255 --no_instance --resize_or_crop none --unet_model_weights ./models/unet/BraTS_T2_weighted/checkpoint_epoch18.pth --unet_input_channels 3 --unet_n_classes 4
