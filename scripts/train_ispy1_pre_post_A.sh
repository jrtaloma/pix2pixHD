### Using RGB images only. Train with U-Net.
python train.py --name ispy1_pre_post_A --gpu_ids 0,1 --batchSize 32 --dataroot ./datasets/ISPY1/sagittal/pre_post_A --netG unet --label_nc 0 --no_instance --no_segmentation --no_segmentation_loss --resize_or_crop scale_width --loadSize 256 --unet_input_channels 3 --unet_n_classes 2
