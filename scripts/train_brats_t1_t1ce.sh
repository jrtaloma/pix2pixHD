### Using RGB images only. Train with U-Net.
python train.py --name brats_t1_t1ce --gpu_ids 0,1 --batchSize 32 --dataroot ./datasets/BraTS_2019/axial/t1_t1ce --netG unet --label_nc 0 --class_labels 0 64 128 255 --resize_or_crop none
