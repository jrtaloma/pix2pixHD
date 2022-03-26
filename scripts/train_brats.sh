### Using RGB images only. Train with U-Net.
python train.py --name brats_t1_t1ce --gpu_ids 0,1 --batchSize 4 --dataroot ./datasets/BraTS_2019/axial/t1_t1ce --netG unet --label_nc 0 --no_instance --resize_or_crop none
