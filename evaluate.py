import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import torch
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.use_encoded_image = True # load encoded image for evaluation

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# evaluate
model = create_model(opt)

if opt.verbose:
    print(model)

MSE = 0
PSNR = 0
SSIM = 0
LPIPS = 0

loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_vgg.requires_grad_(False)

for i,data in enumerate(dataset):
    with torch.no_grad():
        generated = model.inference(data['label'], data['inst'], data['image'])
    print('[{}/{}]: process image... {}'.format(i+1, len(dataset), data['path']))

    generated = generated.cpu()
    target = data['image']

    # LPIPS
    LPIPS += loss_fn_vgg(target, generated).item()

    # Scale in [0,1]
    generated = (generated * 0.5 + 0.5).data.numpy()
    target  =  (target * 0.5 + 0.5).data.numpy()

    # MSE
    MSE += np.mean(np.square(target-generated))

    # PSNR
    PSNR += peak_signal_noise_ratio(target, generated, data_range=1.0)

    # SSIM
    SSIM += structural_similarity(target[0], generated[0], data_range=1.0, channel_axis=0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

MSE /= len(dataset)
PSNR /= len(dataset)
SSIM /= len(dataset)
LPIPS /= len(dataset)

print('Evaluating epoch {}'.format(opt.which_epoch))
print('MSE: {}'.format(MSE))
print('PSNR: {}'.format(PSNR))
print('SSIM: {}'.format(SSIM))
print('LPIPS: {}'.format(LPIPS))
