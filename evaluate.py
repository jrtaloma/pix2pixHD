import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import torch
import numpy as np
from pytorch_msssim import ssim
from tqdm import tqdm
import torchmetrics


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

psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to('cuda')
lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity().to('cuda')


for i,data in tqdm(enumerate(dataset), total=len(dataset)):
    with torch.no_grad():
        generated = model.inference(data['input'], data['seg'], data['target'])
    # print('[{}/{}]: process image... {}'.format(i+1, len(dataset), data['path']))

    target = data['target'].to('cuda')

    # LPIPS
    LPIPS += lpips_metric(target, generated).item()

    # Scale in [0,1]
    generated = generated * 0.5 + 0.5
    target = target * 0.5 + 0.5

    # MSE
    MSE += torch.mean(torch.square(target-generated))

    # PSNR
    PSNR += psnr_metric(target, generated)

    # SSIM
    SSIM += ssim(target, generated, data_range=1.0, size_average=True)

MSE /= len(dataset)
PSNR /= len(dataset)
SSIM /= len(dataset)
LPIPS /= len(dataset)

print('Evaluating epoch {}'.format(opt.which_epoch))
print('MSE: {}'.format(MSE))
print('PSNR: {}'.format(PSNR))
print('SSIM: {}'.format(SSIM))
print('LPIPS: {}'.format(LPIPS))
