import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import torch
import numpy as np
from PIL import Image

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.use_encoded_image = True # load encoded image for evaluation

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# predict
model = create_model(opt)

if opt.verbose:
    print(model)

assert opt.name in ['brats_t1_t1ce', 'brats_t1_t2'], '{opt.name} is not supported'
source_modality = 't1'
target_modality = 't1ce' if opt.name == 'brats_t1_t1ce' else 't2'

save_img_path = os.path.join('./pred', opt.name, opt.phase)
if not os.path.isdir(save_img_path):
    os.makedirs(save_img_path)


for i,data in enumerate(dataset):
    with torch.no_grad():
        generated = model.inference(data['input'], data['target'])
    print('[{}/{}]: process image... {}'.format(i+1, len(dataset), data['path']))

    source = data['input'].cpu().data.numpy()
    generated = generated.cpu().data.numpy()

    # [N, H, W, C]
    source = np.transpose(source, (0,2,3,1))
    generated = np.transpose(generated, (0,2,3,1))

    # scale from [-1,1] to [0,255]
    source = ((source * 0.5 + 0.5) * 255.0).astype(np.uint8)
    generated = ((generated * 0.5 + 0.5) * 255.0).astype(np.uint8)

    # save source image
    image = Image.fromarray(source[0], 'RGB')
    path = data['path'][0].split('/')[-3]
    path = os.path.join(save_img_path, path, source_modality)
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = data['path'][0].split('/')[-1]
    image.save(os.path.join(path, filename))

    # save target image
    image = Image.fromarray(generated[0], 'RGB')
    path = data['path'][0].split('/')[-3]
    path = os.path.join(save_img_path, path, target_modality)
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = data['path'][0].split('/')[-1]
    image.save(os.path.join(path, filename))
