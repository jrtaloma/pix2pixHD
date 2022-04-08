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

save_img_path = './out'
if not os.path.isdir(save_img_path):
    os.mkdir(save_img_path)

max_images = 100
# fix random seed
rng = np.random.RandomState(0)
indexes = rng.randint(0, len(dataset), (max_images))

for i,data in enumerate(dataset):
    if not i in indexes:
        continue
    with torch.no_grad():
        generated = model.inference(data['input'], data['target'])
    print('[{}/{}]: process image... {}'.format(i+1, len(dataset), data['path']))

    source = data['input'].cpu().data.numpy()
    generated = generated.cpu().data.numpy()
    target = data['target'].cpu().data.numpy()
    seg = data['seg'].cpu().data.numpy()

    # [N, H, W, C]
    source = np.transpose(source, (0,2,3,1))
    generated = np.transpose(generated, (0,2,3,1))
    target = np.transpose(target, (0,2,3,1))
    seg = np.transpose(seg, (0,2,3,1))

    # scale from [-1,1] to [0,255]
    source = ((source * 0.5 + 0.5) * 255.0).astype(np.uint8)
    generated = ((generated * 0.5 + 0.5) * 255.0).astype(np.uint8)
    target  =  ((target * 0.5 + 0.5) * 255.0).astype(np.uint8)
    if not opt.no_segmentation:
        seg  =  ((seg * 0.5 + 0.5) * 255.0).astype(np.uint8)

    # concatenate along width
    images = np.concatenate((source[0], generated[0], target[0], seg[0]), 1)
    images = Image.fromarray(images, 'RGB')
    filename = data['path'][0].split('/')[-1]
    images.save(os.path.join(save_img_path, filename))
