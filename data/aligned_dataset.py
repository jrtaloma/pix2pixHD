import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.labels = []

        ### input A (input real images)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (target real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### segmentation maps
        if not opt.no_segmentation:
            self.dir_seg = os.path.join(opt.dataroot, opt.phase + '_seg')
            self.seg_paths = sorted(make_dataset(self.dir_seg))
            labels = []
            for seg_path in self.seg_paths:
                seg = Image.open(seg_path).convert('L')
                seg = np.array(seg).astype(float)
                labels.append(1) if seg.max() > 0 else labels.append(0)
            self.labels = labels

        self.dataset_size = len(self.A_paths) 

    def __getitem__(self, index):        
        ### input A (input real images)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))

        B_tensor = seg_tensor = 0
        ### input B (target real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)                         

        input_dict = {'input': A_tensor, 'target': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
