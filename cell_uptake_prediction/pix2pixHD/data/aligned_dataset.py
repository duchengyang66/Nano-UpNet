"""
Aligned dataset for cell uptake translation.

Each sample is:
  train_A: bright-field image (3 channels)
  train_label: YOLO segmentation label, 3 channels:
      channel 1: instance ID (0=bg, 1+=each cell)
      channel 2: class ID (0=bg, 1..20=cell type, 1-based)
      channel 3: 0
  train_B: red fluorescence image (3 channels)

G input = concat(train_A, train_label) = 6 channels
"""

import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from data.cell_types import CELL_TYPE_TO_IDX
from PIL import Image


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # bright field
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.A_paths = sorted(make_dataset(self.dir_A))
        if len(self.A_paths) == 0:
            raise FileNotFoundError('Empty A dir: %s' % self.dir_A)

        # YOLO segmentation label
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')
        self.label_paths = sorted(make_dataset(self.dir_label))
        if len(self.label_paths) != len(self.A_paths):
            raise ValueError(
                'A count (%d) != label count (%d)' % (len(self.A_paths), len(self.label_paths))
            )

        # fluorescence
        # Always try to load B (covers train + test + val); missing B for no_uptake is OK
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        if os.path.isdir(self.dir_B):
            self.B_paths_dict = {os.path.basename(p): p for p in sorted(make_dataset(self.dir_B))}
        else:
            self.B_paths_dict = {}

        # instance maps (optional, not used by us)
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        # precomputed features (optional)
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        # sanity: all A filenames parse to a known cell type
        for p in self.A_paths:
            fname = os.path.basename(p)
            ct = fname.split('no_uptake')[0].rstrip('_') if 'no_uptake' in fname else fname.split('_')[0]
            if ct not in CELL_TYPE_TO_IDX:
                raise ValueError('Unknown cell type %s in %s' % (ct, fname))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        # bright field
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A)  # (3, H, W)

        # YOLO label (3-channel: instance, class_id, 0)
        label_path = self.label_paths[index]
        label_img = Image.open(label_path).convert('RGB')
        # Safety: if label size != A size, resize to match (handles edge cases like 512x512)
        if label_img.size != A.size:
            label_img = label_img.resize(A.size, Image.NEAREST)
        label_tensor = transform_A(label_img)  # (3, H, W)

        # concat: G input = 6 channels
        net_input = torch.cat([A_tensor, label_tensor], dim=0)  # (6, H, W)

        # fluorescence
        B_tensor = inst_tensor = feat_tensor = 0
        # Try to load B by filename (handles missing B for no_uptake samples)
        a_basename = os.path.basename(A_path)
        if a_basename in self.B_paths_dict:
            B_path = self.B_paths_dict[a_basename]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                feat_tensor = normalize()(transform_A(feat))

        input_dict = {
            'label': net_input,   # 6 channels: bright field + YOLO label
            'inst': inst_tensor,
            'image': B_tensor,
            'feat': feat_tensor,
            'path': A_path,
        }
        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
