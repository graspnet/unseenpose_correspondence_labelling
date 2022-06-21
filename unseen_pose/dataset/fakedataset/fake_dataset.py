import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

class FakeDataset(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        fake_points = torch.randn(100,3)
        voxel_size = 0.01
        coords = (fake_points / voxel_size).int()
        idxs = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[idxs]
        fake_points = fake_points[idxs]
        return coords, fake_points

    def __len__(self):
        return 100

def get_fake_dataloader():
    dataloader = DataLoader(FakeDataset(), batch_size = 4, shuffle = False,
    num_workers=8, collate_fn = ME.utils.batch_sparse_collate)
    return dataloader


import MinkowskiEngine as ME
from unseen_pose.dataset.fakedataset.fake_dataset import get_fake_dataloader
l = get_fake_dataloader()
for idx, data in enumerate(l):
    pass

sinput = ME.SparseTensor(*data)
# def collate_fn(batch):
#     if type(batch[0]).__module__ == 'numpy':
#         return [torch.from_numpy(b) for b in batch]
#     elif isinstance(batch[0], container_abcs.Sequence):
#         return [[torch.from_numpy(sample) for sample in b] for b in batch]
#     elif isinstance(batch[0], container_abcs.Mapping):
#         ret_dict = {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
#         coords_batch = ret_dict['coords']
#         feats_batch = ret_dict['feats']
#         if 'objectness_label' in ret_dict:
#             labels_batch = ret_dict['objectness_label']
#             coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
#             ret_dict['objectness_label'] = labels_batch
#         else:
#             coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
#         ret_dict['coords'] = coords_batch
#         ret_dict['feats'] = feats_batch
#         return ret_dict
    
#     raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))