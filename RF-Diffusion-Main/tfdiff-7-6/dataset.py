import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from tfdiff.params import AttrDict
from glob import glob
from torch.utils.data.distributed import DistributedSampler


# data_key='csi_data',
# gesture_key='gesture',
# location_key='location',
# orient_key='orient',
# room_key='room',
# rx_key='rx',
# user_key='user',


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


import os
import torch
import scipy.io as scio
from glob import glob
from torch.utils.data import Dataset

class WiFiDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.samples = []  # list of (file, start_idx)

        # 遍历所有 .mat 文件并组合连续的512帧
        for path in paths:
            filenames = glob(f'{path}/**/user*.mat', recursive=True)
            for fname in filenames:
                data = scio.loadmat(fname, verify_compressed_data_integrity=False)
                feature = data['feature']  # shape: [N, 90]
                N = feature.shape[0]
                
                # 计算可以提取多少个完整的512帧样本
                num_samples = N // 512
                
                # 记录每个512帧样本的起始位置
                for i in range(num_samples):
                    start_idx = i * 512
                    self.samples.append((fname, start_idx))
                
                print(f"从文件 {fname} 中提取了 {num_samples} 个512帧样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, start_idx = self.samples[idx]
        sample = scio.loadmat(fname, verify_compressed_data_integrity=False)
        
        # 提取连续512帧作为一个样本
        features = sample['feature'][start_idx:start_idx+512]  # shape: [512, 90]
        
        # 只使用第一帧的条件作为整个序列的条件
        cond = sample['cond'][start_idx]  # shape: [6]
        
        # 转换为复数张量
        feature_tensor = torch.from_numpy(features).to(torch.complex64)  # shape: [512, 90]
        cond_tensor = torch.from_numpy(cond).to(torch.complex64)  # shape: [6]
        
        return {
            'data': feature_tensor,  # [512, 90]
            'cond': cond_tensor      # [6]
        }



class FMCWDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename)
        cur_data = torch.from_numpy(cur_sample['feature']).to(torch.complex64)
        cur_cond = torch.from_numpy(cur_sample['cond'].astype(np.int16)).to(torch.complex64)
        return {
            'data': cur_data,
            'cond': cur_cond.squeeze(0)
        }


class MIMODataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dataset = scio.loadmat(self.filenames[idx])
        data = torch.from_numpy(dataset['down_link']).to(torch.complex64)
        cond = torch.from_numpy(dataset['up_link']).to(torch.complex64)
        return {
            'data': torch.view_as_real(data),
            'cond': torch.view_as_real(cond)
        }


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        paths = paths[0]
        self.filenames = []
        self.filenames += glob(f'{paths}/*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dataset = scio.loadmat(self.filenames[idx])
        data = torch.from_numpy(dataset['clean']).to(torch.complex64)
        cond = torch.from_numpy(dataset['disturb']).to(torch.complex64)
        return {
            'data': data,
            'cond': cond
        }


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        sample_rate = self.params.sample_rate
        task_id = self.params.task_id
        ## WiFi Case
        if task_id == 0:
            for record in minibatch:
                # Filter out records that aren't long enough.
                if len(record['data']) < sample_rate:
                    del record['data']
                    del record['cond']
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }
        ## FMCW Case
        elif task_id == 1:
            for record in minibatch:
                # Filter out records that aren't long enough.
                if len(record['data']) < sample_rate:
                    del record['data']
                    del record['cond']
                    continue
                data = torch.view_as_real(record['data']).permute(1, 2, 0)
                down_sample = F.interpolate(data, sample_rate, mode='nearest-exact')
                norm_data = (down_sample - down_sample.mean()) / down_sample.std()
                record['data'] = norm_data.permute(2, 0, 1)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': torch.view_as_real(cond),
            }

        ## MIMO Case
        elif task_id == 2:
            for record in minibatch:
                data = record['data']
                cond = record['cond']
                # print(f'data.shape:{data.shape}')
                norm_data = (data) / cond.std()
                norm_cond = (cond) / cond.std()
                record['data'] = norm_data.reshape(14, 96, 26, 2).transpose(1, 2)
                record['cond'] = norm_cond.reshape(14, 96, 26, 2).transpose(1, 2)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': data,
                'cond': cond,
            }

            ## EEG Case
        if task_id == 3:
            for record in minibatch:
                data = record['data']
                cond = record['cond']

                norm_data = data / cond.std()
                norm_cond = cond / cond.std()

                record['data'] = norm_data.reshape(512, 1, 1)
                record['cond'] = norm_cond.reshape(512)
            data = torch.stack([record['data'] for record in minibatch if 'data' in record])
            cond = torch.stack([record['cond'] for record in minibatch if 'cond' in record])
            return {
                'data': torch.view_as_real(data),
                'cond': torch.view_as_real(cond),
            }

        else:
            raise ValueError("Unexpected task_id.")


def from_path(params, is_distributed=False):
    data_dir = params.data_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(data_dir)
    elif task_id == 1:
        dataset = FMCWDataset(data_dir)
    elif task_id == 2:
        dataset = MIMODataset(data_dir)
    elif task_id == 3:
        dataset = EEGDataset(data_dir)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=8,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True)


def from_path_inference(params):
    cond_dir = params.cond_dir
    task_id = params.task_id
    if task_id == 0:
        dataset = WiFiDataset(cond_dir)
    elif task_id == 1:
        dataset = FMCWDataset(cond_dir)
    elif task_id == 2:
        dataset = MIMODataset(cond_dir)
    elif task_id == 3:
        dataset = EEGDataset(cond_dir)
    else:
        raise ValueError("Unexpected task_id.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        collate_fn=Collator(params).collate,
        shuffle=False,
        num_workers=os.cpu_count()
    )
