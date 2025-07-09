import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import scipy.io as scio
from tfdiff.params import AttrDict
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from tfdiff.tools.read_bf_file import read_bf_file

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


class WiFiDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.dat', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        csi_list = read_bf_file(cur_filename)  # 解析 .dat 文件

        features = []
        cond = []
        for entry in csi_list:
            csi = entry['csi']  # shape: (n_rx, n_tx, n_subcarriers)
            amp = np.abs(csi).astype(np.float32)  # shape: (3, 3, 30)
            if amp.shape != (3, 3, 30):
                continue  # 跳过格式不对的帧
            amp_vec = amp.reshape(-1)  # 3*3*30 = 270
            if len(amp_vec) < 90:
                continue  # 跳过不足的数据
            amp_vec = amp_vec[:90]  # 只取前90个分量
            features.append(amp_vec)
            cond.append(np.zeros(6, dtype=np.float32))  # 可根据 entry 加入实际标签

        if len(features) == 0:
            raise ValueError(f"No valid CSI frames in file: {cur_filename}")

        data_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
        cond_tensor = torch.tensor(np.stack(cond), dtype=torch.float32)

        return {
            'data': data_tensor,  # shape: (N, 90)
            'cond': cond_tensor   # shape: (N, 6)
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
