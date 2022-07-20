import os
import numpy as np
import data.export_dataset as export_dataset
import data.helper as helper
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler

from animation import BVH
from animation.InverseKinematics import JacobianInverseKinematics
from animation.Quaternions import Quaternions

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(data.Dataset):
    def __init__(self):
        self.dataset = np.load('./datasets/styletransfer_generate.npz')
        self.preprocess = np.load('./datasets/preprocess_styletransfer_generate.npz')
        self.data_clips = self.dataset['clips']
        self.data_feet = self.dataset['feet']
        self.data_classes = self.dataset['classes']
        self.samples, self.contacts, self.targets, self.labels = self.make_dataset()

    def make_dataset(self):
        X, F, Y, C = [], [], [], []
        for dom in range(len(styles)):
            dom_idx = [si for si in range(len(self.data_classes))
                       if self.data_classes[si][1] == styles.index(styles[dom])]  # index list that belongs to the domain
            dom_clips = [self.data_clips[cli] for cli in dom_idx]  # clips list (motion data) that belongs to the domain
            dom_feet = [self.data_feet[fti] for fti in dom_idx]
            dom_contents = [self.data_classes[ci][0] for ci in dom_idx]
            X += dom_clips
            F += dom_feet
            Y += [dom] * len(dom_clips)
            C += dom_contents
        return X, F, Y, C

    def __getitem__(self, index):
        x = self.samples[index]
        f = self.contacts[index]
        x = helper.normalize(x, self.preprocess['Xmean'], self.preprocess['Xstd'])
        data = {'posrot': x[:7], 'traj': x[-4:], 'feet': f}
        y = self.targets[index]
        c = self.labels[index]
        return {'x': data, 'y': y, 'c': c}

    def __len__(self):
        return len(self.targets)


class InputFetcher:
    def __init__(self, loader, loader_ref=None):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = 16
        self.device = device

    def fetch_src(self):
        try:
            src = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            src = next(self.iter)
        return src

    def fetch_refs(self):
        try:
            ref = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            ref = next(self.iter_ref)
        return ref

    def __next__(self):
        inputs = {}
        src = self.fetch_src()
        inputs_src = {'x_real': src['x'], 'y_org': src['y'], 'c_real': src['c']}
        inputs.update(inputs_src)

        if self.loader_ref is not None:
            ref = self.fetch_refs()
            z = torch.randn(src['y'].size(0), self.latent_dim)   # random Gaussian noise for x_ref
            z2 = torch.randn(src['y'].size(0), self.latent_dim)  # random Gaussian noise for x_ref2
            inputs_ref = {'x_ref': ref['x'], 'x_ref2': ref['x2'],
                          'c_ref': ref['c'], 'c_ref2': ref['c2'],
                          'y_trg': ref['y'],
                          'z_trg': z, 'z_trg2': z2}
            inputs.update(inputs_ref)
        return to(inputs, self.device)

def create_test_data(filename):
    dataframe, feet = export_dataset.preprocess(filename, slice=False, downsample=1)
    data = np.transpose(dataframe, (2, 0, 1))
    pre_data = np.load('./datasets/preprocess_styletransfer_generate.npz')
    data = helper.normalize(data, pre_data['Xmean'], pre_data['Xstd'])

    x = torch.tensor(data, dtype=torch.float)
    f = torch.tensor(feet, dtype=torch.float)
    x_data = {'posrot': x[:7], 'traj': x[-4:], 'feet': f}

    src = {'x': x_data}
    input = {'x_real': src['x']}
    input = to(input, device, expand_dim=True)
    return x_data, input['x_real']['traj'], input['x_real']['feet']

def create_output_data(output, traj, feet, filename, has_traj=False, has_feet=False):
    pre_data = np.load('./datasets/preprocess_styletransfer_generate.npz')

    output = output.detach().cpu().numpy().copy()
    output = output[0]
    output = helper.denormalize(output, pre_data['Xmean'][:7], pre_data['Xstd'][:7])
    output = np.transpose(output, (1, 2, 0))

    traj = traj.detach().cpu().numpy().copy()
    traj = traj[0]
    traj = helper.denormalize(traj, pre_data['Xmean'][-4:], pre_data['Xstd'][-4:])
    traj = np.transpose(traj, (1, 2, 0))

    # original output
    positions = output[:, :, :3]

    if has_traj:
        positions = helper.restore_animation(positions, traj)

    if has_feet:
        filename = filename[:-4] + '_fs.bvh'
        positions = helper.remove_fs(positions, feet)

    print('Saving animation of %s in bvh...' % filename)
    helper.to_bvh_cmu(positions, filename=filename, frametime=1.0/60.0)

def to(inputs, device, expand_dim=False):
    for name, ele in inputs.items():
        if isinstance(ele, dict):
            for k, v in ele.items():
                if expand_dim:
                    v = torch.unsqueeze(v.clone().detach(), dim=0)
                ele[k] = v.to(device, dtype=torch.float)
        else:
            if expand_dim:
                ele = torch.unsqueeze(torch.tensor(ele), dim=0)
            if name.startswith('y_') or name.startswith('c_'):
                inputs[name] = ele.to(device, dtype=torch.long)
            else:
                inputs[name] = ele.to(device, dtype=torch.float)
    return inputs

def make_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def create_data_loader(which='source'):
    print('Preparing %s dataset during %s phase...' % (which, type))
    if which == 'source':
        dataset = Dataset()
    else:
        raise NotImplementedError

    if type == 'train':
        sampler = make_weighted_sampler(dataset.targets)
        return data.DataLoader(dataset=dataset,
                            batch_size=8,
                            sampler=sampler,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)
    elif type is not None:
        return data.DataLoader(dataset=dataset, batch_size=8)
    else:
        raise NotImplementedError('Please specify dataset type!')