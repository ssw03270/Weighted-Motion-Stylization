import os
import numpy as np
import data.export_dataset as export_dataset
import data.helper as helper
import torch

from animation import BVH
from animation.InverseKinematics import JacobianInverseKinematics
from animation.Quaternions import Quaternions

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_data_loader(type='train'):
    print('create source dataset %s phase...' % type)

    if type == 'train':
        data = np.load('./datasets/styletransfer_generate.npz')
        return data

def create_train_data(dataset):
    for datatype in dataset:
        if datatype == 'clips':
            print('clips dataset shape:', dataset[datatype].shape)
        if datatype == 'feet':
            print('feet dataset shape:', dataset[datatype].shape)
        if datatype == 'classes':
            print('classes dataset shape:', dataset[datatype].shape)
    pre_data = np.load('./datasets/preprocess_styletransfer_generate.npz')
    data = helper.normalize(dataset['clips'], pre_data['Xmean'], pre_data['Xstd'])
    cls = dataset['classes']
    return data, cls

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
    return data, input['x_real']['traj'], input['x_real']['feet']

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
    helper.to_bvh_cmu(positions, filename=filename, frametime=1.0/30.0)

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