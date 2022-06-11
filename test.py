import os
import numpy as np
import torch

import data.data_loader as data_loader

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

output_dir = 'output/'
src_file = "output/walking_neutral.bvh"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    input, traj, feet = data_loader.create_test_data(src_file)
    input_batch = torch.tensor(input, dtype=torch.float32, requires_grad=True, device=device)

    PATH = './model/model_state_dict.pt'
    model = torch.load(PATH)
    model.eval()
    predict = model(input_batch)

    output_file = os.path.join(output_dir, 'output.bvh')
    data_loader.create_output_data(predict, traj, feet[0].cpu().numpy(), output_file, has_traj=True, has_feet=True)