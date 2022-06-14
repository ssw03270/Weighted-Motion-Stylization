import os
import numpy as np
import torch

import data.data_loader as data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

output_dir = 'output/'
src_file = "output/walking_neutral.bvh"

PATH = './model/model_latest.pt'

if __name__ == '__main__':
    input, traj, feet = data_loader.create_test_data(src_file)
    input_batch = torch.tensor(input, dtype=torch.float32, requires_grad=True, device=device)

    for i in range(8):
        style_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        style_vector[i] = 1
        style_vector = torch.tensor(style_vector, dtype=torch.float32, device=device)
        model = torch.load(PATH)
        model.eval()
        predict = model(input_batch, style_vector)

        output_file = os.path.join(output_dir, 'output_' + str(i) + '.bvh')
        data_loader.create_output_data(predict, traj, feet[0].cpu().numpy(), output_file, has_traj=True, has_feet=True)