import os
import numpy as np
import torch

import data.data_loader as data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

output_dir = 'output'
src_file = "output/reference/walking_neutral.bvh"

PATH = './model/model_G_100000.pt'

if __name__ == '__main__':
    input, traj, feet = data_loader.create_test_data(src_file)
    input_batch = input['posrot']

    for i in range(3):
        style_vector = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
        style_vector[0][i] = 1
        style_vector = np.repeat(style_vector, repeats=8, axis=0)
        style_vector = torch.Tensor(style_vector).to(device)
        model = torch.load(PATH)
        model.eval()
        predict = model(input_batch, style_vector)

        output_file = os.path.join(output_dir, 'output_' + str(i) + '.bvh')
        data_loader.create_output_data(predict, traj, feet[0].cpu().numpy(), output_file, has_traj=True, has_feet=True)