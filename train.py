import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data.data_loader as data_loader
import data.sampling as sampling
import network.weighted_network as weighted_network


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    datasets = data_loader.create_data_loader('train')
    train_data, classes = data_loader.create_train_data(datasets)
    input_batch = torch.tensor(train_data, dtype=torch.float32, requires_grad=True, device=device)
    target_batch = torch.tensor(train_data, dtype=torch.float32, device=device)
    print('train data shape:', train_data.shape)
    print('device type:', device)

    model = weighted_network.WeightedNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()
    max_epoch = 10
    for epoch in range(max_epoch):
        running_loss = 0
        step = 0
        total_step = len(input_batch)
        for input, cls in zip(input_batch, classes):
            step += 1
            style_vector = np.zeros(8)
            style_vector[cls[1]] = 1
            style_vector = torch.tensor(style_vector, dtype=torch.float32, device=device)
            output = model(input, style_vector).to(device)
            loss = criterion(output, input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input.shape[0]

            if step % total_step == 0:
                elapsed_time = time.time() - start_time
                print('cost =', '{:.6f}'.format(running_loss / total_step))
                print('Elapsed time: %.3f, Iteration: [%d/%d]' % (elapsed_time, (epoch + 1), max_epoch))

PATH = './model'
torch.save(model.state_dict(), PATH + "/model.pt")
torch.save(model, PATH + "/model_state_dict.pt")