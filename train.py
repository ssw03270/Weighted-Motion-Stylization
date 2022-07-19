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

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)

    return reg

if __name__ == '__main__':
    datasets = data_loader.create_data_loader('train')
    train_data, classes = data_loader.create_train_data(datasets)
    input_batch = torch.tensor(train_data, dtype=torch.float32, requires_grad=True, device=device)
    target_batch = torch.tensor(train_data, dtype=torch.float32, device=device)
    print('train data shape:', train_data.shape)
    print('device type:', device)

    G = weighted_network.Generator().to(device)
    D = weighted_network.Discriminator().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-6)

    start_time = time.time()
    max_epoch = 10000

    for epoch in range(max_epoch):
        running_loss_G = 0
        running_loss_D = 0
        step = 0
        train_size = input_batch.shape[0]
        batch_size = 8
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = input_batch[batch_mask]
        total_step = len(x_batch)


        for input, cls in zip(x_batch, classes):
            step += 1
            style_vector = np.zeros(8)
            style_vector[cls[1]] = 1
            style_vector = torch.tensor(style_vector, dtype=torch.float32, device=device)

            output_G = G(input, style_vector).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            loss_fake = adv_loss(D(output_G), 0)
            loss_real = adv_loss(D(input), 1)
            loss_reg = r1_reg(D(input), input)
            loss_D = loss_real + loss_fake + loss_reg

            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            running_loss_D += loss_D.item() * input.shape[0]

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            loss_G = adv_loss(D(output_G), 1)

            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            running_loss_G += loss_G.item() * input.shape[0]

            if step % total_step == 0:
                elapsed_time = time.time() - start_time
                print('cost G =', '{:.6f}'.format(running_loss_G / total_step))
                print('cost D =', '{:.6f}'.format(running_loss_D / total_step))
                print('Elapsed time: %.3f, Iteration: [%d/%d]' % (elapsed_time, (epoch + 1), max_epoch))



        if (epoch + 1) % 1000 == 0:
            PATH = './model'
            torch.save(G, PATH + "/model_G_" + str(epoch + 1) + ".pt")
            torch.save(D, PATH + "/model_D_" + str(epoch + 1) + ".pt")

        if (epoch + 1) % 100 == 0:
            PATH = './model'
            torch.save(G, PATH + "/model_G_latest.pt")
            torch.save(D, PATH + "/model_D_latest.pt")
