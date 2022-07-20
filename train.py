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

def style_loss(output, target):
    loss = F.cross_entropy(output, target)

    return loss

if __name__ == '__main__':
    loader = data_loader.create_data_loader()
    fetcher = data_loader.InputFetcher(loader)

    G = weighted_network.Generator().to(device)
    D = weighted_network.Discriminator().to(device)
    S = weighted_network.StyleEncoder().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-6)
    optimizer_S = optim.Adam(S.parameters(), lr=1e-5)

    start_time = time.time()
    max_epoch = 100000

    for epoch in range(max_epoch):
        inputs = next(fetcher)
        inputs['x_real']['posrot'].requires_grad = True

        style = inputs['y_org']
        num_style = len(style)
        style_vector = np.zeros((num_style, 8))
        for i in range(num_style):
            style_vector[i][style[i]] = 1
        style_vector = torch.Tensor(style_vector).to(device)

        if not len(style) == 8:
            continue

        output_G = G(inputs['x_real']['posrot'], style_vector)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        loss_fake = adv_loss(D(output_G), 0)
        loss_real = adv_loss(D(inputs['x_real']['posrot']), 1)
        loss_reg = r1_reg(D(inputs['x_real']['posrot']), inputs['x_real']['posrot'])
        loss_D = loss_real + loss_fake + loss_reg

        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # -----------------
        #  Train Style Encoder
        # -----------------

        optimizer_S.zero_grad()

        loss_S = style_loss(S(inputs['x_real']['posrot']), style_vector)

        loss_S.backward(retain_graph=True)
        optimizer_S.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        loss_adv = adv_loss(D(output_G), 1)
        loss_sty = style_loss(S(output_G), style_vector)
        loss_G = loss_adv + loss_sty

        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        # Print Loss
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print('-------------------------------------------------------')
            print('cost G =', '{:.6f}'.format(loss_G))
            print('cost D =', '{:.6f}'.format(loss_D))
            print('cost S =', '{:.6f}'.format(loss_S))
            print('-------------------------------------------------------')
            print('loss_fake =', '{:.6f}'.format(loss_fake), 'loss_real =', '{:.6f}'.format(loss_real), 'loss_reg =', '{:.6f}'.format(loss_reg))
            print('loss_adv =', '{:.6f}'.format(loss_adv), 'loss_sty =', '{:.6f}'.format(loss_sty))
            print('Elapsed time: %.3f, Iteration: [%d/%d]' % (elapsed_time, (epoch + 1), max_epoch))
            print('-------------------------------------------------------')

        if (epoch + 1) % 5000 == 0:
            PATH = './model'
            torch.save(G, PATH + "/model_G_" + str(epoch + 1) + ".pt")
            torch.save(D, PATH + "/model_D_" + str(epoch + 1) + ".pt")

        if (epoch + 1) % 100 == 0:
            PATH = './model'
            torch.save(G, PATH + "/model_G_latest.pt")
            torch.save(D, PATH + "/model_D_latest.pt")
