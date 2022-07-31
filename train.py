import time
import sys
import matplotlib.pyplot as plt
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

def style_encoder_loss(output, target):
    # loss = F.binary_cross_entropy_with_logits(output, target)
    loss = torch.mean(torch.abs(output - target))
    return loss

def content_loss(output, target):
    loss = torch.mean((output - target).norm(dim=3))

    return loss

def style_loss(output, target):
    loss = torch.mean(torch.abs(output - target))

    return loss

def cyc_loss(output, target):
    loss = torch.mean((output - target).norm(dim=3))
    return loss

if __name__ == '__main__':

    loader = data_loader.create_data_loader()
    loader_ref = data_loader.create_data_loader()
    fetcher = data_loader.InputFetcher(loader, loader_ref)

    G = weighted_network.Generator().to(device)
    D = weighted_network.Discriminator().to(device)
    S = weighted_network.StyleEncoder().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.99, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=1e-6,  betas=(0.99, 0.999))
    optimizer_S = optim.Adam(S.parameters(), lr=1e-5, betas=(0.99, 0.999))

    start_time = time.time()
    max_epoch = 500000

    losses_D = []
    losses_S = []
    losses_G = []

    d_time = True

    for epoch in range(max_epoch):
        inputs = next(fetcher)

        # create original style vector
        original_style = inputs['y_org']
        num_style = len(original_style)
        original_style_vector = np.zeros((num_style, 8))
        for i in range(num_style):
            original_style_vector[i][original_style[i]] = 1
        original_style_vector = torch.Tensor(original_style_vector).to(device)

        # create target style vector
        target_style = inputs['y_trg']
        num_style = len(target_style)
        target_style_vector = np.zeros((num_style, 8))
        for i in range(num_style):
            target_style_vector[i][target_style[i]] = 1
        target_style_vector = torch.Tensor(target_style_vector).to(device)

        x_real = inputs['x_real']['posrot']
        x_ref = inputs['x_ref']['posrot']

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        x_real.requires_grad_()
        out = D(x_real, original_style)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x_real)

        with torch.no_grad():
            s_ref = S(x_ref)
            x_fake1 = G(x_real, s_ref)

        out = D(x_fake1, target_style)
        loss_fake = adv_loss(out, 0)

        loss_D = loss_real + loss_fake + loss_reg

        loss_D.backward()
        optimizer_D.step()

        # -----------------
        #  Train Style Encoder
        # -----------------
        optimizer_S.zero_grad()

        loss_S = style_encoder_loss(S(x_real), original_style_vector)

        loss_S.backward()
        optimizer_S.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        s_ref2 = S(x_ref)
        x_fake2 = G(x_real, s_ref2)

        out2 = D(x_fake2, target_style)
        loss_adv = adv_loss(out2, 1)

        s_pred = S(x_fake2)
        loss_sty = style_loss(s_pred, target_style_vector)

        s_real = S(x_real)
        x_rec = G(x_real, s_real)
        loss_content = content_loss(x_rec, x_real)

        x_cyc = G(x_fake2, original_style_vector)
        loss_cyc = cyc_loss(x_cyc, x_real)

        loss_G = loss_adv + loss_sty + loss_content + loss_cyc

        loss_G.backward()
        optimizer_G.step()

        # Print Loss
        if (epoch + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print('=======================================================')
            print('cost G =', '{:.6f}'.format(loss_G))
            print('cost D =', '{:.6f}'.format(loss_D))
            print('cost S =', '{:.6f}'.format(loss_S))
            print('-------------------------------------------------------')
            print('loss_fake =', '{:.6f}'.format(loss_fake), 'loss_real =', '{:.6f}'.format(loss_real), 'loss_reg =', '{:.6f}'.format(loss_reg))
            print('loss_adv =', '{:.6f}'.format(loss_adv), 'loss_sty =', '{:.6f}'.format(loss_sty), 'loss_content =', '{:.6f}'.format(loss_content), 'loss_cyc =', '{:.6f}'.format(loss_cyc))
            print('Elapsed time: %.3f, Iteration: [%d/%d]' % (elapsed_time, (epoch + 1), max_epoch))

            # torch.set_printoptions(sci_mode=False)
            # print(s_pred)

            losses_D.append(loss_D.item())
            losses_S.append(loss_S.item())
            losses_G.append(loss_G.item())

            d_time = not d_time

        if (epoch + 1) % 10000 == 0:
            PATH = './model'
            torch.save(G, PATH + "/model_G_" + str(epoch + 1) + ".pt")

        if (epoch + 1) % 100 == 0:
            PATH = './model'
            torch.save(G, PATH + "/model_G_latest.pt")

    plt.plot(np.array(range(1, len(losses_G) + 1)) * 100, losses_G, label='losses_G')
    plt.plot(np.array(range(1, len(losses_D) + 1)) * 100, losses_D, label='losses_D')
    plt.plot(np.array(range(1, len(losses_S) + 1)) * 100, losses_S, label='losses_S')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('losses')
    plt.legend(loc='upper right')
    plt.show()