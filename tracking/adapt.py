"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

from options import opts


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if volatile:
        return Variable(tensor)
    else:
        with torch.no_grad():
            return Variable(tensor)


def train_tgt(seq_name, src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=opts['c_learning_rate'],
                               betas=(opts['beta1'], opts['beta2']))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=opts['d_learning_rate'],
                                  betas=(opts['beta1'], opts['beta2']))
    len_data_loader = opts['source_batch']

    ####################
    # 2. train network #
    ####################
    ###########################
    # 2.1 train discriminator #
    ###########################
    # make images variable
    images_src = make_variable(src_data_loader)
    images_tgt = make_variable(tgt_data_loader)

    # zero gradients for optimizer
    optimizer_critic.zero_grad()

    # extract and concat features
    feat_src = src_encoder(images_src, out_layer='conv3')
    feat_tgt = tgt_encoder(images_tgt, out_layer='conv3')
    feat_concat = torch.cat((feat_src, feat_tgt), 0)

    # predict on discriminator
    pred_concat = critic(feat_concat.detach())

    # prepare real and fake label
    label_src = make_variable(torch.ones(feat_src.size(0)).long())
    label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
    label_concat = torch.cat((label_src, label_tgt), 0)

    # compute loss for critic
    loss_critic = criterion(pred_concat, label_concat)
    loss_critic.backward()

    # optimize critic
    optimizer_critic.step()

    pred_cls = torch.squeeze(pred_concat.max(1)[1])
    acc = (pred_cls == label_concat).float().mean()

    ############################
    # 2.2 train target encoder #
    ############################

    # zero gradients for optimizer
    optimizer_critic.zero_grad()
    optimizer_tgt.zero_grad()

    # extract and target features
    feat_tgt = tgt_encoder(images_tgt, out_layer='conv3')

    # predict on discriminator
    pred_tgt = critic(feat_tgt)

    # prepare fake labels
    label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

    # compute loss for target encoder
    loss_tgt = criterion(pred_tgt, label_tgt)
    loss_tgt.backward()

    # optimize target encoder
    optimizer_tgt.step()

    #######################
    # 2.3 print step info #
    #######################
    print("d_loss={:.5f} g_loss={:.5f} acc={:.5f}".format(loss_critic.item(),
                    loss_tgt.item(),
                    acc.item()))

    torch.save(critic.state_dict(), os.path.join(
        opts['model_root'],
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        opts['model_root'],
        seq_name + "-ADDA-target-encoder-final.pt"))
    return tgt_encoder
