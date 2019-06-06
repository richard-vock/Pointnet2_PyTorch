from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse

from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from pointnet2.data import Indoor3DSemSeg

lr_clip = 1e-5
bnm_clip = 1e-2
batch_size = 32

if __name__ == "__main__":
    # test_set = Indoor3DSemSeg(4096, train=False)
    # test_loader = DataLoader(
        # test_set,
        # batch_size=32,
        # shuffle=True,
        # pin_memory=True,
        # num_workers=2,
    # )

    train_set = Indoor3DSemSeg(4096)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=6,
        shuffle=True,
    )

    model = Pointnet(num_classes=13, input_channels=6, use_xyz=True)
    model.cuda()

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    it = iter(train_loader)
    data = it.next()
    model.train()

    #self.optimizer.zero_grad()
    #_, loss, eval_res = self.model_fn(self.model, batch)
    _, _, eval_res = model_fn(model, data)

    #loss.backward()

    print(eval_res)
