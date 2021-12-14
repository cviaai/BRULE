import json
import time
from typing import Callable, Any
import sys
import os

from parameters.path import Paths

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../dataset'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.toheatmap import ToHeatMap, heatmap_to_measure

import torch
from torch import optim
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from dataset.lazy_loader import LazyLoader, W300DatasetLoader, Celeba
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2DFactory
from metrics.writers import ItersCounter, send_images_to_tensorboard
from modules.hg import HG_softmax2020
from gan.loss_base import Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

counter = ItersCounter()
writer = SummaryWriter(f"{Paths.default.board()}/w300{int(time.time())}")

print(f"{Paths.default.board()}/w300{int(time.time())}")

def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True
    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        writer.add_scalar(name, loss.item(), counter.get_iter(name))
        return loss
    return decorated


def test(enc):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        content = enc(data)
        content_xy, _ = heatmap_to_measure(content)
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((content_xy - mes.coord).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    print("test loss: ", sum_loss / len(LazyLoader.w300().test_dataset))
    return sum_loss / len(LazyLoader.w300().test_dataset)


encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)
encoder_HG = encoder_HG.cuda()
encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 3])

cont_opt = optim.Adam(encoder_HG.parameters(), lr=5e-5, betas=(0.5, 0.97))

W300DatasetLoader.batch_size = 36
W300DatasetLoader.test_batch_size = 36
Celeba.batch_size = 36

heatmaper = ToHeatMap(64)


def hm_svoego_roda_loss(pred, target):

    pred_xy, _ = heatmap_to_measure(pred)
    t_xy, _ = heatmap_to_measure(target)

    return Loss(
        nn.BCELoss()(pred, target) +
        nn.MSELoss()(pred_xy, t_xy) * 0.0005 +
        (pred - target).abs().mean() * 0.3
    )

for epoch in range(30):

    for i, batch in enumerate(LazyLoader.w300().loader_train):

        counter.update(i + epoch*len(LazyLoader.w300().loader_train))

        data = batch['data'].to(device)
        mes = ProbabilityMeasureFabric(256).from_coord_tensor(batch["meta"]["keypts_normalized"]).cuda()
        target_hm = heatmaper.forward(mes.probability, mes.coord * 63)

        content = encoder_HG(data)
        hm_svoego_roda_loss(content, target_hm).minimize_step(cont_opt)

        if i % 100 == 0:
            test_loss = test(encoder_HG)
            writer.add_scalar("test_loss", test_loss, i + epoch*len(LazyLoader.w300().loader_train))

torch.save(encoder_HG.state_dict(), f"{Paths.default.models()}/hg2_e29.pt")

