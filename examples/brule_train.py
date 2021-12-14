import json
import sys, os

import albumentations


sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/gan/'))

from dataset.lazy_loader import LazyLoader, Celeba, W300DatasetLoader
from dataset.toheatmap import heatmap_to_measure, ToHeatMap, ToGaussHeatMap
from modules.hg import HG_softmax2020
from parameters.path import Paths

from loss.tuner import GoldTuner
from gan.loss.gan_loss import StyleGANLoss
from loss.regulariser import DualTransformRegularizer, BarycenterRegularizer, \
    UnoTransformRegularizer
from transforms_utils.transforms import ToNumpy, NumpyBatch, ToTensor, ResizeMask, \
    NormalizeMask

import random
import time
from typing import Optional, Callable, Any

import torch
from torch import nn, optim, Tensor

from torch.utils.tensorboard import SummaryWriter

from dataset.probmeasure import ProbabilityMeasureFabric, UniformMeasure2DFactory, \
    UniformMeasure2D01
from gan.gan_model import CondStyleGanModel, CondGen3, CondDisc3, \
    CondGenDecode
from gan.loss_base import Loss
from metrics.writers import ItersCounter, send_images_to_tensorboard
from models.munit.enc_dec import StyleEncoder
from stylegan2.model import Generator
from modules.linear_ot import SOT, PairwiseDistance

def handmadew1(m1,m2):
    lambd = 0.002
    with torch.no_grad():
        P = SOT(200, lambd).forward(m1, m2)
        M = PairwiseDistance()(m1.coord, m2.coord).sqrt()
    return (M * P).sum(dim=(1,2)) / 2

def W300IOD(encoder: nn.Module):
    sum_loss = 0
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].to(device)
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        landmarks[landmarks > 1] = 0.99999
        pred_measure = UniformMeasure2DFactory.from_heatmap(encoder(data))
        target = UniformMeasure2D01(torch.clamp(landmarks, max=1))
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += (handmadew1(pred_measure, target) / eye_dist).sum().item()
    return sum_loss / len(LazyLoader.w300().test_dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def imgs_with_mask(imgs, mask, color=[1.0,1.0,1.0]):
    mask = mask[:, 0, :, :]
    res: Tensor = imgs.cpu().detach()
    res = res.permute(0, 2, 3, 1)
    res[mask > 0.00001, :] = torch.tensor(color, dtype=torch.float32)
    res = res.permute(0, 3, 1, 2)
    return res

def stariy_hm_loss(pred, target, coef=1.0):

    pred_mes = UniformMeasure2DFactory.from_heatmap(pred)
    target_mes = UniformMeasure2DFactory.from_heatmap(target)

    return Loss(
        nn.BCELoss()(pred, target) * coef +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * (0.001 * coef) +
        nn.L1Loss()(pred_mes.coord, target_mes.coord) * (0.001 * coef)
    )

def hm_svoego_roda_loss(pred, target):

    pred_xy, _ = heatmap_to_measure(pred)
    with torch.no_grad():
        t_xy, _ = heatmap_to_measure(target)

    return Loss(
        nn.BCELoss()(pred, target) +
        nn.MSELoss()(pred_xy, t_xy) * 0.001
    )

def hm_loss_bes_xy(pred, target):

    return Loss(
        nn.BCELoss()(pred, target)
    )

counter = ItersCounter()
writer = SummaryWriter(f"{Paths.default.board()}/stylegan{int(time.time())}")
print(f"{Paths.default.board()}/stylegan{int(time.time())}")
l1_loss = nn.L1Loss()


def L1(name: Optional[str], writer: SummaryWriter = writer) -> Callable[[Tensor, Tensor], Loss]:

    if name:
        counter.active[name] = True

    def compute(t1: Tensor, t2: Tensor):
        loss = l1_loss(t1, t2)
        if name:
            if counter.get_iter(name) % 10 == 0:
                writer.add_scalar(name, loss, counter.get_iter(name))
        return Loss(loss)

    return compute


def writable(name: str, f: Callable[[Any], Loss]):
    counter.active[name] = True

    def decorated(*args, **kwargs) -> Loss:
        loss = f(*args, **kwargs)
        iter = counter.get_iter(name)
        if iter % 10 == 0:
            writer.add_scalar(name, loss.item(), iter)
        return loss

    return decorated


def entropy(hm: Tensor):
    B, N, D, D = hm.shape
    return Loss(-(hm * hm.log()).sum() / (B * D * D))


def gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt, heatmaper, g_transforms):

    def gan_train(i, real_img, pred_measures, sparse_hm, apply_g=True):
        batch_size = real_img.shape[0]
        latent_size = 512
        requires_grad(generator, True)

        coefs = json.load(open("../parameters/gan_loss.json"))

        if apply_g:
            trans_dict = g_transforms(image=real_img, mask=sparse_hm)
            trans_real_img = trans_dict["image"]
            trans_sparse_hm = trans_dict["mask"]
        else:
            trans_real_img = real_img
            trans_sparse_hm = sparse_hm

        noise = mixing_noise(batch_size, latent_size, 0.9, device)
        fake, _ = generator(trans_sparse_hm, noise, return_latents=False)

        model.disc_train([trans_real_img], [fake], trans_sparse_hm)

        writable("Generator loss", model.generator_loss)([trans_real_img], [fake], [], trans_sparse_hm) \
            .minimize_step(model.optimizer.opt_min)

        if i % 5 == 0:
            noise = mixing_noise(batch_size, latent_size, 0.9, device)

            fake, fake_latent = generator(trans_sparse_hm, noise, return_latents=True)

            fake_latent_test = fake_latent[:, [0, 13], :].detach()
            fake_latent_pred = style_encoder(fake)

            fake_content = encoder_HG(fake)

            restored = decoder(trans_sparse_hm, style_encoder(real_img))
            (
                    writable("BCE content gan", stariy_hm_loss)(fake_content, trans_sparse_hm, 5000) * coefs["BCE content gan"] +
                    L1("L1 restored")(restored, trans_real_img) * coefs["L1 restored"] +
                    L1("L1 style gan")(fake_latent_pred, fake_latent_test) * coefs["L1 style gan"] +
                    R_s(fake.detach(), fake_latent_pred) * coefs["R_s"]
            ).minimize_step(
                model.optimizer.opt_min,
                style_opt
            )

    return gan_train


def train_content(cont_opt, R_b, R_t, real_img, heatmaper, g_transforms):
    requires_grad(encoder_HG, True)

    coefs = json.load(open("../parameters/content_loss.json"))
    content = encoder_HG(real_img)
    pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content)
    sparse_hm = heatmaper.forward(pred_measures.coord * 63)

    ll = (
        writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"] +
        writable("Sparse", hm_loss_bes_xy)(content, sparse_hm.detach()) * coefs["Sparse"] +
        writable("R_t", R_t.__call__)(real_img, sparse_hm) * coefs["R_t"]
    )
    ll.minimize_step(cont_opt)


def content_trainer_with_gan(cont_opt, tuner, heatmaper, encoder_HG, R_b, R_t, model, generator, g_transforms):
    latent_size = 512

    def do_train(real_img):

        batch_size = real_img.shape[0]
        requires_grad(encoder_HG, True)
        requires_grad(generator, False)
        requires_grad(model.loss.discriminator, False)
        img_content = encoder_HG(real_img)
        pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(img_content)
        sparse_hm = heatmaper.forward(pred_measures.coord * 63).detach()
        restored = decoder(img_content, style_encoder(real_img))

        trans_content = g_transforms(image=real_img, mask=img_content)["mask"]

        noise1 = mixing_noise(batch_size, latent_size, 0.9, device)
        fake1, _ = generator(trans_content, noise1)
        trans_fake_content = encoder_HG(fake1.detach())

        coefs = json.load(open("../parameters/content_loss.json"))

        tuner.sum_losses([
            writable("Real-content D", model.loss.generator_loss)(
                real=None,
                fake=[real_img, img_content]) * coefs["Real-content D"],
            writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"],
            writable("Sparse", hm_loss_bes_xy)(img_content, sparse_hm) * coefs["Sparse"],
            writable("R_t", R_t.__call__)(real_img, sparse_hm) * coefs["R_t"],
            L1("L1 image")(restored, real_img) * coefs["L1 image"],
            writable("fake_content loss", stariy_hm_loss)(
                trans_fake_content, trans_content
            ) * coefs["fake_content loss"]
        ]).minimize_step(
            cont_opt
        )

    return do_train


def content_trainer_supervised(cont_opt, encoder_HG, loader):
    heatmaper = ToHeatMap(64)
    def do_train():
        requires_grad(encoder_HG, True)
        w300_batch = next(loader)
        w300_image = w300_batch['data'].to(device)
        w300_mes = ProbabilityMeasureFabric(256).from_coord_tensor(w300_batch["meta"]["keypts_normalized"]).cuda()
        w300_target_hm = heatmaper.forward(w300_mes.probability, w300_mes.coord * 63).detach()
        content300 = encoder_HG(w300_image)

        coefs = json.load(open("../parameters/content_loss.json"))

        writable("W300 Loss", hm_svoego_roda_loss)(content300, w300_target_hm).__mul__(coefs["borj4_w300"]).minimize_step(cont_opt)
    return do_train


def train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number):
    latent_size = 512
    batch_size = 24
    sample_z = torch.randn(8, latent_size, device=device)
    Celeba.batch_size = batch_size
    W300DatasetLoader.batch_size = batch_size
    W300DatasetLoader.test_batch_size = 64

    test_img = next(LazyLoader.celeba().loader)[:8].cuda()

    loss_st: StyleGANLoss = StyleGANLoss(discriminator)
    model = CondStyleGanModel(generator, loss_st, (0.001, 0.0015))

    style_opt = optim.Adam(style_encoder.parameters(), lr=5e-4, betas=(0.9, 0.99))
    cont_opt = optim.Adam(encoder_HG.parameters(), lr=4e-5, betas=(0.5, 0.97))

    g_transforms: albumentations.DualTransform = albumentations.Compose([
        ToNumpy(),
        NumpyBatch(albumentations.Compose([
            ResizeMask(h=256, w=256),
            albumentations.ElasticTransform(p=0.7, alpha=150, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.7, rotate_limit=15),
            ResizeMask(h=64, w=64),
            NormalizeMask(dim=(0, 1, 2))
        ])),
        ToTensor(device),
    ])

    R_t = DualTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img:
        stariy_hm_loss(encoder_HG(trans_dict['image']), trans_dict['mask'])
    )

    R_s = UnoTransformRegularizer.__call__(
        g_transforms, lambda trans_dict, img, ltnt:
        L1("R_s")(ltnt, style_encoder(trans_dict['image']))
    )

    barycenter: UniformMeasure2D01 = UniformMeasure2DFactory.load(f"{Paths.default.models()}/face_barycenter_68").cuda().batch_repeat(batch_size)

    R_b = BarycenterRegularizer.__call__(barycenter, 1.0, 2.0, 4.0)

    tuner = GoldTuner([2.2115, 1.6920, 1.4108, 1.0847, 0.8912, 2.0171], device=device, rule_eps=0.03, radius=1, active=True)

    heatmaper = ToGaussHeatMap(64, 1.0)
    sparse_bc = heatmaper.forward(barycenter.coord * 63)
    sparse_bc = nn.Upsample(scale_factor=4)(sparse_bc).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1) * \
                       torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)
    sparse_bc = (sparse_bc - sparse_bc.min()) / sparse_bc.max()
    send_images_to_tensorboard(writer, sparse_bc, "BC", 0, normalize=False, range=(0, 1))

    trainer_gan = gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt, heatmaper, g_transforms)
    content_trainer = content_trainer_with_gan(cont_opt, tuner, heatmaper, encoder_HG, R_b, R_t, model, generator, g_transforms)

    for i in range(100000):
        counter.update(i)

        requires_grad(encoder_HG, False)
        real_img = next(LazyLoader.celeba().loader).to(device)

        img_content = encoder_HG(real_img)
        pred_measures: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(img_content)
        sparse_hm = heatmaper.forward(pred_measures.coord * 63).detach()

        if i % 3 == 0:
            trainer_gan(i, real_img, pred_measures.detach(), sparse_hm.detach(), apply_g=True)
            content_trainer(real_img)

        if i % 100 == 0:
            coefs = json.load(open("../parameters/content_loss.json"))
            print(i, coefs)
            with torch.no_grad():

                content_test = encoder_HG(test_img)
                pred_measures_test: UniformMeasure2D01 = UniformMeasure2DFactory.from_heatmap(content_test)
                heatmaper_256 = ToGaussHeatMap(256, 2.0)
                sparse_hm_test = heatmaper.forward(pred_measures_test.coord * 63)
                sparse_hm_test_1 = heatmaper_256.forward(pred_measures_test.coord * 255)

                latent_test = style_encoder(test_img)

                sparce_mask = sparse_hm_test_1.sum(dim=1, keepdim=True)
                sparce_mask[sparce_mask < 0.0003] = 0
                iwm = imgs_with_mask(test_img, sparce_mask)
                send_images_to_tensorboard(writer, iwm, "REAL", i)

                fake_img, _ = generator(sparse_hm_test, [sample_z])
                iwm = imgs_with_mask(fake_img, pred_measures_test.toImage(256))
                send_images_to_tensorboard(writer, iwm, "FAKE", i)

                restored = decoder(sparse_hm_test, latent_test)
                iwm = imgs_with_mask(restored, pred_measures_test.toImage(256))
                send_images_to_tensorboard(writer, iwm, "RESTORED", i)

                content_test_256 = nn.Upsample(scale_factor=4)(sparse_hm_test).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1) * \
                    torch.tensor([1.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)

                content_test_256 = (content_test_256 - content_test_256.min()) / content_test_256.max()
                send_images_to_tensorboard(writer, content_test_256, "HM", i, normalize=False, range=(0, 1))

        if i % 50 == 0 and i > 0:
            test_loss = W300IOD(encoder_HG)
            tuner.update(test_loss)
            writer.add_scalar("W300IOD", test_loss, i)

        if i % 10000 == 0 and i > 0:
            torch.save(
                {
                    'g': generator.module.state_dict(),
                    'd': discriminator.module.state_dict(),
                    'c': encoder_HG.module.state_dict(),
                    "s": style_encoder.state_dict()
                },
                f'{Paths.default.models()}/stylegan2_new_{str(i + starting_model_number).zfill(6)}.pt',
            )


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    encoder_HG = HG_softmax2020(num_classes=68, heatmap_size=64)

    print("HG")

    latent = 512
    n_mlp = 5
    size = 256

    generator = CondGen3(Generator(
        size, latent, n_mlp, channel_multiplier=1
    ))

    discriminator = CondDisc3(
        size, channel_multiplier=1
    )

    style_encoder = StyleEncoder(style_dim=latent)

    starting_model_number = 160000
    weights = torch.load(
        f'{Paths.default.models()}/stylegan2_new_{str(starting_model_number).zfill(6)}.pt',
        map_location="cpu"
    )

    discriminator.load_state_dict(weights['d'])
    generator.load_state_dict(weights['g'])
    style_encoder.load_state_dict(weights['s'])

    generator = generator.cuda()
    discriminator = discriminator.to(device)
    encoder_HG = encoder_HG.cuda()
    style_encoder = style_encoder.cuda()
    decoder = CondGenDecode(generator)

    generator = nn.DataParallel(generator, [0, 1, 3])
    discriminator = nn.DataParallel(discriminator, [0, 1, 3])
    encoder_HG = nn.DataParallel(encoder_HG, [0, 1, 3])
    decoder = nn.DataParallel(decoder, [0, 1, 3])

    train(generator, decoder, discriminator, encoder_HG, style_encoder, device, starting_model_number)
