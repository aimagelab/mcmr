from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def get_IoU(mask_gt, mask_pred):
    bs = mask_gt.shape[0]
    mask_gt = (mask_gt.view(bs, -1).detach().cpu().numpy() > 0).astype(np.float32)
    mask_pred = (mask_pred.view(bs, -1).detach().cpu().numpy() > 0).astype(np.float32)

    intersection = mask_gt * mask_pred
    union = mask_gt + mask_pred - intersection
    iou = intersection.sum(1) / (union.sum(1) + 1e-12)

    return iou.mean()


def get_PCK(kp_gt, kp_pred, img_size, alpha=0.1):
    batch_size, n_kpoints, _ = kp_pred.shape

    kp_gt = kp_gt.detach().cpu().numpy()
    kp_gt[:, :, :2] = (kp_gt[:, :, :2] * 127.5) + 127.5
    kp_pred = kp_pred.detach().cpu().numpy()
    kp_pred = (kp_pred * 127.5) + 127.5

    x_margin = alpha * img_size[0]
    y_margin = alpha * img_size[1]

    visible_corrects = np.full(shape=(batch_size, n_kpoints), fill_value=np.nan)
    occluded_corrects = np.full(shape=(batch_size, n_kpoints), fill_value=np.nan)

    for b in range(batch_size):
        for n in range(n_kpoints):
            true_kpoint_idx = kp_gt[b][n]
            pred_kpoint_idx = kp_pred[b][n]

            x_dist = np.abs(true_kpoint_idx[1] - pred_kpoint_idx[1])
            y_dist = np.abs(true_kpoint_idx[0] - pred_kpoint_idx[0])

            is_correct = int(x_dist <= x_margin and y_dist <= y_margin)
            if true_kpoint_idx[2] == 0:  # kpoint coords really annotated on image
                visible_corrects[b][n] = is_correct
            else:  # kpoint coords inferred by keypoints projection on image
                occluded_corrects[b][n] = is_correct

    overall_corrects = np.nansum(np.stack([visible_corrects, occluded_corrects]), axis=0)

    hist_overall = np.nanmean(overall_corrects, 0)
    hist_visible = np.nanmean(visible_corrects, 0)
    hist_occluded = np.nanmean(occluded_corrects, 0)

    hist_overall[np.isnan(hist_overall)] = 0
    hist_visible[np.isnan(hist_visible)] = 0
    hist_occluded[np.isnan(hist_occluded)] = 0

    return np.mean(hist_overall), hist_overall


def get_L1(input_imgs, pred_imgs):
    return torch.nn.L1Loss()(pred_imgs, input_imgs)


def get_SSIM(input_imgs, pred_imgs):
    return ssim(input_imgs, pred_imgs)


def get_FID(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def compute_mean_and_cov(feats_list: list):
    feats = np.stack(feats_list, axis=0)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def get_feat(net, img: torch.Tensor):
    pred = net(img)[-1]
    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred = pred.squeeze()
    return pred.to('cpu').numpy()
