# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.ae_model_zoo import DCAE_HF

dc_ae = DCAE_HF.from_pretrained(f"./dc-ae-f32c32-in-1.0")

# encode
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop

import torch
import numpy as np
import pandas as pd
import threading
import json
import time

def NNFN_ADacGD_PyTorch(M, Omega, k, lamda, eta=1e-10, Gamma=1e-10, theta=float('inf'), Theta=float('inf'), max_iterations=500, tolerance=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M = torch.tensor(M, dtype=torch.float64).to(device)
    Omega = torch.tensor(Omega, dtype=torch.float64).to(device)

    m, n = M.shape
    W_last = torch.randn(m, k, device=device, dtype=torch.float64)
    H_last = torch.randn(n, k, device=device, dtype=torch.float64)
    # U, S, V = torch.svd(M)
    # sqrt_S = torch.sqrt(S)
    # W_last = U[:, :k]@torch.diag(sqrt_S[:k])
    # H_last = (torch.diag(sqrt_S[:k])@V[:, :k].T).T

    eta_W = torch.tensor(eta, dtype=torch.float64).to(device)
    eta_H = torch.tensor(eta, dtype=torch.float64).to(device)
    theta_W = torch.tensor(theta, dtype=torch.float64).to(device)
    theta_H = torch.tensor(theta, dtype=torch.float64).to(device)
    Theta_W = torch.tensor(Theta, dtype=torch.float64).to(device)
    Theta_H = torch.tensor(Theta, dtype=torch.float64).to(device)
    Gamma_W = torch.tensor(Gamma, dtype=torch.float64).to(device)
    Gamma_H = torch.tensor(Gamma, dtype=torch.float64).to(device)

    W = W_last.clone()
    H = H_last.clone()
    Q = torch.mm(W, H.t())
    c = lamda / torch.norm(torch.mm(W, H.t()))


    nabula_W_last = (Q - M)* Omega @ H + lamda * W - c * W @ (H.t() @ H)
    # print(nabula_W_last)
    nabula_H_last = ((Q - M)* Omega).t() @ W + lamda * H - c * H @ (W.t() @ W)

    W = W_last - eta_W * nabula_W_last
    H = H_last - eta_H * nabula_H_last

    Wy = W.clone()
    Hy = H.clone()

    nmse_list = torch.zeros(max_iterations, device=device)

    for _ in range(max_iterations):
        Q = torch.mm(W, H.t())
        c = lamda / torch.norm(torch.mm(W, H.t()))
        nabula_W = (Q - M)* Omega @ H + lamda * W - c * W @ (H.t() @ H)
        # print(nabula_W)
        nabula_H = ((Q - M)* Omega).t() @ W + lamda * H - c * H @ (W.t() @ W)

        last_eta_W = eta_W.clone()
        last_eta_H = eta_H.clone()
        last_Gamma_W = Gamma_W.clone()
        last_Gamma_H = Gamma_H.clone()

        eta_W = min(torch.sqrt(1 + theta_W / 2) * eta_W, (torch.norm(W - W_last)) / (2 * torch.norm(nabula_W - nabula_W_last)))
        eta_H = min(torch.sqrt(1 + theta_H / 2) * eta_H, (torch.norm(H - H_last)) / (2 * torch.norm(nabula_H - nabula_H_last)))
           
        Gamma_W = min(torch.sqrt(1 + Theta_W / 2) * Gamma_W, (torch.norm(nabula_W - nabula_W_last)) / (2 * torch.norm(W - W_last)))
        Gamma_H = min(torch.sqrt(1 + Theta_H / 2) * Gamma_H, (torch.norm(nabula_H - nabula_H_last)) / (2 * torch.norm(H - H_last)))

        beta_W = (torch.sqrt(1 / eta_W) - torch.sqrt(Gamma_W)) / (torch.sqrt(1 / eta_W) + torch.sqrt(Gamma_W))
        beta_H = (torch.sqrt(1 / eta_H) - torch.sqrt(Gamma_H)) / (torch.sqrt(1 / eta_H) + torch.sqrt(Gamma_H))

        W_last = W.clone()
        H_last = H.clone()
        nabula_W_last = nabula_W.clone()
        nabula_H_last = nabula_H.clone()

        Wy_last = Wy.clone()
        Hy_last = Hy.clone()

        Wy = W - eta_W * nabula_W
        Hy = H - eta_H * nabula_H

        W = Wy + beta_W*(Wy - Wy_last)
        H = Hy + beta_H*(Hy - Hy_last) 

        X = torch.mm(W, H.t())

        theta_W = (eta_W/last_eta_W)
        theta_H = (eta_H/last_eta_H)
        Theta_W = (Gamma_W/last_Gamma_W)
        Theta_H = (Gamma_H/last_Gamma_H)

        nmse = torch.norm(X - M) / torch.norm(M)
        nmse_list[_] = nmse
        print(f"{_}_th nmse is {nmse.item()}")

        if nmse < tolerance:
            break

    # Convert the result back to CPU and NumPy for compatibility if needed
    return X, nmse_list

device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()

transform = transforms.Compose([
    DMCrop(512), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
import os
path = "assets/fig/"
# os.makedirs("./compare", exist_ok=True)
# savepath = "./compare/"
savepath = "./"
# for filename in os.listdir(path):
filename = 'girl.png'
image = Image.open(path + filename)
print(filename)
x = transform(image)[None].to(device)
latent = dc_ae.encode(x)
print(latent.shape)


# decode
y = dc_ae.decode(latent)
print(x.shape)
applix = filename.split(".")[0]
save_image(y * 0.5 + 0.5, savepath+applix+"_dc_ae.png")

recovery,_ = NNFN_ADacGD_PyTorch(latent.reshape([128,64]), np.ones([128,64]), 60, 0.01, max_iterations=500)
recovery =  torch.tensor(recovery, dtype=torch.float32).to(device)
print(recovery)
print(_[-1])
print(recovery.shape)
y2 = dc_ae.decode(recovery.reshape([1, 32, 16, 16]).to(device))
print(x.shape)
save_image(y2 * 0.5 + 0.5, savepath+applix+"_dc_ae_recover.png")

from pytorch_msssim import ssim

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Parameters:
        img1 (torch.Tensor): First image tensor, shape [B, C, H, W].
        img2 (torch.Tensor): Second image tensor, shape [B, C, H, W].

    Returns:
        float: SSIM value.
    """
    return ssim(img1, img2, data_range=1.0)

ss = calculate_ssim(y* 0.5 + 0.5, y2 * 0.5 + 0.5)
print("ssim: ", ss)

import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Parameters:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
        max_pixel_value (float): Maximum possible pixel value of the image (1.0 for normalized images).

    Returns:
        float: PSNR value.
    """
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr

psnr = calculate_psnr(y, y2)

print("psnr: ", psnr)
