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

device = torch.device("cuda")
dc_ae = dc_ae.to(device).eval()

transform = transforms.Compose([
    DMCrop(512), # resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
image = Image.open("assets/fig/girl.png")
x = transform(image)[None].to(device)
latent = dc_ae.encode(x)
print(latent.shape)


# decode
y = dc_ae.decode(latent)
print(x.shape)
save_image(y * 0.5 + 0.5, "demo_dc_ae.png")

# reshape latent to 128,64
reshape_latent = latent.reshape([128, 64])
# print(reshape_latent)
# use svd on latent
U, S, V = torch.svd(reshape_latent)
# print(S)
# reserve top 50 singular values
S[64:] = 0
# reconstruct latent
reconstruct_latent = U @ torch.diag(S) @ V.t()
# print(reconstruct_latent)
# save reconstructed latent
y2 = dc_ae.decode(reconstruct_latent.reshape([1, 32, 16, 16]))
save_image(y2 * 0.5 + 0.5, "demo_dc_ae_reconstruct.png")
# compute nmse of reconstructed latent
nmse = torch.norm(reconstruct_latent - reshape_latent) / torch.norm(reshape_latent)
print(nmse)

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

psnr = calculate_psnr(y, y2 )

print("psnr: ", psnr)