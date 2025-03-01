from torchvision.transforms import ToTensor, CenterCrop
from PIL import Image
import torch as ch

def open_png_with_alpha_channel(image_path):
    """
    Opens a PNG that may have an alpha channel and converts it to a pytorch tensor. 
    """
    img_transparent = Image.open(image_path)
    img_transparent = ToTensor()(img_transparent)
    alpha_channel = img_transparent[3:,:,:]
    alpha_channel = ch.vstack([alpha_channel,alpha_channel,alpha_channel])
    rgb_channels = img_transparent[0:3,:,:]
    white_background = ch.ones_like(rgb_channels)
    img = rgb_channels * alpha_channel + white_background * (1 - alpha_channel)
    img = img.unsqueeze(0)
    return img, alpha_channel

def load_and_preproc_image(image_path, transforms):
    """
    Uses pytorch transforms to transform the input image
    """
    img = Image.open(image_path)
    img = transforms(img)
    img = ch.unsqueeze(img, 0)
    return img
