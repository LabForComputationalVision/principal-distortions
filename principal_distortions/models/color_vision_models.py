import torch as ch

# Contains the different modules that will convert the LMS color space
# to a color "representation" space. Here, we drop out channels of LMS
# (a crude approximation, but useful as demo)
color_sim_dict_from_lms = {
    'protan' : ch.nn.Sequential(MatMulColorVision('protan',
                                              [[0,1,0],
                                               [0,1,0],
                                               [0,0,1]]),
                                LMStoOKLAB()),
    'deutan' : ch.nn.Sequential(MatMulColorVision('deutan',
                                              [[1,0,0],
                                               [1,0,0],
                                               [0,0,1]]),
                                LMStoOKLAB()),
    'trichromacy': LMStoOKLAB()
}


class ColorVisionModel(ch.nn.Module):
    """
    Pytorch module that simulates different forms of color 
    blindness. 
    """
    def __init__(self, vision_type=None, alpha_mask=None, device='cuda'):
        super(ColorVisionModel, self).__init__()
        # TODO: the matrices should be buffers and moved with the device placement.
        self.vision_type = vision_type
        self.model_name = vision_type
        if self.vision_type is not None:
            self.vision_rep = color_sim_dict_from_lms[vision_type].to(device)
        self.alpha_mask = alpha_mask
        if alpha_mask is not None:
            self.alpha_mask = alpha_mask.to(device)
        else:
            self.alpha_mask = alpha_mask
        self.LMS_from_linearRGB = self.lms_from_linear_rgb_mat().to(device)
        self.linearRGB_from_LMS = ch.linalg.inv(self.LMS_from_linearRGB).to(device)
        
    def forward(self, x):
        if self.alpha_mask is not None:
            x = self.alpha_mask * x
        x_rgb = linearRGB_from_sRGB_torch(x)
        x_rgb = ch.moveaxis(x_rgb, -3, -1)
        x_lms = ch.matmul(x_rgb, self.LMS_from_linearRGB)
#         x_simulated = ch.matmul(x_lms, self.vision_matrix)
        if self.vision_type is not None:
            x_simulated = self.vision_rep(x_lms)
            return x_simulated
        else:
            return x_lms
    
    def model_to_srgb(self, x_simulated):
        x_simulated = ch.matmul(x_simulated, self.linearRGB_from_LMS)
        x_simulated = ch.moveaxis(x_simulated, -1, -3)
        x_simulated_srgb = sRGB_from_linearRGB_torch(x_simulated)
        return x_simulated_srgb
    
    def lms_from_linear_rgb_mat(self):
        """sRGB Viénot
        Viénot, F., & Le Rohellec, J. (2013). 
        Colorimetry and Physiology - The LMS Specification. 
        https://doi.org/10.1002/9781118562680.ch1
        """
        LMS_from_linearRGB = ch.tensor([
            [17.88240413, 43.51609057,  4.11934969],
            [ 3.45564232, 27.15538246,  3.86713084],
            [ 0.02995656,  0.18430896,  1.46708614]]).T
        return LMS_from_linearRGB


class MatMulColorVision(ch.nn.Module):
    """Color vision matrix multiplication module."""
    def __init__(self, vision_type, vision_mat):
        super(MatMulColorVision, self).__init__()
        self.vision_type = vision_type
        self.register_buffer('vision_mat',
                             ch.tensor(vision_mat, dtype=ch.float))
    
    def forward(self, x):
        return ch.matmul(x, self.vision_mat.T)

    
class LMStoOKLAB(ch.nn.Module):
    """Color vision conversion module from LMS to OKLAB"""
    def __init__(self, clipped_grad=1.):
        super(LMStoOKLAB, self,).__init__()
        self.vision_type = 'trichromacy_oklab' 
        m2 = [[ 0.2104542553,  0.793617785,  -0.0040720468],
             [ 1.9779984951, -2.428592205,   0.4505937099],
             [ 0.0259040371,  0.7827717662, -0.808675766 ]]
        self.register_buffer('conversion_mat',
                             ch.tensor(m2, dtype=ch.float))
        self.clip_value = clipped_grad
        self.clipped_power = ClippedPower.apply
    
    def forward(self, x):
        x = self.clipped_power(x, self.clip_value, 1/3)
        return ch.matmul(x, self.conversion_mat.T)

    
class ClippedPower(ch.autograd.Function):
    """
    Takes the power of a signal and clips its gradients to the 
    provided values in the backwards pass
    """
    @staticmethod
    def forward(ctx, x, clip_value, power):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        ctx.power = power
        return ch.pow(x, power)
 
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = ctx.power * ch.pow(x, ctx.power-1)
        return grad_output * ch.clamp(g, -ctx.clip_value, ctx.clip_value), None, None


def sRGB_from_linearRGB_torch(im, im_in_min=0., im_in_max=1., gamma_value=2.2):
    """Convert linearRGB (renormalized to 0-1) to sRGB
    (returns in the range 0-1) applying the gamma correction.

    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB
    Code inspired by https://github.com/e-roe/dalton_simul

    Parameters
    ==========
    im : tensor of shape (M,N,3) with dtype float
    im_in_min : float corresponding to the minimum value used
        for normalization of im
    im_in_max : float corresponding to the maximum value used
        for normalization of im

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output sRGB image, range [0,1]
    """
    # Renormalize to be between 0-1
    im = (im - im_in_min) / (im_in_max - im_in_min)
    im = ch.clamp(im, 0., 1.)
    
    cutoff = 0.0031308
    cutoff_mask = im<=cutoff
    
    im[cutoff_mask] = im[cutoff_mask] * 12.92
    im[~cutoff_mask] = 1.055 * im[~cutoff_mask]**(1./gamma_value) - 0.055
    
    return ch.clamp(im, 0., 1.)


def linearRGB_from_sRGB_torch(im, im_in_min=0., im_in_max=1.,
                              gamma_value=2.2):
    """Convert sRGB (renormalized to 0-1) to linear RGB
    (returns in the range 0-1), removing the gamma correction.

    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB
    Code inspired by https://github.com/e-roe/dalton_simul

    Parameters
    ==========
    im : tensor of shape (M,N,3) with dtype float
    im_in_min : float corresponding to the minimum value used
        for normalization of im
    im_in_max : float corresponding to the maximum value used
        for normalization of im

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output RGB image, range [0,1]
    """
    # Renormalize to be between 0-1
    im = (im - im_in_min) / (im_in_max - im_in_min)
    im = ch.clamp(im, 0., 1.)
    
    cutoff = 0.04045
    cutoff_mask = im<=cutoff
    
    im[cutoff_mask] = im[cutoff_mask] / 12.92
    im[~cutoff_mask] = ((im[~cutoff_mask] + 0.055) / 1.055)**gamma_value
    return ch.clamp(im, 0., 1.)
