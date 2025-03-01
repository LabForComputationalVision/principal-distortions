from plenoptic.simulate.models import OnOff, LuminanceGainControl
import plenoptic as po
from collections import OrderedDict
import torch

def get_LN_model(device='cuda'):
    # LN, LinearNonLinear 
    ln = po.simulate.models.frontend.LinearNonlinear((31, 31), 
                                                     pad_mode="replicate").to(device)
    params_dict_ln = {"center_surround.center_std":torch.as_tensor([.5339]),
                   "center_surround.surround_std":torch.as_tensor([6.148]),
                   "center_surround.amplitude_ratio":torch.as_tensor([1.25])}
    ln.load_state_dict({k: torch.as_tensor([v]) for k, v in params_dict_ln.items()})
    ln.model_name = 'LinearNonLinear (LN)'
    return ln

def get_LG_model(device='cuda'):
    # LG, LuminanceGainControl
    lg = po.simulate.LuminanceGainControl((31, 31), pad_mode="replicate").to(device)
    params_dict_lg = {'luminance_scalar': 14.95, 'luminance.std': 4.235, 
                   'center_surround.center_std': 1.962, 'center_surround.surround_std': 4.235,
                   'center_surround.amplitude_ratio': 1.25}
    lg.load_state_dict({k: torch.as_tensor([v]) for k, v in params_dict_lg.items()})
    po.tools.remove_grad(lg)
    lg.eval()
    lg.model_name = 'LuminanceGainControl (LG)'
    return lg

def get_LGG_model(device='cuda'):
    # LGG, LuminanceContrastGainControl
    lgg = po.simulate.LuminanceContrastGainControl((31, 31), pad_mode="replicate").to(device)
    params_dict_lgg = {"luminance_scalar": torch.as_tensor([2.94]),
                   "contrast_scalar": torch.as_tensor([34.03]),
                   "center_surround.center_std" : torch.as_tensor([.7363]),
                   "center_surround.surround_std" : torch.as_tensor([48.37]),
                   "center_surround.amplitude_ratio" : torch.as_tensor([1.25]),
                   "luminance.std" : torch.as_tensor([170.99]),
                   "contrast.std" : torch.as_tensor([2.658])}
    lgg.load_state_dict({k: torch.as_tensor([v]) for k, v in params_dict_lgg.items()})
    po.tools.remove_grad(lgg)
    lgg.eval()
    lgg.model_name = 'LuminanceContrastGainControl (LGG)'
    return lgg

def get_LGN_model(device='cuda'):
    # OnOff (LGN)
    onoff_f = po.simulate.OnOff(kernel_size=(31, 31),
                                pretrained=False,
                                pad_mode="replicate",
                                apply_mask=False).to(device)
    params_dict_onoff = OrderedDict(
        [
            ("luminance_scalar", torch.as_tensor([3.2637, 14.3961])),
            ("contrast_scalar", torch.as_tensor([7.3405, 16.7423])),
            ("center_surround.center_std", torch.as_tensor([1.237, 0.3233])),
            ("center_surround.surround_std", torch.as_tensor([30.12, 2.184])),
            ("center_surround.amplitude_ratio", torch.as_tensor([1.25])),
            ("luminance.std", torch.as_tensor([76.4, 2.184])),
            ("contrast.std", torch.as_tensor([7.49, 2.43])),
        ]
    )
    onoff_f.load_state_dict({k: v for k, v in params_dict_onoff.items()})
    po.tools.remove_grad(onoff_f)
    onoff_f.eval()
    onoff_f.model_name = 'OnOff (LGN)'
    return onoff_f

def get_identity_model():
    model_identity = torch.nn.Identity()
    model_identity.model_name = 'Identity'
    return model_identity

EARLY_VISUAL_MODEL_DICT = {'LN':get_LN_model,
                           'LG':get_LG_model,
                           'LGG':get_LGG_model,
                           'LGN':get_LGN_model,
                           'Identity':get_identity_model}
