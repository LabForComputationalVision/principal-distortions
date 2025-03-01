from torchvision import transforms
import torch as ch
import pickle as pckl
import argparse
from . import utils
from .image_helpers import load_and_preproc_image
from .models.early_vision_models import *
import os
import hashlib

def hash_string(s):
    """Generate a SHA-256 hash of the input string."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]  # Use the first 8 characters of the hash

EARLY_VISUAL_MODEL_DICT = {'LN':get_LN_model,
                           'LG':get_LG_model,
                           'LGG':get_LGG_model,
                           'LGN':get_LGN_model,
                           'Identity':get_identity_model}

image_preprocessing_dict = {
    'EARLY_VISUAL_PROCESSING' : transforms.Compose([
                            transforms.Grayscale(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            ])
    }

def run_neural_nets_principal_distortions(model_names, 
                                          image_path,
                                          lr_start_end,
                                          num_steps,
                                          tensor_norm,
                                          perturb_clamp_scale,
                                          gaussian_mask,
                                          print_step,
                                          objective_cutoff,
                                          device,
                                          image_preprocessing):    
    test_image = load_and_preproc_image(image_path, image_preprocessing).to(device)

    if gaussian_mask: # TODO: Handle this better or just remove the option. 
        g_window = Gauss2DWindow(test_image.shape[0], test_image.shape[0]/3)
        gaussian_mask = g_window.gauss_2d.to(device)
    else:
        gaussian_mask=None
    
    model_list = []
    for model_name in model_names:
        model = EARLY_VISUAL_MODEL_DICT[model_name](device=device)
        model.to(device)
        model_list.append(model)
    
    # TODO: Could make this more complicated, but this is fine for now 
    learning_rate_schedule = ch.logspace(lr_start_end[0],lr_start_end[-1],num_steps, base=10, device=device)
            
    opt_a, opt_b, objective_hist = utils.optimal_perturbations(model_list,
                                                               test_image,
                                                               learning_rate=learning_rate_schedule,
                                                               max_step=num_steps,
                                                               print_step=print_step,
                                                               tensor_norm=tensor_norm,
                                                               perturb_clamp_scale=perturb_clamp_scale,
                                                               alpha_mask=gaussian_mask,
                                                             )
    return opt_a, opt_b, objective_hist, test_image

class Gauss2DWindow(ch.nn.Module):
    def __init__(self, width, std):
        super().__init__()
        self.gauss_2d = self.make_gauss_2d(width, std)
        
    def forward(self, x):
        return ch.mul(x, self.gauss_2d)
    
    def make_gauss_1d(self, width, std):
        return ch.signal.windows.gaussian(width,std=std)
    
    def make_gauss_2d(self, width, std):
        gauss_1d = self.make_gauss_1d(width, std)
        gauss_2d = ch.unsqueeze(gauss_1d,1) @ ch.unsqueeze(gauss_1d,0)
        return gauss_2d


def save_results(results, output_path, params):
    with open(output_path, 'wb') as f:
        pckl.dump({"opt_perturbation_a": results[0],
                   "opt_perturbation_b": results[1],
                   "objective_hist":results[2],
                   "test_image":results[3],
                   "meta_data": params}, f)

def str_model_name_to_list(model_name_str):
    model_names = model_name_str.split(';')
    return model_names

def filename_from_args(args):
    base_path = args.output_base_path
    custom_model_name = args.output_custom_exp_name
    lr_str = f"lr_{args.lr_start_end[0]}_{args.lr_start_end[1]}"
    steps_str = f"steps_{args.num_steps}"
    tensor_norm_str = f"norm_{args.tensor_norm}"
    clamp_str = f"clamp_{args.perturb_clamp_scale}"
    gaussian_str = f"gaussian_{args.gaussian_mask}"
    cutoff_str = f"cutoff_{args.objective_cutoff}"
    seed_str = f"seed_{args.seed}"
    image_path_str = args.image_path.split('/')[-1].split('.')[0]
    model_names = str_model_name_to_list(args.model_names)
    
    # If custom model name is provided, use it; otherwise use the model names
    model_name = custom_model_name if custom_model_name else '_'.join([f"{model}" for model in model_names])
    if len(model_name)>40:
        model_name = hash_string(model_name)
    
    # Combine parts of the filename
    unique_str = f"{image_path_str}_{lr_str}_{steps_str}_{tensor_norm_str}_{clamp_str}_{gaussian_str}_{cutoff_str}_{seed_str}"

    # Return full path
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, model_name), exist_ok=True)
    return os.path.join(base_path, model_name, unique_str)        

def load_pckl_file(args):
    """Generate the filename based on arguments and load the pickle file if it exists."""
    args.image_path = parse_input_image_arg(args.image_path)

    # Rebuild the file path from args
    output_path = filename_from_args(args)
    
    if os.path.exists(output_path):
        print(f"Loading results from {output_path}")
        with open(output_path, 'rb') as f:
            data = pckl.load(f)
        return data
    else:
        raise FileNotFoundError(f"Pickle file not found at {output_path}")

def get_default_args():
    return {
        'lr_start_end': [-1., -3.],
        'num_steps': 100,
        'tensor_norm': 0.1,
        'perturb_clamp_scale': 0.001,
        'gaussian_mask': False, # Note: this is not used in argparse
        'seed':0,
        'print_step': 50,
        'objective_cutoff': 1000,
        'device': 'cuda',
        'image_preproc_type': 'EARLY_VISUAL_PROCESSING',
        'output_base_path': './results',
        'output_custom_exp_name': None,
    }

def parse_input_image_arg(x):
    if type(x)==int:
        image_path = f'images/kodaktid2008/kodim{x:02d}.png'
    else:
        image_path = x
    return image_path

def main(args):
    # Parse model_names
    model_names = str_model_name_to_list(args.model_names)
    image_preprocessing = image_preprocessing_dict[args.image_preproc_type]

    args.image_path = parse_input_image_arg(args.image_path)

    # Run the neural network with principal distortions
    results = run_neural_nets_principal_distortions(
        model_names,
        args.image_path,
        args.lr_start_end,
        args.num_steps,
        args.tensor_norm,
        args.perturb_clamp_scale,
        args.gaussian_mask,
        args.print_step,
        args.objective_cutoff,
        args.device,
        image_preprocessing,
    )

    # Generate a unique filename or use the custom model name
    output_path = filename_from_args(args)
    params = vars(args)
    save_results(results, output_path, params)

def int_or_str(value):
    try:
        return int(value)  # Try converting to int
    except ValueError:
        return value       # If conversion fails, return as string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural nets with principal distortions and save the output.")

    default_args = get_default_args()
    parser.add_argument('--model_names', type=str, required=True,
                        help="List of model names for early visual models (in the format: 'model1;model2')")
    parser.add_argument('--image_path', type=int_or_str, required=True, help="Path to the input image")
    parser.add_argument('--lr_start_end', type=float, nargs=2, default=default_args['lr_start_end'], help="Learning rate start and end values, log space (default: [0.,-2.])")
    parser.add_argument('--num_steps', type=int, default=default_args['num_steps'], help="Number of optimization steps (default: 100)")
    parser.add_argument('--tensor_norm', type=float, default=default_args['tensor_norm'], help="Tensor normalization value (default: 0.1)")
    parser.add_argument('--perturb_clamp_scale', type=float, default=default_args['perturb_clamp_scale'], help="Perturbation clamp scale (default: 0.001)")
    parser.add_argument('--gaussian_mask', action='store_true', help="Use Gaussian mask if this flag is set (default: False)")
    parser.add_argument('--print_step', type=int, default=default_args['print_step'], help="How often to print steps (default: 50)")
    parser.add_argument('--seed', type=int, default=default_args['seed'], help="How often to print steps (default: 0)")
    parser.add_argument('--objective_cutoff', type=float, default=default_args['objective_cutoff'], help="Objective cutoff value (default:1000)")
    parser.add_argument('--device', type=str, default=default_args['device'], help="Device to run the model on (default: 'cuda')")
    parser.add_argument('--image_preproc_type', type=str, default=default_args['image_preproc_type'], help="Type of preprocessing for the images, key in image_preproc_dict (default:DEFAULT_IMAGENET_TEST)")
    parser.add_argument('--output_base_path', type=str, default=default_args['output_base_path'], help="Base path to save the output file (default: './results')")
    parser.add_argument('--output_custom_exp_name', type=str, default=default_args['output_custom_exp_name'], help="Optional name for the set of models included in the experiment if it is a large set")

    args = parser.parse_args()

    main(args)
