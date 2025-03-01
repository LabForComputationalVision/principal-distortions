from torchvision import transforms
import torch as ch
import pickle as pckl
import argparse
from . import utils
from .image_helpers import load_and_preproc_image
from .models.model_helpers import * 
import os
import hashlib

def hash_string(s):
    """Generate a SHA-256 hash of the input string."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]  # Use the first 8 characters of the hash

image_preprocessing_dict = {
    'DEFAULT_IMAGENET_TEST' : transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            ]),
    'EVA_TRANFORMERS_IMAGENET_TEST' : transforms.Compose([
                            transforms.Resize(448,interpolation=transforms.InterpolationMode.BICUBIC, max_size=None),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            ]),
    'VITs_IMAGENET_TEST': transforms.Compose([
                            transforms.Resize(248,interpolation=transforms.InterpolationMode.BICUBIC, max_size=None),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            ])
    }

def run_neural_nets_principal_distortions(model_layer_pairs, 
                                          image_path,
                                          model_dir,
                                          lr_start_end,
                                          num_steps,
                                          tensor_norm,
                                          perturb_clamp_scale,
                                          gaussian_mask,
                                          print_step,
                                          objective_cutoff,
                                          device,
                                          image_preprocessing,
                                          model_type):
    test_image = load_and_preproc_image(image_path, image_preprocessing).to(device)

    if gaussian_mask: # TODO: Handle this better or just remove the option. 
        g_window = Gauss2DWindow(test_image.shape[0], test_image.shape[0]/4)
        gaussian_mask = g_window.gauss_2d.to(device)
    else:
        opt_mask=None
    
    model_list = []
    for (model_name, layer) in model_layer_pairs:
        if model_type=='featheretal2023':
            model, _ = load_metamer_models(model_dir, model_name)
            model.to(device)
            model_list.append(WrapMetamerModelLayer(model, layer))
        elif model_type=='timm':
            print(f'Loading {model_name}:{layer}')
            model = WrapTimmModelLayer(model_name, layer)
            model.to(device)
            model_list.append(model)
        else:
            raise NotImplementedError("Implemented model types are 'timm' and 'featheretal2023'")
    
    # TODO: Could make this more complicated, but this is fine for now 
    learning_rate_schedule = ch.logspace(lr_start_end[0],lr_start_end[-1],num_steps, base=10, device=device)

    with ch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        opt_a, opt_b, objective_hist = utils.optimal_perturbations(model_list,
                                                                   test_image,
                                                                   learning_rate=learning_rate_schedule,
                                                                   max_step=num_steps,
                                                                   print_step=print_step,
                                                                   tensor_norm=tensor_norm,
                                                                   perturb_clamp_scale=perturb_clamp_scale,
                                                                   alpha_mask=opt_mask,
                                                                  )

    return opt_a, opt_b, objective_hist, test_image

class Gauss2DWindow(ch.nn.Module):
    def __init__(self, width, std):
        super().__init__()
        self.gauss_2d = make_gauss_2d(width, std)
        
    def forward(self, x):
        return ch.mul(x, self.gauss_2d)
    
    def make_gauss_1d(self, width, std):
        return ch.signal.windows.gaussian(width,std=std)
    
    def make_gauss_2d(self, gauss_1d, width, std):
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

def str_model_layer_to_list(model_layer_str):
    model_layer_pairs = [(model_layer.split(',')[0], model_layer.split(',')[1]) for model_layer in model_layer_str.split(';')]
    return model_layer_pairs

def filename_from_args(args, im_type):
    if im_type not in ['orig', 'blur_fg', 'blur_bg']:
        raise ValueError
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
    class_str = args.image_path.split('/')[-2]
    model_layer_pairs = str_model_layer_to_list(args.model_layer_pairs)
    
    # If custom model name is provided, use it; otherwise use the model-layer pairs
    model_name = custom_model_name if custom_model_name else '_'.join([f"{model}_{layer}" for model, layer in model_layer_pairs])
    if len(model_name)>40:
        model_name = hash_string(model_name)
    
    # Combine parts of the filename
    unique_str = f"{class_str}_{image_path_str}_{im_type}_{lr_str}_{steps_str}_{tensor_norm_str}_{clamp_str}_{gaussian_str}_{cutoff_str}_{seed_str}"

    # Return full path
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, model_name), exist_ok=True)
    return os.path.join(base_path, model_name, unique_str)        

def load_pckl_file(args, im_type):
    """Generate the filename based on arguments and load the pickle file if it exists."""
    args.image_path = parse_input_image_arg(args.image_path, im_type)

    # Rebuild the file path from args
    output_path = filename_from_args(args, im_type)
    
    if os.path.exists(output_path):
        print(f"Loading results from {output_path}")
        with open(output_path, 'rb') as f:
            data = pckl.load(f)
        return data
    else:
        raise FileNotFoundError(f"Pickle file not found at {output_path}")

def get_default_args():
    return {
        'lr_start_end': [0., -2.],
        'num_steps': 100,
        'tensor_norm': 0.1,
        'perturb_clamp_scale': 0.001,
        'gaussian_mask': False, # Note: this is not used in argparse
        'seed':0,
        'print_step': 50,
        'objective_cutoff': 1000,
        'device': 'cuda',
        'image_preproc_type': 'DEFAULT_IMAGENET_TEST',
        'output_base_path': './results',
        'output_custom_exp_name': None,
        'model_type':'featheretal2023',
        'im_type':None,
    }

def parse_input_image_arg(x, im_type):
    if type(x)==int:
        with open('images/bg_challenge/2024_11_17_BG_Files.txt', 'r') as file:
            image_list = file.readlines()

        # Strip newline characters from each line
        image_list = [line.strip() for line in image_list]
        image_path = image_list[x]
    else:
        image_path = x

    if im_type=='orig':
        image_path = os.path.join('images/bg_challenge/blur_exp_subset_orig', image_path)
    elif im_type=='blur_fg':
        image_path = os.path.join('images/bg_challenge/blur_exp_subset_blur_foreground', image_path)
    elif im_type=='blur_bg':
        image_path = os.path.join('images/bg_challenge/blur_exp_subset_blur_background', image_path)
    else:
        raise ValueError

    return image_path

def main(args):
    # Parse model_layer_pairs
    model_layer_pairs = str_model_layer_to_list(args.model_layer_pairs)
    image_preprocessing = image_preprocessing_dict[args.image_preproc_type]

    raw_image_path = args.image_path

    if args.im_type is None:
        run_im_types = ['orig'] # ['orig', 'blur_fg', 'blur_bg']
    else:
        run_im_types = [args.im_type]

    # Run this for each of the three image types, orig, blur_fg, blur_bg
    for im_type in run_im_types:
        args.image_path = parse_input_image_arg(raw_image_path, im_type)

        # Generate a unique filename or use the custom model name
        output_path = filename_from_args(args, im_type)
        params = vars(args)
        print(output_path)
        if os.path.exists(output_path):
            print('File already exists! Delete file or implement overwrite if you want to run again')
            continue

        # Run the neural network with principal distortions
        results = run_neural_nets_principal_distortions(
            model_layer_pairs,
            args.image_path,
            args.model_dir,
            args.lr_start_end,
            args.num_steps,
            args.tensor_norm,
            args.perturb_clamp_scale,
            args.gaussian_mask,
            args.print_step,
            args.objective_cutoff,
            args.device,
            image_preprocessing,
            args.model_type,
        )
    
        save_results(results, output_path, params)

def int_or_str(value):
    try:
        return int(value)  # Try converting to int
    except ValueError:
        return value       # If conversion fails, return as string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural nets with principal distortions and save the output.")

    default_args = get_default_args()
    parser.add_argument('--model_layer_pairs', type=str, required=True,
                        help="List of model names and layers (in the format: 'model1,layer1;model2,layer2')")
    parser.add_argument('--image_path', type=int_or_str, required=True, help="Path to the input image")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the saved models, with build scripts")
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
    parser.add_argument('--model_type', type=str, default='featheretal2023', help="Choose the type of model to load, options are ['featheretal2023' and 'timm']")
    parser.add_argument('--im_type', type=str, default=None, help="Choose the type of base image to run, options are 'orig', 'blur_fg', 'blur_bg', default is to run all")

    args = parser.parse_args()

    main(args)
