import matplotlib.pylab as plt
import numpy as np
import torch as ch
import pkg_resources
import seaborn as sns

from . import utils
from .run_neural_nets_principal_distortions import load_pckl_file
from .run_early_visual_models_principal_distortions import load_pckl_file as load_pckl_file_early_visual_models
from .run_neural_nets_principal_distortions_bg_images import load_pckl_file as load_pckl_file_bg_images

imagenet_classes_txt = pkg_resources.resource_filename(__name__, "imagenet_classes.txt")
with open(imagenet_classes_txt, "r") as f:
    imagenet_categories = [s.strip() for s in f.readlines()]

def get_example_image(early_vision_models=False):
    if early_vision_models:
        image_path = pkg_resources.resource_filename(__name__, "images/parrot.png")
    else:
        image_path = pkg_resources.resource_filename(__name__, "images/ChickadeeCC.jpg")
    return image_path

def plot_model_1d_scatter(model_list, test_image, perturb_a, perturb_b,
                          opacity_dict=None, x_lims=None, y_lims=None, 
                          title=None, ax=None, y_locations=None,
                          model_cmap=None, model_layer_offset=None,
                          print_imagenet_category=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,2))
    
    if model_cmap is None:
        model_cmap = sns.color_palette('colorblind')

    model_cmap_idx = 0
    model_colors = {}
    model_markers = {}
    model_locations = {}
    model_name_list = []
    if y_locations is None:
        y_locations = len(model_list) - np.arange(len(model_list))
    if model_layer_offset is None:
        model_layer_offset = 0

    for m_idx, m in enumerate(model_list):
        t_1, t_2 = utils.compute_model_thresh(m, test_image, perturb_a, perturb_b)
        d_e1 = 1/t_1
        d_e2 = 1/t_2
        log_ratio = ch.log(d_e1 /d_e2).detach().cpu().numpy()
        if m.model_name not in model_colors.keys():
            model_colors[m.model_name] = model_cmap[model_cmap_idx]
            model_locations[m.model_name] = y_locations[model_cmap_idx]
            model_cmap_idx+=1
            model_name_list.append(m.model_name)
        m_color = model_colors[m.model_name]

        if opacity_dict is not None:
            model_opacity = opacity_dict[m.layer_name]
        else:
            model_opacity = 1

        ax.scatter(log_ratio,
                   model_locations[m.model_name] + (model_layer_offset*model_opacity - model_layer_offset*0.5),
                   color=m_color,
#                    alpha=model_opacity, # Changed to only modify the size. 
#                    alpha=0.25,
                   s=400*model_opacity**2)

        if print_imagenet_category and ('final_softmax' in m.layer_name):
            model_output_a = m(test_image+perturb_a*40)
            model_output_b = m(test_image+perturb_b*40)
            model_output_orig = m(test_image)
            orig_max_idx = np.argmax(model_output_orig.cpu().detach().numpy().ravel())
            class_orig = imagenet_categories[orig_max_idx]
            print(f'{m.model_name} | Orig: {class_orig}, Prob {model_output_orig[0,orig_max_idx]}')
            a_max_idx = np.argmax(model_output_a.cpu().detach().numpy().ravel())
            class_a = imagenet_categories[a_max_idx]
            norm_a = ch.norm(model_output_a-model_output_orig)
            print(f'{m.model_name} | A: {class_a}, Prob {model_output_a[0,a_max_idx]},{norm_a}')
            b_max_idx = np.argmax(model_output_b.cpu().detach().numpy().ravel())
            class_b = imagenet_categories[b_max_idx]
            norm_b = ch.norm(model_output_b-model_output_orig)
            print(f'{m.model_name} | B: {class_b}, Prob {model_output_b[0,b_max_idx]},{norm_b}')
            print('\n')

    (y_tick_labels, y_tick_vals) = zip(*model_locations.items())
    ax.set_yticks(y_tick_vals, y_tick_labels)
    ax.set_ylim([min(y_tick_vals)-0.5,max(y_tick_vals)+0.5])
    ax.yaxis.set_tick_params(length=0)
    ax.spines[['right', 'left', 'top']].set_visible(False)
    if x_lims is not None:
        ax.set_xlim(x_lims)
    if y_lims is not None:
        ax.set_ylim(y_lims)
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('log(d(e_1) / d(e_2))')
    return ax

def plot_model_1d_with_error_bars(model_list, test_image_list,
                                  perturb_a_list, perturb_b_list,
                                  opacity_dict=None, x_lims=None, y_lims=None,
                                  model_cmap=None,
                                  title=None, ax=None, y_locations=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,2))
    
    if model_cmap is None:
        model_cmap = sns.color_palette('colorblind')
    model_cmap_idx = 0
    model_colors = {}
    model_markers = {}
    model_locations = {}
    model_name_list = []
    if y_locations is None:
        y_locations = len(model_list) - np.arange(len(model_list))

    for m_idx, m in enumerate(model_list):
        log_ratio_list = []
        for (test_image, perturb_a, perturb_b) in zip(test_image_list, perturb_a_list, perturb_b_list):
            t_1, t_2 = utils.compute_model_thresh(m, test_image, perturb_a, perturb_b)
            d_e1 = 1/t_1
            d_e2 = 1/t_2
            log_ratio = ch.log(d_e1 /d_e2).detach().cpu().numpy()
            log_ratio_list.append(log_ratio)
            
        if m.model_name not in model_colors.keys():
            model_colors[m.model_name] = model_cmap[model_cmap_idx]
            model_locations[m.model_name] = y_locations[model_cmap_idx]
            model_cmap_idx+=1
            model_name_list.append(m.model_name)
        m_color = model_colors[m.model_name]

        if opacity_dict is not None:
            model_opacity = opacity_dict[m.layer_name]
        else:
            model_opacity = 1
            
        x_val = np.mean(log_ratio_list)
        x_err = np.std(log_ratio_list)
        ax.errorbar(x_val,
                    model_locations[m.model_name],#+model_opacity/2-0.25,
                    xerr=x_err,
                    color=m_color,
                    fmt='.',
                    capsize=8,
#                     alpha=0.8,
                    markersize=25*model_opacity)#**2)

    (y_tick_labels, y_tick_vals) = zip(*model_locations.items())
    ax.set_yticks(y_tick_vals, y_tick_labels)
    ax.set_ylim([min(y_tick_vals)-0.5,max(y_tick_vals)+0.5])
    ax.yaxis.set_tick_params(length=0)
    ax.spines[['right', 'left', 'top']].set_visible(False)
    if x_lims is not None:
        ax.set_xlim(x_lims)
    if y_lims is not None:
        ax.set_ylim(y_lims)
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('log(d(e_1) / d(e_2))')
    return ax

def process_tensor_image_for_imshow(x):
    if x.shape[1]==1:
        return np.rollaxis(x.detach().cpu().numpy()[0,:,:,:], 0, 3)
    else:
        return np.rollaxis(x.detach().cpu().numpy()[0,:,:,:], 0, 3)

def imshow_rgb_or_grey(x):
    if x.shape[-1]==1:
        plt.imshow(x, interpolation='none',  vmin=0, vmax=1,  cmap='gray')
    elif x.shape[-1]==3:
        plt.imshow(x, interpolation='none')


def show_examples_with_varying_perturbations(all_args, 
                                             image_idx_list, 
                                             model_list=None,
                                             opacity_dict=None,
                                             perturb_scales=[200,100,50],
                                             vertical_layout=False,
                                             model_for_positive_thresh=None,
                                             model_cmap=None,
                                             bg_image_type=None,
                                             print_imagenet_category=False,
                                             flip_positive_perturbation=False,
                                             log_thresh_lims=[-12,12],
                                             model_layer_offset=None):
    # For ImageNet Images that are grayscale
    plt.rcParams['image.cmap'] = 'gray'
    
    # Rescale the printed eps values to assume norm 1. For optimization we sometimes 
    # use a smaller norm. 
    s_correction = 1 / all_args.tensor_norm
  
    num_images = len(image_idx_list)
    if model_list is None:
        num_plot_elements = len(perturb_scales)*2 + 3
        if vertical_layout:
            base_plot_offset = 1
        else:
            base_plot_offset = 0
    else:
        num_plot_elements = len(perturb_scales)*2 + 4
        if vertical_layout:
            base_plot_offset = num_images + 1
        else:
            base_plot_offset = 1

    if vertical_layout:
        fig = plt.figure(figsize=(6*num_images,6*(num_plot_elements)))
    else:
        fig = plt.figure(figsize=(6*(num_plot_elements),6*num_images))
        
    for plt_idx, image_idx in enumerate(image_idx_list):
        all_args.image_path = image_idx
        if bg_image_type is not None:
            output = load_pckl_file_bg_images(all_args, bg_image_type)
        elif hasattr(all_args, 'model_layer_pairs'):
            output = load_pckl_file(all_args)
        else:
            output = load_pckl_file_early_visual_models(all_args)
        test_image = output['test_image']

        opt_perturbation_a_tmp = output['opt_perturbation_a']
        opt_perturbation_b_tmp = output['opt_perturbation_b']
    
        if model_for_positive_thresh is not None:
            # Flip so that one of the models is always positive. 
            t_1_tmp, t_2_tmp = utils.compute_model_thresh(model_for_positive_thresh,
                                                  test_image,
                                                  opt_perturbation_a_tmp,
                                                  opt_perturbation_b_tmp)
            d_e1_tmp = 1/t_1_tmp
            d_e2_tmp = 1/t_2_tmp
            log_ratio_tmp = ch.log(d_e1_tmp /d_e2_tmp).detach().cpu().numpy()

            if log_ratio_tmp>0:
                if not flip_positive_perturbation:
                    opt_perturbation_a=opt_perturbation_a_tmp.clone()
                    opt_perturbation_b=opt_perturbation_b_tmp.clone()
                else:
                    opt_perturbation_a=opt_perturbation_b_tmp.clone()
                    opt_perturbation_b=opt_perturbation_a_tmp.clone()
            else:
                if not flip_positive_perturbation:
                    opt_perturbation_b=opt_perturbation_a_tmp.clone()
                    opt_perturbation_a=opt_perturbation_b_tmp.clone()
                else:
                    opt_perturbation_b=opt_perturbation_b_tmp.clone()
                    opt_perturbation_a=opt_perturbation_a_tmp.clone()
        else:
            if not flip_positive_perturbation:
                opt_perturbation_a=opt_perturbation_a_tmp.clone()
                opt_perturbation_b=opt_perturbation_b_tmp.clone()
            else:
                opt_perturbation_a=opt_perturbation_b_tmp.clone()
                opt_perturbation_b=opt_perturbation_a_tmp.clone()

        if vertical_layout:
            plot_offset = plt_idx + base_plot_offset

            # Natural Image
            plt.subplot((num_plot_elements),num_images,int(np.ceil(num_plot_elements/2)-1)*num_images + plot_offset)
            processed_image = process_tensor_image_for_imshow(test_image)
            imshow_rgb_or_grey(process_tensor_image_for_imshow(test_image))
            plt.axis('off')
            plt.title('Original Image')

            if model_list is not None:
                ax = plt.subplot((num_plot_elements)*2,num_images,plt_idx + 1) # Make this plot not as tall. 
                ax = plot_model_1d_scatter(model_list, test_image,
                         opt_perturbation_a, opt_perturbation_b,
                         opacity_dict=opacity_dict,
                                           ax = ax,
                         x_lims=log_thresh_lims,
                         title='Optimal Perturbations',
                                           model_cmap=model_cmap,
                                           print_imagenet_category=print_imagenet_category,
                                           model_layer_offset=model_layer_offset)

            # Just Perturbations
            if len(perturb_scales)>0:
                scale_just_perturb = perturb_scales[0]
            else:
                scale_just_perturb = 200
            plt.subplot(num_plot_elements,num_images,plot_offset)
            imshow_rgb_or_grey(process_tensor_image_for_imshow(0.5 + opt_perturbation_a*scale_just_perturb))
            plt.axis('off')
            plt.title(f'eps_1 (*{scale_just_perturb/s_correction})')

            plt.subplot(num_plot_elements,num_images, (num_plot_elements-1) * num_images + 1 + plt_idx)
            imshow_rgb_or_grey(process_tensor_image_for_imshow(0.5 + opt_perturbation_b*scale_just_perturb))
            plt.axis('off')
            plt.title(f'eps_2 (*{scale_just_perturb/s_correction})')

            # Perturbed Images
            for p_idx, p_multiply in enumerate(perturb_scales):
                plt.subplot((num_plot_elements), num_images,(p_idx+1)*num_images + plot_offset)
                tmp_a = test_image + opt_perturbation_a*p_multiply
                imshow_rgb_or_grey(process_tensor_image_for_imshow(tmp_a))
                plt.axis('off')
                plt.title(f'Image + eps_1 (*{p_multiply/s_correction})')
               
                plt.subplot((num_plot_elements),num_images,(num_plot_elements-2-p_idx)*num_images + 1 + plt_idx)
                tmp_b = test_image + opt_perturbation_b*p_multiply
                imshow_rgb_or_grey(process_tensor_image_for_imshow(tmp_b))
                plt.axis('off')
                plt.title(f'Image + eps_2 (*{p_multiply/s_correction})')
                
        else: # Horizontal Layout
            plot_offset = plt_idx*(num_plot_elements) + base_plot_offset

            # Natural Image
            plt.subplot(num_images,(num_plot_elements),int(np.ceil(num_plot_elements/2)) + plot_offset)
            imshow_rgb_or_grey(process_tensor_image_for_imshow(test_image))
            plt.axis('off')
            plt.title('Original Image')

            # Include the dot plot as the first plot
            if model_list is not None:
                ax = plt.subplot(num_images,num_plot_elements,plot_offset)
                ax = plot_model_1d_scatter(model_list, test_image,
                         opt_perturbation_a, opt_perturbation_b,
                         opacity_dict=opacity_dict,
                                           ax = ax,
                         x_lims=log_thresh_lims,
                         title='Optimal Perturbations',
                                           model_cmap=model_cmap,
                                           print_imagenet_category=print_imagenet_category,
                                           model_layer_offset=model_layer_offset)

            # Just Perturbations
            scale_just_perturb = 200
            plt.subplot(num_images,(num_plot_elements), plot_offset+1)
            imshow_rgb_or_grey(process_tensor_image_for_imshow(0.5 + opt_perturbation_a*scale_just_perturb))
            plt.axis('off')
            plt.title(f'eps_1 {scale_just_perturb/s_correction}')

            if model_list is None:
                plt.subplot(num_images,(num_plot_elements), plot_offset + num_plot_elements)
            else:
                plt.subplot(num_images,(num_plot_elements), plot_offset + num_plot_elements - 1)
            imshow_rgb_or_grey(process_tensor_image_for_imshow(0.5 + opt_perturbation_b*scale_just_perturb))
            plt.axis('off')
            plt.title(f'eps_2 {scale_just_perturb/s_correction}')

            # Perturbed Images
            for p_idx, p_multiply in enumerate(perturb_scales):
                plt.subplot(num_images,(num_plot_elements), plot_offset + 2 + p_idx)
                tmp_a = test_image + opt_perturbation_a*p_multiply
                imshow_rgb_or_grey(process_tensor_image_for_imshow(tmp_a))
                plt.axis('off')
                plt.title(f'Image + eps_1 (*{p_multiply/s_correction})')

                if model_list is None:
                    plt.subplot(num_images,(num_plot_elements), plot_offset + num_plot_elements - (p_idx + 1))
                else:
                    plt.subplot(num_images,(num_plot_elements), plot_offset + num_plot_elements - (p_idx + 1) - 1)
                tmp_b = test_image + opt_perturbation_b*p_multiply
                imshow_rgb_or_grey(process_tensor_image_for_imshow(tmp_b))
                plt.axis('off')
                plt.title(f'Image + eps_2 (*{p_multiply/s_correction})')
