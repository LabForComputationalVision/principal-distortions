import matplotlib.pyplot as plt
import torch
import warnings

from torch import nn
from torchvision import models

def fisher_info_matrix_vector_product(model, stimulus, vector):
    """Compute the product of the Fisher information matrix of a model at an stimulus and a vector"""
    jacob_vector_product = torch.autograd.functional.jvp(model, stimulus, vector)[1] # Check and make sure we don't need to square something here? 
    matrix_vector_product = torch.autograd.functional.vjp(model, stimulus, jacob_vector_product)[1]

    return matrix_vector_product

def perturbation_update(models, stimulus, perturbation_a, perturbation_b):
    """Compute the update to the perturbation"""

    n_models = len(models)
    device = perturbation_a.device

    FI_epsilon_a = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)
    FI_epsilon_b = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)
    
    sensitivity_a = torch.zeros(n_models, device=device)
    sensitivity_b = torch.zeros(n_models, device=device)
    
    scaled_FI_epsilon_a = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)
    scaled_FI_epsilon_b = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)
    
    diff_log_dis_thresh = torch.zeros(n_models, device=device)

    for idx, model in enumerate(models):

        # compute the product of the Fisher information matrix and the perturbation: $v(n)=I_n@eps$
        FI_epsilon_a[idx] = fisher_info_matrix_vector_product(model, stimulus, perturbation_a)
        FI_epsilon_b[idx] = fisher_info_matrix_vector_product(model, stimulus, perturbation_b)

        # compute the squared sensitivity: $t^2(n)=<eps, v(n)>$ 
        sensitivity_a[idx] = torch.inner(perturbation_a.flatten(), FI_epsilon_a[idx].flatten())
        sensitivity_b[idx] = torch.inner(perturbation_b.flatten(), FI_epsilon_b[idx].flatten())

        # scale the product by the (squared) discrimination threshold: $u(n)=v(n)/t(n)$
        scaled_FI_epsilon_a[idx] = FI_epsilon_a[idx]/sensitivity_a[idx]
        scaled_FI_epsilon_b[idx] = FI_epsilon_b[idx]/sensitivity_b[idx]

        # difference logarithm of the discrimination threshold: $r(n)=log t_1^2(n)-log t_2^2(n)$
        diff_log_dis_thresh[idx] = torch.log(sensitivity_a[idx]) - torch.log(sensitivity_b[idx])

    # take the mean across models: r = mean(r(n))
    diff_log_dis_thresh_mean = torch.mean(diff_log_dis_thresh)

    # take the mean across models: u = mean(u(n))
    scaled_FI_epsilon_mean_a = torch.mean(scaled_FI_epsilon_a, 0)
    scaled_FI_epsilon_mean_b = torch.mean(scaled_FI_epsilon_b, 0)

    # objective
    objective = torch.mean((diff_log_dis_thresh - diff_log_dis_thresh_mean)**2)

    for idx in range(len(models)):
        if idx==0:
            update_a = (diff_log_dis_thresh[idx] - diff_log_dis_thresh_mean)*(scaled_FI_epsilon_a[idx] - scaled_FI_epsilon_mean_a)
            update_b = -(diff_log_dis_thresh[idx] - diff_log_dis_thresh_mean)*(scaled_FI_epsilon_b[idx] - scaled_FI_epsilon_mean_b)
        else:
            update_a += (diff_log_dis_thresh[idx] - diff_log_dis_thresh_mean)*(scaled_FI_epsilon_a[idx] - scaled_FI_epsilon_mean_a)
            update_b += -(diff_log_dis_thresh[idx] - diff_log_dis_thresh_mean)*(scaled_FI_epsilon_b[idx] - scaled_FI_epsilon_mean_b)

    return update_a, update_b, objective

def compute_diff_objective(models, stimulus, perturbation_a, perturbation_b):
    # Note, this is actually 2*var(log(d(a)/d(b))), but it is the same optimization problem. We just define d based on the sqrt in the text. 
    n_models = len(models)
    device = perturbation_a.device

    FI_epsilon_a = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)
    FI_epsilon_b = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)

    sensitivity_a = torch.zeros(n_models, device=device)
    sensitivity_b = torch.zeros(n_models, device=device)

    scaled_FI_epsilon_a = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)
    scaled_FI_epsilon_b = torch.zeros((n_models, stimulus.shape[0], stimulus.shape[1], stimulus.shape[2], stimulus.shape[3]), device=device)

    diff_log_dis_thresh = torch.zeros(n_models, device=device)

    for idx, model in enumerate(models):

        # compute the product of the Fisher information matrix and the perturbation: $v(n)=I_n@eps$
        FI_epsilon_a[idx] = fisher_info_matrix_vector_product(model, stimulus, perturbation_a)
        FI_epsilon_b[idx] = fisher_info_matrix_vector_product(model, stimulus, perturbation_b)

        # compute the (squared) discrimination threshold: $t^2(n)=<eps, v(n)>$
        sensitivity_a[idx] = torch.inner(perturbation_a.flatten(), FI_epsilon_a[idx].flatten())
        sensitivity_b[idx] = torch.inner(perturbation_b.flatten(), FI_epsilon_b[idx].flatten())

        # scale the product by the (squared) discrimination threshold: $u(n)=v(n)/t(n)$
        scaled_FI_epsilon_a[idx] = FI_epsilon_a[idx]/sensitivity_a[idx]
        scaled_FI_epsilon_b[idx] = FI_epsilon_b[idx]/sensitivity_b[idx]

        # difference logarithm of the discrimination threshold: $r(n)=log t_1^2(n)-log t_2^2(n)$
        diff_log_dis_thresh[idx] = torch.log(sensitivity_a[idx]) - torch.log(sensitivity_b[idx])

    # take the mean across models: r = mean(r(n))
    diff_log_dis_thresh_mean = torch.mean(diff_log_dis_thresh)

    # objective
    objective = torch.mean((diff_log_dis_thresh - diff_log_dis_thresh_mean)**2)

    return objective

def compute_model_thresh(model, stimulus, perturbation_a, perturbation_b):
    # compute the product of the Fisher information matrix and the perturbation: $v(n)=I_n@eps$
    model_FI_epsilon_a = fisher_info_matrix_vector_product(model, stimulus, perturbation_a)
    model_FI_epsilon_b = fisher_info_matrix_vector_product(model, stimulus, perturbation_b)

    # compute the (squared) discriminability: $d^2(n)=<eps, v(n)>$
    model_discriminability_a = torch.inner(perturbation_a.flatten(), model_FI_epsilon_a.flatten())
    model_discriminability_b = torch.inner(perturbation_b.flatten(), model_FI_epsilon_b.flatten())

    # The threshold is 1/discriminability
    model_thresh_a = 1/torch.sqrt(model_discriminability_a)
    model_thresh_b = 1/torch.sqrt(model_discriminability_b)

    return model_thresh_a, model_thresh_b

def compute_l2_distance(model, stimulus, scaled_perturbation_a, scaled_perturbation_b):
    model_response_a = model(stimulus + scaled_perturbation_a)
    model_response_b = model(stimulus + scaled_perturbation_b)
    model_response_orig = model(stimulus)
    response_norm_a = torch.linalg.norm(model_response_orig - model_response_a)
    response_norm_b = torch.linalg.norm(model_response_orig - model_response_b)
    return response_norm_a, response_norm_b

def optimal_perturbations(models,
                          stimulus,
                          learning_rate=1e-3,
                          threshhold=None,
                          max_step=1000,
                          print_step=100,
                          alpha_mask=None,
                          objective_cutoff=None,
                          perturb_clamp_scale=0.001,
                          perturbation_a=None,
                          perturbation_b=None,
                          stim_max_value = 1.,
                          stim_min_value = 0.,
                          custom_perturbation_processing = None,
                          show_plots=True,
                          keep_best = False,
                          tensor_norm=1.):
    """Compute the perturbation of an stimulus that maximizes the variance of the projected Fisher information matrices"""

    if threshhold is not None:
        warnings.warn('Threshhold input is depricated.')
 
    if type(learning_rate)==float:
        learning_rate = torch.ones(max_step) * learning_rate

    if perturbation_a is None:
        perturbation_a = torch.randn(stimulus.shape, device=stimulus.device)
    if perturbation_b is None:
        perturbation_b = torch.randn(stimulus.shape, device=stimulus.device)

    if alpha_mask is not None: # Initial weighting 
        perturbation_a = alpha_mask * perturbation_a
        perturbation_b = alpha_mask * perturbation_b

    perturbation_a /= torch.linalg.norm(perturbation_a)
    perturbation_b /= torch.linalg.norm(perturbation_b)

    perturbation_a *= tensor_norm
    perturbation_b *= tensor_norm
    
    update_size = 1
    step = 0
    objective = 0

    objective_hist = []

    max_perturb_val = stim_max_value-stimulus
    min_perturb_val = stim_min_value-stimulus

    if keep_best:
        best_objective = objective
        best_perturb_a = perturbation_a
        best_perturb_b = perturbation_b

    for step in range(max_step):
        update_a, update_b, objective = perturbation_update(models, stimulus, perturbation_a, perturbation_b)

        update_size_a = torch.linalg.norm(update_a)
        update_size_b = torch.linalg.norm(update_b)
        update_size = (update_size_a + update_size_b) / 2
        update_a = torch.clamp(update_a, -1, 1)
        update_b = torch.clamp(update_b, -1, 1)

        if alpha_mask is not None: # Weight each update
            update_a = alpha_mask * update_a
            update_b = alpha_mask * update_b

        if update_size > tensor_norm:
            update_a /= update_size_a
            update_b /= update_size_b
            update_a *= tensor_norm
            update_b *= tensor_norm

        perturbation_a += learning_rate[step]*update_a
        perturbation_b += learning_rate[step]*update_b

        perturbation_a /= torch.linalg.norm(perturbation_a)
        perturbation_b /= torch.linalg.norm(perturbation_b)

        perturbation_a *= tensor_norm
        perturbation_b *= tensor_norm

        # Note: We want each pixel of the perturbation to be small enough so that when we scale it we stay
        # within the range of valid images. 
        if perturb_clamp_scale is not None:
            perturbation_a = torch.clamp(perturbation_a, perturb_clamp_scale*min_perturb_val, perturb_clamp_scale*max_perturb_val)
            perturbation_b = torch.clamp(perturbation_b, perturb_clamp_scale*min_perturb_val, perturb_clamp_scale*max_perturb_val)

        if custom_perturbation_processing is not None:
            perturbation_a = custom_perturbation_processing(perturbation_a)
            perturbation_b = custom_perturbation_processing(perturbation_b)

        objective_after_processing = compute_diff_objective(models, stimulus, perturbation_a, perturbation_b)

        if keep_best and (best_objective<objective_after_processing):
            best_objective = objective_after_processing.clone()
            best_perturb_a = perturbation_a.clone()
            best_perturb_b = perturbation_b.clone()
            objective_hist.append(objective_after_processing)
        elif keep_best and (best_objective>=objective_after_processing):
            objective_hist.append(best_objective)
        else:
            objective_hist.append(objective_after_processing)
   
        if (step%print_step==0) or (step==(max_step-1)):
            print(f'step {step}, size of update: {update_size_a}, {update_size_b}, objective: {objective_after_processing}')
            if (perturbation_a.shape[1] == 1) & show_plots:
                plt.imshow(perturbation_a[0,0].to('cpu'), cmap='gray')
                plt.show()
                plt.imshow(perturbation_b[0,0].to('cpu'), cmap='gray')
                plt.show()
            elif show_plots:
                plt.imshow(0.5 + torch.moveaxis(perturbation_a, -3, -1)[0].to('cpu'))
                plt.show()
                plt.imshow(0.5 + torch.moveaxis(perturbation_b, -3, -1)[0].to('cpu'))
                plt.show()

        # If things are sufficiently optimized cut off the optimization early
        if objective_cutoff is not None:
            if objective_after_processing >= objective_cutoff:
                break
 
    if objective_cutoff is not None:
        if objective < objective_cutoff:
            warnings.warn('The number of iterations reached the maximum number of '
                          'iterations without reaching specified cutoff. Consider increasing the '
                          'number of optimization steps or changing the learning rate.')

    if keep_best:
        perturbation_a = best_perturb_a
        perturbation_b = best_perturb_b

    return perturbation_a, perturbation_b, objective_hist

