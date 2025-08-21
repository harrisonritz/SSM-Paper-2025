# import

import os
import sys
from datetime import datetime
import numpy as np
import scipy as sp
import time

import torch
from torch import nn

import siegelmiller_task as get_task
from task_model import RNN

import gc



print("\n=====================\nPREPARING THE FIT\n=====================\n")

print(f"pytorch version: {torch.__version__}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
print(f"python version: {sys.version}")


# settings ==============================================================================================================
is_local = sys.platform.startswith('darwin')
print(f"Is local? {is_local}")

if is_local == True:
    import matplotlib.pyplot as plt
    save_fld = "/Users/hr0283/Dropbox (Brown)/HallM_NeSS/RNN_TS/task-optimized"
else:
    save_fld = "/scratch/gpfs/hr0283/HallM_NeSS/RNN_TS/task-optimized"
print(f"Save folder: {save_fld}")   



# PROCESS ARGUMENTS ==============================================================================================================

print("\n\nSETUP FIT ======================\n")
print("ARGS: ", sys.argv)


# ARG 1: which fit to run ==================================
if len(sys.argv) > 1:
    which_fit = sys.argv[1]
else:
    which_fit = "GRU"

print(f"Fit: {which_fit}")


# ARG 2: which task to run ==================================
if len(sys.argv) > 2:
    which_task = sys.argv[2]
else:
    which_task = "iti-30"

print(f"Task: {which_task}")


# ARG 3: which seed =========================================
if len(sys.argv) > 3:

    if is_local:
        n_fits = 1
    else:
        n_fits = 32

    min_fit = int(sys.argv[3])
    min_fit = n_fits*(min_fit-1) + 1

else:
    min_fit         = 1
    n_fits          = 256


fit_list = range(min_fit, min_fit + n_fits)
print(f"Fit list: {fit_list}")





# defaults parameters ==============================================================================================================
net_name    = "GRU"

save_fit        = False
do_test         = True


n_targets       = 2
n_layers        = 1
mask_type       = "trial"

# single task
n_single        = 0

# switch task
n_switch        = 100


save_outputs_local    = True
print("Save outputs: ", save_outputs_local)



global device
device = None
if is_local:
    device = torch.device("mps")
else:
    device = torch.device("cuda:0")
print(f"Device: {device}")


# parse parameters ==============================================================================================================
nreps        = 256
nsingle     = 0
nswitch     = 100
winit       = "xavier"
ntrials     = 2
noise       = 100
fit_init    = 0

# net name
print("which_fit:", which_fit)
split_fit = which_fit.split('_')

# net defaults
net_name = split_fit[0]
if net_name == "RNN":
    nswitch = 1000 # default switch

# read in network parameters
print("which_task:", which_task)
params = {}
for pair in split_fit[1:]:
    key, value = pair.split('-')
    params[key] = value
    print(key, value)
    

nreps = int(params.get('nreps', nreps))
nsingle = int(params.get('nsingle', nsingle))
nswitch = int(params.get('nswitch', nswitch))
winit = params.get('winit', winit)
ntrials = int(params.get('ntrials', ntrials))
noise = float(params.get('noise', noise))
fit_init = bool(params.get('fitinit', fit_init))

# rename
n_reps = nreps
n_switch = nswitch
n_single = nsingle

print("NET PARAMS --- net_name:", net_name, 
      "|| n_reps:", n_reps, 
      "|| n_single", n_single, 
      "|| n_switch:", n_switch,  
      "|| weight init:", winit,
      "|| n_trials:", ntrials,
      "|| noise:", noise)


# PARSE NETWORK  ==============================================================================================================
if net_name == "GRU":
    
    hidden_size     = 108
    lr              = .01
    weight_decay    = .01

elif net_name == "GRU-no-reset":
    
    hidden_size     = 134
    lr              = .01
    weight_decay    = .01

elif net_name == "GRU-no-update":
    
    hidden_size     = 134
    lr              = .01
    weight_decay    = .01

elif net_name == "RNN":

    hidden_size     = 190
    lr              = .001
    weight_decay    = .01

else:
    print("Unknown model")
    sys.exit()


# create folder/save name
fld_name = "NET-" + which_fit + "__TASK-" + which_task



# generate inputs ==============================================================================================================
inputs, targets, mask, conditions, _, _, _, _ = get_task.generate_trials(n_reps,
                                                                         which_task=which_task,
                                                                         n_trials=2)


inputs_single_np = inputs[:,conditions['trial_sel0']==1,:].clone()
targets_single_np = targets[:,conditions['trial_sel0']==1,:].clone()
mask_single_np = mask[:,conditions['trial_sel0']==1,:].clone()

inputs_full_np = inputs.clone()
targets_full_np = targets.clone()
mask_full_np = mask.clone()



# generate test/sim inputs ==============================================================================================================
clean_sim = False
if ntrials == 2:
    n_sims = 32
    n_sim_step = 128 # number of repetitions of noisy_step_loss
else:
    n_sims = 4
    n_sim_step = 16 # number of repetitions of noisy_step_loss
print(f"\n\nNumber of sims: {n_sims}")


# generate the sim inputs
sim_inputs, _, _, sim_conditions, input_set, target_set, mask_set, condition_set = get_task.generate_trials(n_sims,
                                                                                      which_task=which_task+'_bias-0_isTest-1',
                                                                                      n_trials=ntrials)



print("\n=====================\nTASK INFO\n=====================\n")
print("fld_name:", fld_name)
print("Size of inputs:", inputs.shape)
print("Size of targets:", targets.shape)
print("Size of mask:", mask.shape)

print("Size of inputs set:", input_set.shape)
print("Size of targets set:", target_set.shape)
print("Size of mask set:", mask_set.shape)

print("mask type:", mask_type)

print("n single epochs:", n_single)
print("n switch epochs:", n_switch)
print("hiden size:", hidden_size)
print("learning rate:", lr)
print("weight decay:", weight_decay)

print("clean sim:", clean_sim)
print("n sims:", n_sims)

print('Sim inputs shape:', sim_inputs.shape)

print("switch train pct: ", np.mean(conditions['switch1']))
print("switch sim pct: ", np.mean(sim_conditions['switch1']))

# if is_local: 
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(inputs[42,:,:], label='')
#     plt.plot(mask[42,:,:], label='', linestyle='--', color='k')
#     plt.plot(conditions['trial_sel1'], label='', linestyle=':', color='k')
#     plt.plot(conditions['ssm_epoch1'], label='', linestyle=':', color='r')
#     plt.xlabel('Time Step')
#     plt.ylabel('Input Value')
#     plt.title('inputs train')

#     plt.subplot(1, 2, 2)
#     plt.plot(input_set[2,:,:], label='')
#     plt.plot(mask_set[2,:,:], label='', linestyle='--', color='k')
#     plt.plot(condition_set['ssm_epoch1'], label='', linestyle=':', color='r')
#     if ntrials == 3:
#         plt.plot(condition_set['ssm_epoch2'], label='', linestyle=':', color='r')
#     plt.xlabel('Time Step')
#     plt.ylabel('Input Value')
#     plt.title('inputs test')

#     plt.tight_layout()
#     plt.show()




## Fit model to task ==============================================================================================================
for rr in fit_list:

    print("\n\n\n==========================================\n")

    # reset name and seed
    save_name = fld_name + "__" + str(rr)
    print(f"Fit {rr} of {rr + n_fits - 1}: {save_name}\n")
    _ = torch.manual_seed(rr)


    print("==========================================\\n")



    # setup network ==============================================================================================================
    rnn = RNN(input_size=inputs.size(-1), hidden_size=hidden_size, output_size=n_targets,
            net_name=net_name, fit_name=save_name, winit=winit, fit_init=fit_init,
            device=device)

    rnn.input_sigma = noise/100

    print(rnn)

    n_param = rnn.count_parameters()
    print(f"Number of parameters: {n_param}")


    # send the model to the device
    rnn = rnn.to(device)
    rnn.compile()

    input_set = input_set.to(device)
    target_set = target_set.to(device)
    mask_set = mask_set.to(device)

    # run the fit ==============================================================================================================
    start_time = time.time()
  
    noisy_loss = np.zeros(n_single + n_switch)
    clean_loss = np.zeros(n_single + n_switch)

    print("\nFitting the RNN at ", datetime.now().strftime("%H:%M:%S"), "...\n")







    # single ============================================================================================================== single
    print('fitting single task...')

    # add to device
    inputs_single = inputs_single_np.to(device)
    targets_single = targets_single_np.to(device)
    mask_single = mask_single_np.to(device)

    # fit
    noisy_loss_single, clean_loss_single = rnn.fit(
                                    inputs=inputs_single, 
                                    targets=targets_single, 
                                    mask=mask_single, 
                                    n_epochs=n_single, 
                                    lr=lr, weight_decay=weight_decay,
                                    verbose=True,
                                    do_test=do_test, 
                                    input_test=input_set, target_test=target_set, mask_test=mask_set)
    

    # clean memory
    del inputs_single, targets_single, mask_single
    gc.collect()
    # ==============================================================================================================




    # switch ============================================================================================================== switch
    print('fitting switching task...')

    # add to device
    inputs_full = inputs_full_np.to(device)
    targets_full = targets_full_np.to(device)
    mask_full = mask_full_np.to(device)

    # fit
    noisy_loss_switch, clean_loss_switch = rnn.fit(
                                    inputs=inputs_full, 
                                    targets=targets_full, 
                                    mask=mask_full, 
                                    n_epochs=n_switch, 
                                    lr=lr, weight_decay=weight_decay,
                                    verbose=True,
                                    do_test=do_test, 
                                    input_test=input_set, target_test=target_set, mask_test=mask_set)
    
    # clean memory
    del inputs_full, targets_full, mask_full
    gc.collect()
    # ==============================================================================================================








    # combine losses
    noisy_loss = np.concatenate((noisy_loss_single, noisy_loss_switch))
    clean_loss = np.concatenate((clean_loss_single, clean_loss_switch))

    
    elapsed_time = time.time() - start_time

    print(f"\nElapsed time: {elapsed_time:.2f} seconds\n\n")
    # if is_local:
        # os.system('say -v "Daniel" [[volm 0.5]] pytorch has fit in ' + str(int(elapsed_time)) + ' seconds')



    # get loss ==============================================================================================================
    loss_function_step = nn.BCEWithLogitsLoss(reduction='none', weight=mask_set)

    with torch.no_grad():


        # clean step loss
        outputs_clean,_ = rnn.forward(input_set, 
                                      rnn.init_hidden(input_set.shape[0], do_zeros=True).to(rnn.device))
        clean_loss_step = loss_function_step(outputs_clean, target_set).cpu().numpy()
        print('clean loss step: ',clean_loss_step.mean())


        # (average) noisy step loss
        noisy_loss_step = np.zeros(clean_loss_step.shape)
        outputs_noisy = torch.zeros_like(input_set)
        for _ in range(n_sim_step):
            outputs_noisy,_ = rnn.forward(input_set + torch.randn_like(input_set)*rnn.input_sigma, 
                                         rnn.init_hidden(input_set.shape[0]).to(rnn.device))
            noisy_loss_step += loss_function_step(outputs_noisy, target_set).cpu().numpy()
        noisy_loss_step /= n_sim_step
        print('noisy loss step: ',noisy_loss_step.mean())



    # Plot the noisy and clean loss
    if is_local: 
        smooth_noisy_loss = sp.ndimage.gaussian_filter1d(noisy_loss, sigma=10)
        smooth_clean_loss = sp.ndimage.gaussian_filter1d(clean_loss, sigma=10)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(smooth_noisy_loss, label='Noisy Loss')
        plt.plot(smooth_clean_loss, label='Clean Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Noisy and Clean Loss for Single Task')
        plt.legend()


        step_np = np.array(clean_loss_step);

        plt.subplot(1, 3, 2)
        plt.plot(np.mean(np.log(step_np), axis=(0,2)))
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Clean Loss Step')


        switch_np = np.array(condition_set['switch1'])

        plt.subplot(1, 3, 3)
        plt.plot(np.mean(np.log(step_np[switch_np == False,:,:]) - np.log(step_np[switch_np == True, :, :]), axis=(0, 2)), 
                 color='black', linewidth=2)
        
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Clean Loss Step (Switch1 = 0)')


        plt.tight_layout()
        plt.show()


         # Plot the outputs and targets for an example trial
        example_trial = 50  # Choose the index of the trial to plot

        plt.figure(figsize=(10, 5))
        plt.plot(torch.sigmoid(outputs_clean[example_trial,:,0]).cpu().numpy(), color='magenta', linestyle='--')
        plt.plot(torch.sigmoid(outputs_clean[example_trial,:,1]).cpu().numpy(), color='red', linestyle='--')
        plt.plot(torch.sigmoid(outputs_noisy[example_trial,:,0]).cpu().numpy(), color='magenta', linestyle=':')
        plt.plot(torch.sigmoid(outputs_noisy[example_trial,:,1]).cpu().numpy(), color='red', linestyle=':')
        plt.plot(target_set[example_trial,:,0].cpu().numpy(), color='cyan', linestyle='-')
        plt.plot(target_set[example_trial,:,1].cpu().numpy(), color='blue', linestyle='-')
        plt.plot(np.sign(input_set[example_trial,:,3].cpu().numpy()), color='black', linestyle=':')
        plt.plot(np.sign(input_set[example_trial,:,1].cpu().numpy() + input_set[example_trial,:,2].cpu().numpy()), color='black', linestyle=':')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Outputs and Targets for Example Trial')
 
        

        plt.tight_layout()
        plt.show()

    
    

    # package for SSM  ==============================================================================================================
    print("\n=====================\nSimulating for SSM...\n")

    # Simulate from the RNN
    sim_inputs = sim_inputs.to(device)
    
    print("Size of repeated input set:", sim_inputs.shape)

    latents = torch.zeros(sim_inputs.shape[0], sim_inputs.shape[1], rnn.hidden_size)
    rnn = rnn.to(device).eval()
    with torch.no_grad():

        if clean_sim:
            h0 = rnn.init_hidden(sim_inputs.shape[0], do_zeros=clean_sim)
            inputs_fwd = sim_inputs
        else:
            h0 = rnn.init_hidden(sim_inputs.shape[0], do_zeros=clean_sim)
            inputs_fwd = sim_inputs + nn.init.normal_(torch.empty(sim_inputs.shape), std=rnn.input_sigma).to(rnn.device)

        outputs, latents = rnn.forward(inputs_fwd, h0)




    # save the simulation ==============================================================================================================
    
    
    # Save just the losses
    if not is_local:

        losses_to_save = {
            'condition_set': condition_set,
            'noisy_loss': noisy_loss,
            'clean_loss': clean_loss,
            'noisy_loss_step': noisy_loss_step,
            'clean_loss_step': clean_loss_step,
        }

        losses_folder = f'{save_fld}/results/ssm/{fld_name}_loss'
        if not os.path.exists(losses_folder):
            try:
                os.makedirs(losses_folder, exist_ok=True)
            except:
                print("Error creating folder")

        losses_filename = f'{losses_folder}/{save_name}.mat'

        print(f"Saving losses to: {losses_filename}")
        sp.io.savemat(losses_filename, losses_to_save, long_field_names=True)


    # save just the latents
    if not is_local:

        ssm_folder = f'{save_fld}/results/ssm/{fld_name}_ssm'
        if not os.path.exists(ssm_folder):
            try:
                os.makedirs(ssm_folder, exist_ok=True)
            except:
                print("Error creating folder")

        ssm_filename = f'{ssm_folder}/{save_name}.mat'
        print(f"Saving simulation to: {ssm_filename}")

        recur_weight = rnn.layer_recur.weight_hh_l0.detach().cpu().numpy()
        recur_bias = rnn.layer_recur.bias_hh_l0.detach().cpu().numpy()
        in_weight =  rnn.layer_recur.weight_ih_l0.detach().cpu().numpy()
        in_bias =  rnn.layer_recur.bias_ih_l0.detach().cpu().numpy()
        out_weight = rnn.layer_out.weight.detach().cpu().numpy()

        # layer_recur.weight_ih_l0
        # layer_recur.weight_hh_l0
        # layer_recur.bias_ih_l0
        # layer_recur.bias_hh_l0
        # layer_out.weight

        sp.io.savemat(ssm_filename, 
                {   'sim_latents': latents.cpu().numpy(),
                    'sim_inputs': sim_inputs.cpu().numpy(),
                    'out_weight': out_weight,
                    'recur_weight': recur_weight,
                    'recur_bias': recur_bias,
                    'in_weight': in_weight,
                    'in_bias': in_bias,
                    'sim_conditions': sim_conditions,
                    'condition_set': condition_set,
                    }, long_field_names=True, do_compression=True)
    


    # save locally, if requested
    # if is_local:
    #     if save_outputs_local:
    #         print("Saving simulation to: ", f'{save_fld}/results/ssm/{fld_name}/{save_name}.mat')
    #         sp.io.savemat(f'{save_fld}/results/ssm/{fld_name}/{save_name}.mat', 
    #                 {   
    #                     'sim_latents': latents.cpu().numpy(),
    #                     'sim_outputs': outputs.cpu().numpy(),
    #                     'sim_inputs': sim_inputs.cpu().numpy(),
    #                     'sim_conditions': sim_conditions,
    #                     'condition_set': condition_set,
    #                     'noisy_loss': noisy_loss,
    #                     'clean_loss': clean_loss,
    #                     'noisy_loss_step': noisy_loss_step,
    #                     'clean_loss_step': clean_loss_step,
    #                 }, long_field_names=True)



    # clean up ==============================================================================================================
    del rnn, outputs, latents
    gc.collect()
    if is_local:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()


    print(f"\n\n\nFit {rr} of {n_fits}: {save_name} complete\n")
    print("==========================================\\n")

