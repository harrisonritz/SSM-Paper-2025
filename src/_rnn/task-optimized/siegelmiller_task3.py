#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:00:09 2020

@author: langdon

Functions for generating trial data.
"""

import numpy as np
from scipy.sparse import random
import torch

from scipy import stats
import scipy.ndimage

import copy


# device = torch.device("mps")


def generate_input_target_stream3(context, motion_coh, color_coh, size_coh, n_steps, cue_on, cue_off,
                                 stim_on, stim_off, dec_on, dec_off, n_targets=1, cue_gain=100.0):
    """
    Generate input and target sequence for a given set of trial conditions.

    :return: input stream
    :return: target stream
    """
    # Convert trial events to discrete time

    # Transform coherence to signal
    motion_r = (1 + motion_coh) / 2
    motion_l = 1 - motion_r
    color_r = (1 + color_coh) / 2
    color_l = 1 - color_r
    size_r = (1 + size_coh) / 2
    size_l = 1 - size_r

    # Cue input stream
    cue_input = np.zeros([n_steps, 9])
    if context == "motion":
        cue_input[cue_on:cue_off, 0] = (cue_gain/100)*np.ones([cue_off - cue_on, 1]).squeeze()
    elif context == "color":
        cue_input[cue_on:cue_off, 1] = (cue_gain/100)*np.ones([cue_off - cue_on, 1]).squeeze()
    else:  # size
        cue_input[cue_on:cue_off, 2] = (cue_gain/100)*np.ones([cue_off - cue_on, 1]).squeeze()

    # Motion input stream
    motion_input = np.zeros([n_steps, 9])
    motion_input[stim_on:stim_off, 3] = motion_r * np.ones([stim_off - stim_on])
    motion_input[stim_on:stim_off, 4] = motion_l * np.ones([stim_off - stim_on])

    # Color input stream
    color_input = np.zeros([n_steps, 9])
    color_input[stim_on:stim_off, 5] = color_r * np.ones([stim_off - stim_on])
    color_input[stim_on:stim_off, 6] = color_l * np.ones([stim_off - stim_on])

    # Size input stream
    size_input = np.zeros([n_steps, 9])
    size_input[stim_on:stim_off, 7] = size_r * np.ones([stim_off - stim_on])
    size_input[stim_on:stim_off, 8] = size_l * np.ones([stim_off - stim_on])

    # Input stream is rectified sum of baseline, task and noise signals.
    input_stream = np.maximum(cue_input + motion_input + color_input + size_input, 0)

    # Target stream
    if n_targets == 2:
        target_stream = 0.5 * np.ones([n_steps, 2])
        if (context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0) or (context == "size" and size_coh > 0):
            target_stream[dec_on:dec_off, 0] = np.ones([dec_off - dec_on, 1]).squeeze()
            target_stream[dec_on:dec_off, 1] = np.zeros([dec_off - dec_on, 1]).squeeze()
        else:
            target_stream[dec_on:dec_off, 0] = np.zeros([dec_off - dec_on, 1]).squeeze()
            target_stream[dec_on:dec_off, 1] = np.ones([dec_off - dec_on, 1]).squeeze()
    else:
        target_stream = 0.5 * np.ones([n_steps, 1])
        if (context == "motion" and motion_coh > 0) or (context == "color" and color_coh > 0) or (context == "size" and size_coh > 0):
            target_stream[dec_on:dec_off, ] = np.ones([dec_off - dec_on, 1])
        else:
            target_stream[dec_on:dec_off, ] = np.zeros([dec_off - dec_on, 1])

    return input_stream, target_stream







def generate_trials3( n_reps,
                     which_task = "iti-30",
                     n_trials = 2):
    """
    Create a set of trials consisting of inputs, targets and trial conditions.

    :param tau:
    :param trial_events:
    :param n_trials: number of trials per sequence
    :param alpha:
    :param sigma_in:
    :param baseline:
    :param n_coh:

    :return: dataset
    :return: mask
    :return: conditions: array of dict
    """

    #cohs = np.hstack((-10 ** np.linspace(0, -2, n_coh), 10 ** np.linspace(-2, 0, n_coh)))

    # task-specific parameters    
    print("\n\nTASK GEN ======================\n")
    iti = 10
    mot = 15
    col = 15
    bias = 0
    cuegain = 100
    
    print("which_task:", which_task)
    params = {}
    for pair in which_task.split('_'):
        key, value = pair.split('-')
        params[key] = value
        print(key, value)

    
    iti = int(params.get('iti', iti))
    mot = int(params.get('mot', mot))
    col = int(params.get('col', col))
    bias = int(params.get('bias', bias))
    cuegain = int(params.get('cuegain', cuegain))

    print("TASK PARAMS --- ITI:", iti, "|| MOT:", mot, "|| COL:", col, "|| BIAS:", bias, '|| CUEGAIN:', cuegain)


        
    # Core Parameters
    n_targets = 2
    n_inputs = 9

    motion_cohs = [-int(mot)/100, int(mot)/100]
    color_cohs = [-int(col)/100, int(col)/100]
    size_cohs = [-int(col)/100, int(col)/100]  # Assuming size uses the same coherence as color

    motion_len = len(motion_cohs)
    color_len = len(color_cohs)
    size_len = len(size_cohs)

    print("motion cohs:", motion_cohs, "|| color cohs:", color_cohs, "|| size cohs:", size_cohs)
    print("motion_len:", motion_len, "|| color_len:", color_len, "|| size_len:", size_len)

    # first trial
    iti_dur     = [0, int(iti), int(iti)]
    cue_dur     = [10, 10, 10]
    isi_dur     = [20, 20, 20]
    wait_dur    = [10, 10, 10]
    stim_dur    = [40, 40, 40]

    print('iti_durs:', iti_dur)
    

    # first 
    tot_dur = np.zeros(n_trials, dtype=int)
    cue_on = np.zeros(n_trials, dtype=int)
    cue_off = np.zeros(n_trials, dtype=int)
    stim_on = np.zeros(n_trials, dtype=int)
    stim_off = np.zeros(n_trials, dtype=int)
    dec_on = np.zeros(n_trials, dtype=int)
    dec_off = np.zeros(n_trials, dtype=int)
    for dd in range(n_trials):
        tot_dur[dd] = iti_dur[dd] + cue_dur[dd] + isi_dur[dd] + stim_dur[dd]
        cue_on[dd] = iti_dur[dd]
        cue_off[dd] = iti_dur[dd] + cue_dur[dd]

        stim_on[dd] = iti_dur[dd] + cue_dur[dd] + isi_dur[dd]
        stim_off[dd] = tot_dur[dd]

        dec_on[dd] = iti_dur[dd] + cue_dur[dd] + isi_dur[dd] + wait_dur[dd]
        dec_off[dd] = tot_dur[dd]


    # create condition list
    inputs_list = []
    targets_list = []
    conditions = dict()
    for trial in range(n_trials):
        conditions.update({
                    f'context{trial}': [], 
                    f'motion_context{trial}': [],
                    f'color_context{trial}': [],
                    f'size_context{trial}': [],
                    f'motion_coh{trial}': [], 
                    f'color_coh{trial}': [], 
                    f'size_coh{trial}': [], 
                    f'correct_choice{trial}': [],
                    f'switch{trial}': []
                    })
        
        conditions.update({
            f'events_on{trial}': [
                            cue_on[trial], cue_off[trial], 
                            stim_on[trial], stim_off[trial], 
                            dec_on[trial], dec_off[trial]]
            })
        
        trial_sel = np.zeros(sum(tot_dur))
        if trial == 0:
            trial_sel[0:tot_dur[0]] = 1
        elif trial == 1:
            trial_sel[(tot_dur[0]):sum(tot_dur[0:2])] = 1
        elif trial == 2:
            trial_sel[(sum(tot_dur[0:2])):sum(tot_dur[0:3])] = 1
        conditions.update({
            f'trial_sel{trial}': trial_sel
            })
        

        


    # set tasks
    tasks = ["motion", "color", "size"]
    task_list = []
    switch_list = []

    if n_trials == 2:
        for task1 in tasks:
            for task2 in tasks:
                task_list.append([task1, task2])
                switch_list.append([-1, task1 != task2])
    elif n_trials == 3:
        for task1 in tasks:
            for task2 in tasks:
                for task3 in tasks:
                    task_list.append([task1, task2, task3])
                    switch_list.append([-1, task1 != task2, task2 != task3])
    else:
        print("Invalid value for n_trials. Please choose 2 or 3.")

    # set motion coherence
    motion_list = []
    if n_trials == 2:
        for motion1 in motion_cohs:
            for motion2 in motion_cohs:
                motion_list.append([motion1, motion2])
    elif n_trials == 3:
        for motion1 in motion_cohs:
            for motion2 in motion_cohs:
                for motion3 in motion_cohs:
                    motion_list.append([motion1, motion2, motion3])
    else:
        print("Invalid value for n_trials. Please choose 2 or 3.")

    # set color coherence
    color_list = []
    if n_trials == 2:
        for color1 in color_cohs:
            for color2 in color_cohs:
                color_list.append([color1, color2])
    elif n_trials == 3:
        for color1 in color_cohs:
            for color2 in color_cohs:
                for color3 in color_cohs:
                    color_list.append([color1, color2, color3])
    else:
        print("Invalid value for n_trials. Please choose 2 or 3.")

    # set size coherence
    size_list = []
    if n_trials == 2:
        for size1 in size_cohs:
            for size2 in size_cohs:
                size_list.append([size1, size2])
    elif n_trials == 3:
        for size1 in size_cohs:
            for size2 in size_cohs:
                for size3 in size_cohs:
                    size_list.append([size1, size2, size3])
    else:
        print("Invalid value for n_trials. Please choose 2 or 3.")
    

        
    print("task_list:", task_list)
    print("motion_list:", motion_list)
    print("color_list:", color_list)
    print("size_list:", size_list)

    # simulate data
    print("n_reps:", n_reps)
    for rep in range(n_reps):
        for task_n in range(len(task_list)):
            for motions in motion_list:
                for colors in color_list:
                    for sizes in size_list:
                        input_trial = np.zeros((0, n_inputs))
                        targets_trial = np.zeros((0, n_targets))
                        
                        for trial in range(n_trials):
                            cur_task = task_list[task_n][trial]
                            cur_switch = switch_list[task_n][trial]

                            if (bias == 1) & (trial == 1) & (rep >= (n_reps/2)):
                                cur_task = task_list[task_n][trial-1]
                                cur_switch = False

                            if (bias == 2) & (trial == 1) & (rep >= (n_reps/2)):
                                if task_list[task_n][trial-1] == "motion":
                                    cur_task = "color"
                                elif task_list[task_n][trial-1] == "color":
                                    cur_task = "size"
                                elif task_list[task_n][trial-1] == "size":
                                    cur_task = "motion"
                                cur_switch = True

                            cur_motion = motions[trial]
                            cur_color = colors[trial]
                            cur_size = sizes[trial]
                            cur_correct_choice = 1 if ((cur_task == "motion" and cur_motion > 0) or 
                                                       (cur_task == "color" and cur_color > 0) or 
                                                       (cur_task == "size" and cur_size > 0)) else -1

                            conditions[f'context{trial}'].append(cur_task)
                            conditions[f'motion_context{trial}'].append(cur_task == "motion")
                            conditions[f'color_context{trial}'].append(cur_task == "color")
                            conditions[f'size_context{trial}'].append(cur_task == "size")
                            conditions[f'motion_coh{trial}'].append(cur_motion)
                            conditions[f'color_coh{trial}'].append(cur_color)
                            conditions[f'size_coh{trial}'].append(cur_size)
                            conditions[f'correct_choice{trial}'].append(cur_correct_choice)
                            conditions[f'switch{trial}'].append(cur_switch)

                            input_stream, target_stream = generate_input_target_stream3(cur_task,
                                                                                        cur_motion,
                                                                                        cur_color,
                                                                                        cur_size,
                                                                                        tot_dur[trial],
                                                                                        cue_on[trial],
                                                                                        cue_off[trial],
                                                                                        stim_on[trial],
                                                                                        stim_off[trial],
                                                                                        dec_on[trial],
                                                                                        dec_off[trial],
                                                                                        n_targets=n_targets,
                                                                                        cue_gain=cuegain)
                            
                            input_trial = np.concatenate((input_trial, input_stream), axis=0)
                            targets_trial = np.concatenate((targets_trial, target_stream), axis=0)

                        inputs_list.append(input_trial)
                        targets_list.append(targets_trial)
        
        if rep == 0:
            condition_set = copy.deepcopy(conditions)

    

    # n_cond = task_len * motion_len * color_len * motion_len * color_len 
    n_cond = len(inputs_list)//n_reps
    print("n cond:", n_cond, "|| trial per cond:", n_reps, '|| total trials:', len(inputs_list))


    # conditions  (inputs, targets, mask, ssm parameters)
    inputs = torch.tensor(np.stack(inputs_list, 0)).float()
    targets = torch.tensor(np.stack(targets_list, 0)).float()

    mask = torch.zeros_like(targets)
    mask[:, (dec_on[0]):(dec_off[0]),:] = 1
    mask[:, (tot_dur[0] + dec_on[1]):(tot_dur[0] + dec_off[1]),:] = 1

    # ssm1
    ssm_epoch0 = np.zeros(mask.shape[1])
    ssm_epoch0[(cue_on[1]):(dec_on[1])] = 1
    conditions.update({'ssm_epoch0': ssm_epoch0})
    condition_set.update({'ssm_epoch0': ssm_epoch0})

    # ssm2
    ssm_epoch1 = np.zeros(mask.shape[1])
    ssm_epoch1[(tot_dur[0]+cue_on[1]):(tot_dur[0]+dec_on[1])] = 1
    conditions.update({'ssm_epoch1': ssm_epoch1})
    condition_set.update({'ssm_epoch1': ssm_epoch1})
    
    if n_trials == 3:
        mask[:, (tot_dur[0] + tot_dur[1] + dec_on[2]):(tot_dur[0] + tot_dur[1] + dec_off[2]),:] = 1

        # trial 3 select
        ssm_epoch2 = np.zeros(mask.shape[1])
        ssm_epoch2[(tot_dur[0]+tot_dur[1]+cue_on[2]):(tot_dur[0]+tot_dur[1]+dec_on[2])] = 1
        conditions.update({'ssm_epoch2': ssm_epoch2})
        condition_set.update({'ssm_epoch2': ssm_epoch2})


       
        
    # SSM conditions
    ssm_switch1 = np.where(conditions['switch1'], 1, -1)
    ssm_task1 = np.where(conditions['motion_context1'], 1, -1)
    ssm_taskSwitch1 = ssm_task1 * (ssm_switch1==1)
    ssm_taskRepeat1 = ssm_task1 * (ssm_switch1==-1)

    conditions.update({
        'ssm_switch1': ssm_switch1,
        'ssm_task1': ssm_task1,
        'ssm_taskSwitch1': ssm_taskSwitch1,
        'ssm_taskRepeat1': ssm_taskRepeat1,
        })

    if n_trials == 3:

        ssm_switch2 = np.where(conditions['switch2'], 1, -1)
        ssm_task2 = np.where(conditions['motion_context2'], 1, -1)
        ssm_taskSwitch2 = ssm_task2 * (ssm_switch2==1)
        ssm_taskRepeat2 = ssm_task2 * (ssm_switch2==-1)

        conditions.update({
            'ssm_switch2': ssm_switch2,
            'ssm_task2': ssm_task2,
            'ssm_taskSwitch2': ssm_taskSwitch2,
            'ssm_taskRepeat2': ssm_taskRepeat2,
            })
    
    # SSM condition set
    ssm_switch1 = np.where(condition_set['switch1'], 1, -1)
    ssm_task1 = np.where(condition_set['motion_context1'], 1, -1)
    ssm_taskSwitch1 = ssm_task1 * (ssm_switch1==1)
    ssm_taskRepeat1 = ssm_task1 * (ssm_switch1==-1)

    condition_set.update({
            'ssm_switch1': ssm_switch1,
            'ssm_task1': ssm_task1,
            'ssm_taskSwitch1': ssm_taskSwitch1,
            'ssm_taskRepeat1': ssm_taskRepeat1,
            })

    if n_trials == 3:

        ssm_switch2 = np.where(condition_set['switch2'], 1, -1)
        ssm_task2 = np.where(condition_set['motion_context2'], 1, -1)
        ssm_taskSwitch2 = ssm_task2 * (ssm_switch2==1)
        ssm_taskRepeat2 = ssm_task2 * (ssm_switch2==-1)

        condition_set.update({
            'ssm_switch2': ssm_switch2,
            'ssm_task2': ssm_task2,
            'ssm_taskSwitch2': ssm_taskSwitch2,
            'ssm_taskRepeat2': ssm_taskRepeat2,
            })
    
    
    # print('ssm_epoch:', len(condition_set['ssm_epoch']), sum(condition_set['ssm_epoch']), condition_set['ssm_epoch'])

    # unique conditions
    input_set = inputs[:n_cond,:,:].clone()
    target_set = targets[:n_cond,:,:].clone()
    mask_set = mask[:n_cond,:].clone()

    print("input_set:", input_set.shape, "|| target_set:", target_set.shape, "|| mask_set:", mask_set.shape, "|| condition_set:", len(condition_set['context0']))
    print("switch percent training:", np.mean(conditions['switch0']), np.mean(conditions['switch1']))
    print("switch percent set:", np.mean(condition_set['switch0']), np.mean(condition_set['switch1']))

    return inputs, targets, mask, conditions, input_set, target_set, mask_set, condition_set 




