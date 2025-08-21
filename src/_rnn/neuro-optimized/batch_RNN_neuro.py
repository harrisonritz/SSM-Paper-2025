

# INIT ======================================================
import os
import sys
from datetime import datetime
import numpy as np
import scipy as sp
import scipy.io as sio

import torch
from torch import nn

from neural_model import RNN



# os.environ['PYTORCH_ENABLE_MPS_FALLBACK = 1']
print("\n\nSTARTING ======================\n")
print(f'date: {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}')
print(f"pytorch version: {torch.__version__}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
print(f"python version: {sys.version}")




#  SETUP ======================================================
model_type = "RNN"
record_fit = False
eeg_dataset = "A23"

hidden_size  = 134 # RNN: 134, GRU: ??
n_epochs     = 5000
n_repeats    = 1

pt_num = 2;
xdim = 112;
xdim_list = range(16,129,16)




lr           = .001
weight_decay = .01


print("\n\nSETUP FIT ======================\n")



# arguments from command line ======================================================
print("ARGS: ", sys.argv)

if len(sys.argv) > 1:
    pt_num = int(sys.argv[1])
print(f"\npt_num: {pt_num}")

if len(sys.argv) > 2:
    model_type = sys.argv[2]
print(f"model name: {model_type}")

if len(sys.argv) > 3:
    xdim = xdim_list[int(sys.argv[3])]
print(f"xdim: {xdim}\n")

if len(sys.argv) > 4:
    eeg_dataset = sys.argv[4]
print(f"eeg_dataset: {eeg_dataset}\n")



# Generate the filename
fit_name = f"{model_type}_{eeg_dataset}_neuro-sweep-5k"
filename = f"losses__{fit_name}_pt{pt_num}_xdim{xdim}.mat"
print(f"\nfit_name: {fit_name}")
print("filename:", filename)

    


# cluster parameters ======================================================
on_cluster = sys.platform.startswith('linux')
print(f"on_cluster: {on_cluster}")

if on_cluster:
    n_repeats = 32
else: 
    n_repeats = 1
print(f"n_repeat: {n_repeats}")

if on_cluster:
    root_dir = '/scratch/gpfs/hr0283/HallM_NeSS'
    if eeg_dataset == "A23":
        fit_path  = f'fit-results/EM-mat/2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask/2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask_Pt{pt_num}_xdim{xdim}.mat'
    elif eeg_dataset == "H19":
        fit_path  = f'fit-results/EM-mat/2024-10-31-10h_H19__prepTrial200-prevTask/2024-10-31-10h_H19__prepTrial200-prevTask_Pt{pt_num}_xdim{xdim}.mat'
    else: 
        raise ValueError(f"Unknown eeg_dataset: {eeg_dataset}")
else:
    root_dir = '/Users/hr0283/Dropbox (Brown)/HallM_NeSS'
    if eeg_dataset == "A23":
        fit_path  = f'della-outputs/2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask/2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask_Pt{pt_num}_xdim{xdim}.mat'
    elif eeg_dataset == "H19":
        fit_path  = f'della-outputs/2024-10-31-10h_H19__prepTrial200-prevTask/2024-10-31-10h_H19__prepTrial200-prevTask_Pt{pt_num}_xdim{xdim}.mat'
    else: 
        raise ValueError(f"Unknown eeg_dataset: {eeg_dataset}")

print('fit_path: ', fit_path)




if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")



# load fit ======================================================
fit_data = sp.io.loadmat(os.path.join(root_dir, fit_path))
print(fit_data.keys())


# load y
y_train = np.swapaxes(fit_data['dat']['y_train'][0][0], 0, -1)
y_test = np.swapaxes(fit_data['dat']['y_test'][0][0], 0, -1)

n_train, n_times, n_chans = y_train.shape
n_test, _, _ = y_test.shape

n_bases=fit_data['dat']['n_bases'][0][0][0][0]
n_u = (fit_data['dat']['u_train'][0][0].shape[0]//n_bases)-1
n_u0 = fit_data['dat']['u0_train'][0][0].shape[0]



u_train = np.swapaxes(fit_data['dat']['u_train'][0][0], 0, -1)
u_train = u_train[:,0,range(0,u_train.shape[2],n_bases)]
u_train = np.tile(u_train[:,np.newaxis,1:], (1, n_times, 1))

u0_train = np.zeros((n_train, n_times, n_u0))
u0_train[:,0,:] = np.swapaxes(fit_data['dat']['u0_train'][0][0], 0, -1)

u_test = np.swapaxes(fit_data['dat']['u_test'][0][0], 0, -1)
u_test = u_test[:,0,range(0,u_test.shape[2],n_bases)]
u_test = np.tile(u_test[:,np.newaxis,1:], (1, n_times, 1))

u0_test = np.zeros((n_test, n_times, n_u0))
u0_test[:,0,:] = np.swapaxes(fit_data['dat']['u0_test'][0][0], 0, -1)


inputs_train = torch.from_numpy(np.concatenate(
    [   u0_train,
        np.concatenate([np.zeros((n_train,1,n_chans)), y_train[:,range(0, n_times-1),:]], 1), 
        np.concatenate([np.zeros((n_train,1,n_u)), u_train[:,range(0, n_times-1),:]], 1)
     ], axis=-1)
     ).to(device, dtype=torch.float32)

targets_train = torch.from_numpy(y_train).to(device, dtype=torch.float32)


inputs_test = torch.from_numpy(np.concatenate(
    [   u0_test,
        np.concatenate([np.zeros((n_test,1,n_chans)), y_test[:,range(0, n_times-1),:]], 1), 
        np.concatenate([np.zeros((n_test,1,n_u)), u_test[:,range(0, n_times-1),:]], 1)
     ], axis=-1)
     ).to(device, dtype=torch.float32)

# inputs_test = torch.from_numpy(np.concatenate([y_test[:,range(0, n_times-1),:], u_test[:,range(0, n_times-1),:]], axis=-1)).to(device, dtype=torch.float32)
targets_test = torch.from_numpy(y_test).to(device, dtype=torch.float32)

input_size = inputs_train.shape[-1]
target_size = targets_train.shape[-1]

print("Size of input train:", inputs_train.shape)
print("Size of target train:", targets_train.shape)
print("Size of input test:", inputs_test.shape)
print("Size of target test:", targets_test.shape)

print("Input size:", input_size)
print("Target size:", target_size)

A = fit_data['mdl']['A'][0][0]
B = fit_data['mdl']['B'][0][0]
C = fit_data['mdl']['C'][0][0]
Q = fit_data['mdl']['Q'][0][0]['mat'][0][0]
R = fit_data['mdl']['R'][0][0]['mat'][0][0]
B0 = fit_data['mdl']['B0'][0][0]
P0 = fit_data['mdl']['P0'][0][0]['mat'][0][0]

print("Size of A:", A.shape)
print("Size of B:", B.shape)
print("Size of C:", C.shape)
print("Size of Q:", Q.shape)
print("Size of R:", R.shape)
print("Size of B0:", B0.shape)
print("Size of P0:", P0.shape)

total_param = A.size + B.size + C.size + Q.size + R.size + B0.size + P0.size

def calculate_rnn_params(input_size, hidden_size, target_size):

    if model_type == "RNN-untied":

        # hidden-input, target-hidden, hidden-hidden, [1-bias_hi, 2-bias_th, 3-bias_hh]
        rnn_size = (hidden_size * input_size) + (hidden_size * target_size) + (hidden_size * hidden_size) +  4*hidden_size 

    else:

        # hidden-input, hidden-hidden, [1-bias_hi, 2-bias_hh]
        rnn_size = (hidden_size * input_size) + (hidden_size * hidden_size) + 3*hidden_size 

    return rnn_size

def find_optimal_hidden_size(input_size, target_size, total_param):
    left, right = 1, 1000
    best_hidden_size = 1
    best_diff = float('inf')
    
    while left <= right:
        mid = (left + right) // 2
        rnn_params = calculate_rnn_params(input_size, mid, target_size)
        
        if rnn_params < total_param:
            left = mid + 1
        else:
            right = mid - 1
        
        diff = abs(rnn_params - total_param)
        if diff < best_diff:
            best_diff = diff
            best_hidden_size = mid
    
    return best_hidden_size

optimal_hidden_size = find_optimal_hidden_size(input_size,target_size, total_param)
print(f"Optimal hidden_size: {optimal_hidden_size}")
print(f"Number of parameters with optimal hidden_size: {calculate_rnn_params(input_size,optimal_hidden_size, target_size)}")
print(f"Total parameters in the original model: {total_param}")

# Update hidden_size
hidden_size = optimal_hidden_size

all_train_loss = []
all_test_loss = []
min_test_loss = np.inf


## Fit model ======================================================
for i in range(n_repeats):

    print(f"\n\n\nFitting repeat {i+1} of {n_repeats} ============\n")
    _ = torch.manual_seed(i)

    # SETUP RNN
    rnn = RNN(input_size=input_size, target_size=target_size, hidden_size=hidden_size,
            model_type=model_type, fit_name=fit_name, 
            device=device)

    rnn.to(device=device)


    n_param = rnn.count_parameters()
    print(f"Number of parameters: {n_param}")
    print("Device of inputs:", inputs_train.device)
    print("Device of targets:", targets_train.device)
    print("Device of rnn:", rnn.device)



    train_loss, test_loss = rnn.fit(inputs=inputs_train, targets=targets_train, n_epochs=n_epochs,
                                    inputs_test=inputs_test, targets_test=targets_test, 
                                    lr=lr, weight_decay=weight_decay,
                                    record_fit=record_fit, verbose=True,
                                    do_test=True)


    print('min test loss:', np.min(test_loss))

    all_train_loss.append(train_loss)
    all_test_loss.append(test_loss)


min_test_loss = np.min(all_test_loss)
print("min_test_loss:", min_test_loss)

# Save all_train_loss and all_test_loss to a .mat file


# Create a dictionary to hold the data
data_to_save = {
    'all_train_loss': np.array(all_train_loss, dtype=object).reshape(1, -1),
    'all_test_loss': np.array(all_test_loss, dtype=object).reshape(1, -1),
    'min_test_loss': min_test_loss
}


# Ensure the directory exists
save_dir = os.path.join(root_dir, 'model_outputs', fit_name)
try:
    os.makedirs(save_dir, exist_ok=True)
except:
    print("Error creating folder")

# Full path for the file
full_path = os.path.join(save_dir, filename)
print("full_path:", full_path)

# Save the data
sio.savemat(full_path, data_to_save)

print(f"Losses saved to: {full_path}")










# %%
