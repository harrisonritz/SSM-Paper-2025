# model
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


# Parametrization Classes ==================================================

class param_MRU(nn.Module):
    def forward(self, W):
        w_xr, w_xz, w_xn = torch.split(W, W.shape[0] // 3)
        w_xf = torch.mean(torch.stack([w_xr, w_xz]), dim=0)
        return torch.cat([w_xf, -w_xf, w_xn])


class param_set_gates(nn.Module):

    def __init__(self, reset_weight=[1,0], update_weight=[1,0]):
        super().__init__()
        self.reset_weight = reset_weight
        self.update_weight = update_weight


    def forward(self, W):

        w_xr, w_xz, w_xn = torch.split(W, W.shape[0] // 3)

        w_xr = self.reset_weight[0]*w_xr + self.reset_weight[1]
        w_xz = self.update_weight[0]*w_xz + self.update_weight[1]

        return torch.cat([w_xr, w_xz, w_xn])


# ============================================================================



class RNN(nn.Module):

    # Module Initialization ==================================================
    def __init__(self, input_size=8, input0_size=1,  target_size=8, hidden_size=32, bias=False,  
                 model_type = "GRU", fit_name = "test",
                 device=torch.device("mps")):
        
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.target_size = target_size
        self.n_layers = 1
        self.input_sigma = 1.
        self.model_type = model_type
        self.fit_name = fit_name
        self.rnn_nonlin = 'relu'



        # shared parameters

        # encoding weights (inputs --> hidden)
        self.encode_weights = nn.Parameter(torch.empty(hidden_size, input_size, device=device, requires_grad=True, dtype=torch.float32))

        if self.model_type == "RNN-untied":
            self.decode_weights = nn.Parameter(torch.empty(target_size, hidden_size, device=device, requires_grad=True, dtype=torch.float32))


        # initial hidden state (input0 --> hidden)
        # self.init_weights = nn.Parameter(torch.empty(self.n_layers, input0_size, hidden_size, device=device, requires_grad=True,dtype=torch.float32))
        # nn.init.xavier_normal_(self.init_weights, gain=nn.init.calculate_gain(nonlinearity='linear'))

        if self.rnn_nonlin == "tanh":
            self.nonlin_F = nn.Tanh()
        elif self.rnn_nonlin == "relu":
            self.nonlin_F = nn.ReLU()
        else:
            raise ValueError("Nonlinearity not recognized")



        # basic RNN ==================================================
        if (self.model_type == "RNN") or (self.model_type == "RNN-untied"):

            print(f"=== fitting {self.model_type} ===")

            self.layer_recur = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                    nonlinearity=self.rnn_nonlin, batch_first=True)
        
        
        # GRU ==================================================
        elif self.model_type == "GRU":

            print("=== fitting GRU ===")
            
            self.layer_recur = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                    batch_first=True)
            

        # MRU ==================================================
        elif self.model_type == "MRU": 

            print("=== fitting MRU ===")

            self.layer_recur = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                batch_first=True)

            parametrize.register_parametrization(self.layer_recur, "weight_ih_l0", param_MRU())
            parametrize.register_parametrization(self.layer_recur, "weight_hh_l0", param_MRU())
            parametrize.register_parametrization(self.layer_recur, "bias_hh_l0", param_MRU())
            parametrize.register_parametrization(self.layer_recur, "bias_ih_l0", param_MRU())

        else:
            raise ValueError("Model name not recognized")
        

        # custom initialization
        for name, param in self.named_parameters():
            # print('name:', name)
            if "weight" in name:
                print(name, 'init with xavier (nonlin)')
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(nonlinearity='sigmoid')) # hack for GRUs, shouldn't effect RNNs
                nn.init.xavier_normal_(param[-hidden_size:, :], gain=nn.init.calculate_gain(nonlinearity='relu' if (((model_type == "RNN") or (model_type == "RNN-untied")) & (self.rnn_nonlin == "relu")) else "tanh"))
            elif "bias" in name:
                print(name, 'init with normal(.1, .5)')
                nn.init.normal_(param, 0.1, std=.5)
            else:
                print("~~ no init:", name, param.shape)

        print('~~ layer_recur.weight_hh_l0 re-init with orthogonal')
        nn.init.orthogonal_(self.layer_recur.weight_hh_l0,  gain=nn.init.calculate_gain(nonlinearity='relu' if (((model_type == "RNN") or (model_type == "RNN-untied")) & (self.rnn_nonlin == "relu")) else "tanh"))

        print('~~ encode_weights re-init with xavier')
        nn.init.xavier_normal_(self.encode_weights, gain=nn.init.calculate_gain(nonlinearity='linear'))


        self.to(device)
            



    # Prediction ==================================================
    def forward(self, inputs):
        # Forward pass

        encode = F.linear(inputs, self.encode_weights)
        latents, _ = self.layer_recur(encode)

        output_weights = self.decode_weights if self.model_type == "RNN-untied" else self.encode_weights[:, :self.target_size].T
        outputs = F.linear(latents, output_weights)

        return outputs, latents

    


    # Initialization of hidden state ==================================================
    def init_hidden(self, n_epochs, do_zeros=False):
 
        if do_zeros:
            return nn.init.zeros_(torch.empty(self.n_layers, n_epochs, self.hidden_size)).to(self.device)
        else:
            return nn.init.normal_(torch.empty(self.n_layers, n_epochs, self.hidden_size), std=1.).to(self.device)



    # Loss function ==================================================
    def loss_function(self, output, target):
    
        return F.mse_loss(output, target)
    



    # Count parameters ==================================================
    def count_parameters(self):

        n_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.model_type == "MRU" or self.model_type == "GRU-no-reset" or self.model_type == "GRU-no-update":
            n_param -= (self.hidden_size + self.hidden_size)*(self.hidden_size + 1)

        return n_param






    # fitting function  ==================================================
    def fit(self, inputs, targets, n_epochs=100, 
            lr=.01, weight_decay=0.01, 
            record_fit=True, verbose=False,
            inputs_test=[], targets_test=[],
            do_test=False):
        
        if len(inputs_test)==0 or len(targets_test)==0:
            print("No test data provided")
            do_test = False

        # define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # save the fit
 
            

        train_loss, test_loss = [],[]

        self.train()
        for cur_epoch in range(n_epochs):

            # print(cur_epoch, n_epochs)

            # initialize the hidden, input, mask
            # h0_epoch = self.h0.expand(-1,inputs.shape[0],-1).contiguous()
            # h0_epoch = torch.matmul(input0, self.init_weights)
            inputs_epoch = inputs

            # zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # prediction
            # outputs,_ = self.forward(inputs_epoch, h0_epoch)
            outputs,_ = self.forward(inputs_epoch)


            # LOSS ==================================================
            # main loss
            loss = self.loss_function(outputs, targets)


            # graident
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)

            # update the weights
            train_loss.append(loss.item())
            optimizer.step()

            # store noiseless test loss
            if do_test:

                optimizer.zero_grad(set_to_none=True)
                
                outputs_test,_ = self.forward(inputs_test)
                loss = self.loss_function(outputs_test, targets_test)         
                test_loss.append(loss.item())
      

            # print
            if verbose:
                if (cur_epoch > 0) & ((cur_epoch+1) % 50 == 0):
                    print('Epoch: {}/{}.............'.format(cur_epoch, n_epochs), end=' ')
                    print("train loss: {:.4f}".format(train_loss[-1]), end=' ')
                    if do_test:
                        print(" || test loss: {:.4f}".format(test_loss[-1]))


        self.train_loss = train_loss
        self.test_loss = test_loss

        return train_loss, test_loss


