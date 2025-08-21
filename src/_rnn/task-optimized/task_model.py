from datetime import datetime
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
# from torch.utils.tensorboard import SummaryWriter


# Parametrization Classes ==================================================

class param_MRU(nn.Module):
    def forward(self, W):
        w_xr, w_xz, w_xn = torch.split(W, W.shape[0] // 3)
        w_xf = torch.mean(torch.stack([w_xr, w_xz]), dim=0)
        return torch.cat([w_xf, -w_xf, w_xn])


class param_no_reset(nn.Module):

    def __init__(self, reset_scale, reset_bias):
        super().__init__()

        self.reset_scale = reset_scale
        self.reset_bias = reset_bias

    def forward(self, W):

        w_reset, w_update, w_hidden = torch.split(W, W.shape[0] // 3)

        w_reset = torch.mul(self.reset_scale, w_reset)
        w_reset.add_(self.reset_bias)

        return torch.cat([w_reset, w_update, w_hidden])
    
class param_no_update(nn.Module):

    def __init__(self, update_scale, update_bias):
        super().__init__()

        self.update_scale = update_scale
        self.update_bias = update_bias

    def forward(self, W):

        w_reset, w_update, w_hidden = torch.split(W, W.shape[0] // 3)

        w_update = torch.mul(self.update_scale, w_update)
        w_update.add(self.update_bias)
        
        return torch.cat([w_reset, w_update, w_hidden])


# ============================================================================



class RNN(nn.Module):

    # Module Initialization ==================================================
    def __init__(self, input_size=6, hidden_size=32, output_size=1,  bias=False,  
                 net_name = "GRU", fit_name = "test", winit="xavier", fit_init=0,
                 device=torch.device("mps")):
        
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = 1
        self.input_sigma = 1.
        self.net_name = net_name
        self.fit_name = fit_name
        self.fit_init = fit_init

        if net_name == "RNN":
            self.nonlin = 'relu'
        else:
            self.nonlin = 'tanh'
        print("nonlin:", self.nonlin)

        if self.fit_init == 1:
            self.h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.hidden_size))
            nn.init.xavier_normal_(self.h0)
        
        # RNN ==================================================
        if self.net_name == "RNN":

            print("=== fitting RNN ===")

            self.layer_recur = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                    nonlinearity=self.nonlin, batch_first=True)
            self.layer_out = nn.Linear(hidden_size, output_size, bias=bias)
        

        # GRU ==================================================
        elif self.net_name == "GRU":

            print("=== fitting GRU ===")
            
            self.layer_recur = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                    batch_first=True)
            self.layer_out = nn.Linear(hidden_size, output_size, bias=bias)

            
        # MRU ==================================================
        elif self.net_name == "MRU": 

            print("=== fitting MRU ===")

            self.layer_recur = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                batch_first=True)
            self.layer_out = nn.Linear(hidden_size, output_size, bias=bias)

            parametrize.register_parametrization(self.layer_recur, "weight_ih_l0", param_MRU())
            parametrize.register_parametrization(self.layer_recur, "weight_hh_l0", param_MRU())
            parametrize.register_parametrization(self.layer_recur, "bias_hh_l0", param_MRU())
            parametrize.register_parametrization(self.layer_recur, "bias_ih_l0", param_MRU())


        # GRU (no reset gate) ==================================================
        elif self.net_name == "GRU-no-reset": 

            print("=== fitting GRU (no reset gate) ===")

            self.layer_recur = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                batch_first=True)
            self.layer_out = nn.Linear(hidden_size, output_size, bias=bias)

            self.to(device)

            reset_weight_scale  = torch.scalar_tensor(0., device=device, requires_grad=False)
            reset_weight_bias   = torch.scalar_tensor(0., device=device, requires_grad=False)
            reset_bias_scale     = torch.scalar_tensor(0., device=device, requires_grad=False)
            reset_bias_bias     = torch.scalar_tensor(100., device=device, requires_grad=False)
  
            parametrize.register_parametrization(self.layer_recur, "weight_ih_l0", param_no_reset(reset_scale=reset_weight_scale, reset_bias=reset_weight_bias) )
            parametrize.register_parametrization(self.layer_recur, "weight_hh_l0", param_no_reset(reset_scale=reset_weight_scale, reset_bias=reset_weight_bias) )

            parametrize.register_parametrization(self.layer_recur, "bias_hh_l0", param_no_reset(reset_scale=reset_bias_scale, reset_bias=reset_bias_bias) )
            parametrize.register_parametrization(self.layer_recur, "bias_ih_l0", param_no_reset(reset_scale=reset_bias_scale, reset_bias=reset_bias_bias) )



        # GRU (no update gate) ==================================================
        elif self.net_name == "GRU-no-update": 

            print("=== fitting GRU (no update gate) ===")

            self.layer_recur = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers, 
                                batch_first=True)
            self.layer_out = nn.Linear(hidden_size, output_size, bias=bias)

            self.to(device)

            update_weight_scale  = torch.scalar_tensor(0., device=device, requires_grad=False)
            update_weight_bias   = torch.scalar_tensor(0., device=device, requires_grad=False)
            update_bias_scale     = torch.scalar_tensor(0., device=device, requires_grad=False)
            update_bias_bias     = torch.scalar_tensor(-100., device=device, requires_grad=False)
  
            parametrize.register_parametrization(self.layer_recur, "weight_ih_l0", param_no_update(update_scale=update_weight_scale, update_bias=update_weight_bias) )
            parametrize.register_parametrization(self.layer_recur, "weight_hh_l0", param_no_update(update_scale=update_weight_scale, update_bias=update_weight_bias) )

            parametrize.register_parametrization(self.layer_recur, "bias_hh_l0", param_no_update(update_scale=update_bias_scale, update_bias=update_bias_bias) )
            parametrize.register_parametrization(self.layer_recur, "bias_ih_l0", param_no_update(update_scale=update_bias_scale, update_bias=update_bias_bias) )

        else:
            raise ValueError("Model name not recognized")



        # custom initialization
        for name, param in self.named_parameters():
            print(name)
            if "weight" in name:
                if winit == "xavier":
                    nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(nonlinearity='sigmoid'))
                    nn.init.xavier_normal_(param[-hidden_size:, :], gain=nn.init.calculate_gain(nonlinearity=self.nonlin))
                elif winit == "orth":
                    nn.init.orthogonal_(param, gain=nn.init.calculate_gain(nonlinearity='sigmoid'))
                    nn.init.orthogonal_(param[-hidden_size:, :], gain=nn.init.calculate_gain(nonlinearity=self.nonlin))
                else:
                    raise ValueError("Initialization not recognized")

            elif "bias" in name:
                nn.init.normal_(param, 1., std=.25)
                nn.init.normal_(param[-hidden_size:], 0., std=.25)
            else:
                print("no init:", name, '__', "weight" in name,'__', param.shape)
        

        self.to(device)
            



    # Prediction ==================================================
    def forward(self, inputs, hidden=torch.empty(0,0,0)):

        # check if hidden input, if not initialize
        if hidden.shape[0] == 0:
            print('init hidden')
            hidden = self.init_hidden(inputs.shape[0]).to(self.device)

        # forward pass
        latents,_ = self.layer_recur(inputs, hidden)
        outputs = self.layer_out(latents)
        return outputs, latents





    # Initialization of hidden state ==================================================
    def init_hidden(self, n_epochs, do_zeros=False):
 
        if self.fit_init:
            h0 = self.h0.repeat(1, n_epochs, 1)
        else:
            h0 = nn.init.zeros_(torch.empty(self.n_layers, n_epochs, self.hidden_size)).to(self.device)

        if do_zeros:
            return h0
        else:
            return h0 + nn.init.normal_(torch.empty(self.n_layers, n_epochs, self.hidden_size), std=1.).to(self.device)


    # Count parameters ==================================================
    def count_parameters(self):

        n_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.net_name == "MRU" or self.net_name == "GRU-no-reset" or self.net_name == "GRU-no-update":
            n_param -= (self.hidden_size + self.input_size + 2)*(self.hidden_size)

        return n_param




    # fitting function  ==================================================
    def fit(self, inputs, targets, mask, n_epochs=100, 
            lr=.01, weight_decay=0.01, 
            verbose=False,
            do_test=False, input_test=[], target_test=[], mask_test=[]):


        # optimizer
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr=lr, weight_decay=weight_decay)
    
        # loss function
        loss_function = nn.BCEWithLogitsLoss(weight=mask)
    
        # preallocate
        noisy_loss, clean_loss = [],[]
        inputs_train = torch.empty_like(inputs, device=self.device)
        outputs_train = torch.empty_like(inputs, device=self.device)

        if do_test:
            loss_function_test = nn.BCEWithLogitsLoss(weight=mask_test)
            outputs_test = torch.empty_like(input_test, device=self.device)


        self.train()
        for cur_epoch in range(n_epochs):

            # print(cur_epoch, n_epochs)

            # initialize the hidden state & inputs
            h0 = self.init_hidden(inputs.shape[0]).to(self.device)
            inputs_train = inputs + torch.randn_like(inputs)*self.input_sigma

            # zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # prediction
            outputs_train,_ = self.forward(inputs_train, h0)

            # compute loss
            loss_train = loss_function(outputs_train, targets)
            noisy_loss.append(loss_train.item())

            # calculate graident
            loss_train.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)

            # update weights
            optimizer.step()

            # store losses
            if do_test:

                optimizer.zero_grad(set_to_none=True)

                # predict                
                outputs_test,_ = self.forward(input_test, self.init_hidden(input_test.shape[0], do_zeros=True).to(self.device))
                
                # loss
                loss_test = loss_function_test(outputs_test, target_test)
                clean_loss.append(loss_test.item())


            # print
            if verbose:
                if (cur_epoch > 0) & ((cur_epoch+1) % 50 == 0):
                    print('Epoch: {}/{}.....'.format(cur_epoch, n_epochs), end=' ')
                    print("noisy loss: {:.4f}".format(noisy_loss[-1]), end=' ')
                    if do_test:
                        print(" || clean loss: {:.4f}".format(clean_loss[-1]))



        return noisy_loss, clean_loss





