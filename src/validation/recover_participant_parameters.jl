
ROOT=""

# SET PATHS =============================================================
run_cluster = length(ARGS)!=0;

NeSS_dir = "$ROOT/HallM_NeSS/src"
save_dir = NeSS_dir;

push!(LOAD_PATH, pwd());
push!(LOAD_PATH, "$(pwd())/../");
push!(LOAD_PATH, NeSS_dir);
if run_cluster
    println(LOAD_PATH)
end
# =============================================================




# LOAD PACKAGES =============================================================
using Accessors
using Dates
using LinearAlgebra
using MAT
using NeSS
using PDMats
using Plots
using Random
using StatsBase
# =============================================================



# CONFIGURE SYSYSTEM ==============================================================
BLAS.set_num_threads(1)
set_zero_subnormals(true);


println("\n========== SYSTEM INFO ==========")
try
    display(versioninfo(verbose=true))
catch
    try
        println("Julia Version = $(VERSION)")
        display(versioninfo())
    catch
    end
end
println("BLAS config = $(BLAS.get_config())")
println("BLAS threads = $(BLAS.get_num_threads())")
println("========================================\n")
# =============================================================

# parameters
rand_seed=99
which_pt = 7 # good: 6



# Generative Paramters
println("loading data ...")

# save_name = "paramrec__H19_$(which_pt)_"
# dat = matread("/Users/hr0283/Dropbox (Brown)/HallM_NeSS/della-outputs/2024-10-31-10h_H19__prepTrial200-prevTask/2024-10-31-10h_H19__prepTrial200-prevTask_Pt$(which_pt)_xdim112.mat")


save_name = "paramrec__GRU_450-50-short_$(which_pt)_"
dat = matread("$ROOT/della-outputs/_RNN/EM-mat/GRU_single450_switch50_150reps_ITI20_cue100_trial2_mixTrain/GRU_single450_switch50_150reps_ITI20_cue100_trial2_mixTrain_Pt$(which_pt)_xdim112.mat")



A = dat["mdl"]["A"];
B = dat["mdl"]["B"];
Q = PDMat(dat["mdl"]["Q"]["mat"]);

C = dat["mdl"]["C"];
R = PDMat(dat["mdl"]["R"]["mat"]);

B0 = dat["mdl"]["B0"];
P0 = PDMat(dat["mdl"]["P0"]["mat"]);

sim = (A = A, B = B, Q = Q, C = C, R = R, B0 = B0, P0 = P0);


u_train = dat["dat"]["u_train"]
u0_train = dat["dat"]["u0_train"]
u_test = dat["dat"]["u_test"]
u0_test = dat["dat"]["u0_test"]


n_train = size(u_train, 3);
n_test =  size(u_test, 3);
n_times = size(u_train, 2);

n_chans = size(sim.C, 1);
x_dim = size(sim.A, 1);
ssid_lag = x_dim;
u0_dim = size(u0_train, 1);
u_dim = size(u_train, 1);


println("\nFIT INFO\nn_train = $n_train, \nn_test = $n_test, \nn_times = $n_times")
println("n_chans = $n_chans, \nx_dim = $x_dim, \nssid_lag = $ssid_lag")
println("u0_dim = $u0_dim, \nu_dim = $u_dim\n\n")




println("simulating data ...")
x_train, y_train_orig  = NeSS.generate_lds_trials( sim.A, sim.B, sim.Q,
                                    sim.C, sim.R, 
                                    sim.B0, sim.P0,
                                    u_train, u0_train,  
                                    n_times, n_train);


x_test, y_test_orig  = NeSS.generate_lds_trials( sim.A, sim.B, sim.Q,
                                        sim.C, sim.R, 
                                        sim.B0, sim.P0,
                                        u_test,u0_test,  
                                        n_times, n_test);





S = core_struct(
    prm=param_struct(
        seed = rand_seed,
        model_name = "MODEL_NAME",
        changelog = "CHANGELOG",
        filename = "HallMcMaster2019_ITI100-Cue200-ISI400-Trial200_srate@125_filt@0-30",
        pt_list = 1:30,

        max_iter_em = run_cluster ? 2e4 : 1e4,
        test_iter = 100,
        early_stop = true,

        x_dim_fast = round.(Int64, 16:16:128),
        x_dim_slow = round.(Int64, 96:16:128),
        
        NeSS_dir = NeSS_dir,
        save_dir = save_dir,
        do_save = run_cluster ? true : false, 

        y_transform = "PCA",
        PCA_ratio = .99,

        do_trial_sel = true, # only current & previous accurate

        ssid_fit = length(ARGS) > 2 ? ARGS[3] : "fit", # fit, load
        ssid_save =  length(ARGS) > 3 ? parse(Bool, ARGS[4]) : false, # SAVE SSID AND THEN EXIT

        ssid_type = :CVA,
        ssid_lag = 128,
        ), 

    dat=data_struct(
        n_train = n_train,
        n_test =  n_test,
        n_times = n_times,
        # x_dim = x_dim,
        u_dim = u_dim,
        u0_dim = u0_dim,
        n_chans = n_chans,
        n_bases = dat["dat"]["n_bases"];
        dt = dat["dat"]["dt"],

        x_dim = 112, # x_dim default

        basis_name = "bspline",


        ),

    res=results_struct(),

    est=estimates_struct(),

    mdl=model_struct(),

);







# fill-in data struct
@reset S.dat.y_train_orig = y_train_orig;
@reset S.dat.y_train = y_train_orig;
@reset S.dat.u_train = u_train;
@reset S.dat.u0_train = u0_train;
@reset S.dat.y_dim = size(y_train_orig, 1);

@reset S.dat.y_test_orig = y_test_orig;
@reset S.dat.y_test = y_test_orig;
@reset S.dat.u_test = u_test;
@reset S.dat.u0_test = u0_test;

@reset S.dat.W = I(S.dat.y_dim);
@reset S.dat.mu = zeros(S.dat.y_dim);



# whiten and init moments
# println("whitening ...");
# S = deepcopy(NeSS.whiten_y(S)); # ======== apply PCA to y

# set estimates
@reset S.est = deepcopy(set_estimates(S));

# null loglik
NeSS.null_loglik!(S);








# FIT THE MODEL =======================================================================
@reset S.res.startTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
println("Starting fit at $(S.res.startTime_all)")



# Subspace Identification (SSID) ==================
if S.prm.ssid_fit == "fit"

     # fit SSID
     @reset S = NeSS.fit_SSID(S);

elseif S.prm.ssid_fit == "load"

    # load previously-fit SSID
    @reset S = NeSS.load_SSID(S);

end
# ================================================


# Expectation Maximization (EM) ==================
@reset S = NeSS.fit_EM(S);
# ================================================



@reset S.res.endTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
println("Finished fit at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))")
#  =======================================================================







try

    # plot traces
    NeSS.plot_loglik_traces(S)


    # plot stacked fit
    NeSS.plot_y_preds(S, 20)


    # plot model
    NeSS.plot_params(S)

 
catch
end







# PLOT PARAMETER RECOVERY ===============================================================================================

using LinearAlgebra, StatsBase


# factor alignment (Train)
# P = NeSS.posterior_mean(S, S.dat.y_train, S.dat.y_train_orig, S.dat.u_train, S.dat.u0_train);
# x_est = reshape(stack(P.smooth_mean), (S.dat.x_dim, S.dat.n_times*S.dat.n_train));
# x_sim = reshape(x_train, (S.dat.x_dim, S.dat.n_times*S.dat.n_train));

# factor alignment (Test)
P = NeSS.posterior_mean(S, S.dat.y_test, S.dat.y_test_orig, S.dat.u_test, S.dat.u0_test);
x_est = reshape(stack(P.smooth_mean), (S.dat.x_dim, S.dat.n_times*S.dat.n_test));
x_sim = reshape(x_test, (S.dat.x_dim, S.dat.n_times*S.dat.n_test));

w_x = x_sim/x_est;



# plot latent match
p1=plot(((w_x*x_est[:,1:150])')[1:5:end,:], label="")
p2=plot(x_sim[1:5:end,1:150]', label="")
# p3=plot((S.mdl.C*x_est[:,1:150])', label="")
# p4=plot((S.dat.y_train[:,1:150,1])', label="")

plot(p1,p2,layout=(1,2), size=(1200,600))



# compare sv spectrum
eig_sim=eigen(sim.A).values
eig_n4=eigen(w_x*S.res.mdl_ssid.A/w_x).values
eig_fit=eigen(w_x*S.mdl.A/w_x).values

real_plt = plot(size=(800,400), title="A: real eigval")
plot!(real.(eig_sim), label="sim", linewidth=3)
plot!(real.(eig_n4), label="fit init", linewidth=1)
plot!(real.(eig_fit), label="fit final", linewidth=2)

im_plt = plot(size=(800,400), title="A: eigval")
# scatter!((imag.(eig_sim)), label="sim", markersize=6)
# scatter!((imag.(eig_fit)), label="fit final", markersize=4)
# scatter!((imag.(eig_n4)), label="fit init", markersize=3)
plot!(sin.(-pi:.001:pi), cos.(-pi:.001:pi), label="", linewidth=2, color=:black, linestyle=:dash)
scatter!(eig_sim, label="sim", markersize=6)
scatter!(eig_fit, label="fit final", markersize=4)
scatter!(eig_n4, label="fit init", markersize=3)


plot(real_plt, im_plt, layout=(1,2))    




# ====== PLOT ESTIMATES =================================================================================================




# plot parameters

# A
a0 = heatmap(I(x_dim) - w_x*S.res.mdl_ssid.A/w_x, clim=(-.5, .5))
title!("I-A init")
a1 = heatmap(I(x_dim) - w_x*S.mdl.A/w_x, clim=(-.5, .5))
title!("I-A est")
a2 = heatmap(I(x_dim) - sim.A, clim=(-.5, .5))
title!("I-A sim - $(round(cor(vec(I - w_x*(S.mdl.A)/w_x), vec(I- sim.A)), digits=4))")
println("A: $(round(cor(vec(I - w_x*(S.mdl.A)/w_x), vec(I- sim.A)), digits=4))") 


# B
b0 = heatmap(w_x*S.res.mdl_ssid.B, clim=(-.1, .1))
title!("B init")
b1 = heatmap(w_x*S.mdl.B, clim=(-.1, .1))
title!("B est")
b2 = heatmap(sim.B, clim=(-.1, .1))
title!("B sim $(round(cor(vec(w_x*S.mdl.B), vec(sim.B)), digits=4))")
println("B: $(round(cor(vec(w_x*S.mdl.B), vec(sim.B)), digits=4))") 


histogram(sum(S.mdl.B.^2,dims=1))
scatter(sqrt.(sum(S.mdl.B.^2,dims=1))', diag(cor((w_x*S.mdl.B), sim.B)), label="")

dat["dat"]["pred_name"][sortperm(diag(cor((w_x*S.mdl.B), sim.B)))]

# Q
q0 = heatmap(Matrix(w_x*S.res.mdl_ssid.Q*w_x'))
title!("Q init")
q1 = heatmap(Matrix(w_x*S.mdl.Q*w_x'))
title!("Q est")
q2 = heatmap(Matrix(sim.Q))
title!("Q sim $(round(cor(vec(w_x*S.mdl.Q*w_x'), vec(sim.Q)), digits=4))")
println("Q: $(round(cor(vec(w_x*S.mdl.Q*w_x'), vec(sim.Q)), digits=4))") 



# C
c0 = heatmap(Matrix(S.dat.W*S.res.mdl_ssid.C/w_x))
title!("C init")
c1 = heatmap(Matrix(S.dat.W*S.mdl.C/w_x))
title!("C est")
c2 = heatmap(sim.C)
title!("C sim $(round(cor(vec(S.dat.W*S.mdl.C/w_x), vec(sim.C)), digits=5))")
println("C: $(round(cor(vec(S.dat.W*S.mdl.C/w_x), vec(sim.C)), digits=5))") 


# R
fr0 = heatmap(Matrix(S.dat.W*S.res.mdl_ssid.R*S.dat.W'))
title!("R init")
fr1 = heatmap(Matrix(S.dat.W*S.mdl.R*S.dat.W'))
title!("R est")
fr2 = heatmap(Matrix(sim.R))
title!("R sim $(round(cor(vec(Matrix(S.dat.W*S.mdl.R*S.dat.W')), vec(sim.R)), digits=4))")
@show 
println("R: $(round(cor(vec(Matrix(S.dat.W*S.mdl.R*S.dat.W')), vec(sim.R)), digits=4))") 



# R
Rw_ssid = S.dat.W*S.res.mdl_ssid.R*S.dat.W';
Cw_ssid = S.dat.W*(S.res.mdl_ssid.C/w_x);

Rw = S.dat.W*S.mdl.R*S.dat.W';
Cw = S.dat.W*(S.mdl.C/w_x);

circ0 = heatmap(Matrix((Cw_ssid'/Rw_ssid)*Cw_ssid))
title!("CiRC init")
circ1 = heatmap(Matrix((Cw'/Rw)*Cw))
title!("CiRC est")
circ2 = heatmap(Matrix((sim.C'/sim.R)*sim.C))
title!("CiRC sim $(round(cor(vec(Matrix((Cw'/Rw)*Cw)), vec((sim.C'/sim.R)*sim.C)), digits=4))")
@show 
println("CiRC: $(round(cor(vec(Matrix((Cw'/Rw)*Cw)), vec((sim.C'/sim.R)*sim.C)), digits=4))") 




# P0
p00 = heatmap(w_x*S.res.mdl_ssid.P0*w_x')
title!("P0 init")
p01 = heatmap(w_x*S.mdl.P0*w_x')
title!("P0 est")
p02 = heatmap(Matrix(sim.P0))
title!("P0 sim $(round(cor(vec(w_x*S.mdl.P0*w_x'), vec(sim.P0)), digits=4))")
println("P0: $(round(cor(vec(w_x*S.mdl.P0*w_x'), vec(sim.P0)), digits=4))") 
 


# B0
B00 = heatmap(w_x*S.res.mdl_ssid.B0[:,:,1])
title!("B0 init")
B01 = heatmap(w_x*S.mdl.B0[:,:,1])
title!("B0 est")
B02 = heatmap(sim.B0[:,:,1])
title!("B0 sim $(round(cor(vec(w_x*S.mdl.B0), vec(sim.B0)), digits=4))")
println("B0: $(round(cor(vec(w_x*S.mdl.B0), vec(sim.B0)), digits=4))") 



plot(   a0,a1,a2,
        b0,b1,b2,
        q0,q1,q2,
        c0,c1,c2,
        fr0,fr1,fr2,
        circ0,circ1,circ2,
        p00,p01, p02, 
        B00,B01, B02,
        layout=(8,3), size=(1000,1600))


using MATLAB
write_matfile("$(save_name)_xdim$(S.dat.x_dim)_pt$(which_pt).mat", 
        a0 = w_x*S.res.mdl_ssid.A/w_x,
        a1 = w_x*S.mdl.A/w_x,
        a2 = sim.A,
        b0 = w_x*S.res.mdl_ssid.B,
        b1 = w_x*S.mdl.B,
        b2 = sim.B,
        q0 = w_x*S.res.mdl_ssid.Q*w_x',
        q1 = w_x*S.mdl.Q*w_x',
        q2 = sim.Q,
        c0 = S.dat.W*S.res.mdl_ssid.C/w_x,
        c1 = S.dat.W*S.mdl.C/w_x,
        c2 = sim.C,
        fr0 = S.dat.W*S.res.mdl_ssid.R*S.dat.W',
        fr1 = S.dat.W*S.mdl.R*S.dat.W',
        fr2 = sim.R,
        circ0 = Matrix((S.dat.W*S.res.mdl_ssid.C/w_x)'/Rw_ssid*(S.dat.W*S.res.mdl_ssid.C/w_x)),
        circ1 = Matrix((S.dat.W*S.mdl.C/w_x)'/Rw*(S.dat.W*S.mdl.C/w_x)),
        circ2 = Matrix((sim.C'/sim.R)*sim.C),
        p00 = w_x*S.res.mdl_ssid.P0*w_x',
        p01 = w_x*S.mdl.P0*w_x',
        p02 = sim.P0,
        B00 = w_x*S.res.mdl_ssid.B0[:,:,1],
        B01 = w_x*S.mdl.B0[:,:,1],
        B02 = sim.B0[:,:,1],
        mdl=S.mdl,
        dat=S.dat,
        res=S.res,
        prm=S.prm,
        w_x = w_x,
        );

println("saved to $(save_name)_xdim$(S.dat.x_dim)_pt$(which_pt).mat")





