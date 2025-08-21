# new recover parameters


# imports
if Sys.isapple()
    NeSS_dir = "/Users/hr0283/Dropbox (Brown)/HallM_NeSS/src"
    save_dir = NeSS_dir;
else
    NeSS_dir = "/home/hr0283/HallM_NeSS/src"
    save_dir = "/scratch/gpfs/hr0283/HallM_NeSS/src";
end

# add paths
push!(LOAD_PATH, pwd());
push!(LOAD_PATH, "$(pwd())/../");
push!(LOAD_PATH, NeSS_dir);
if Sys.isapple() == 0
    println(LOAD_PATH)
end

# load modules
using NeSS

# load packages
using Accessors
using Random
using LinearAlgebra
using Dates
using Plots
using StatsBase
using BSplines
using Serialization

using MAT
try
    using MATLAB
catch
    println("NO MATLAB")
end

using PDMats
using BenchmarkTools
using JET



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
Random.seed!(99)


model_name = "GRU_A23_justSwitch_noiseSSM";
model_id = "step4_permssid"
splines_step = 4;



# load SSM
dat = matread("/Users/hr0283/Dropbox (Brown)/HallM_NeSS/RNN_TS/task-optimized/results/ssm/$(model_name)/$(model_name).mat")
dat_latents = permutedims(convert(Array{Float64}, dat["latents"]), (3,2,1));
dat_inputs = permutedims(convert(Array{Float64}, dat["inputs"]), (3,2,1));
dat_conds =  dat["conditions"];













# get inputs
cue_hot =  (abs.(dat_inputs[1,:,1] .- dat_inputs[2,:,1]) .> 0);
trial_hot =  (abs.(dat_inputs[3,:,1] .- dat_inputs[4,:,1]) .> 0);
second_hot = (1:size(dat_inputs,2) .> round(size(dat_inputs,2)/2));

prev_cue_sel = findfirst((cue_hot) .& (second_hot.==0));

cue_start = findfirst((cue_hot) .& (second_hot.==1));
cue_end = findlast((cue_hot) .& (second_hot.==1));
isi_end = findfirst((trial_hot) .& (second_hot.==1)) -1


time_sel =  cue_start:(isi_end+11)
cue_sel = 1:(cue_end-cue_start)+1
n_times = length(time_sel)


println("epoch duration = $n_times")
       
train_sel = randperm!(collect(1:512));
test_sel = randperm!(collect(513:768));
# train_sel = 1:512;
# test_sel = 513:768;
n_trials = 768;



# get task and switch inputs (trial vector)
pred_task = (dat_inputs[1,cue_start,1:n_trials] .- dat_inputs[2,cue_start,1:n_trials])/2.0;
global pred_switch = zeros(size(pred_task));
for ii in axes(pred_switch,1)
    if dat_conds[ii]["switch"]
        global pred_switch[ii] = 1.0;
    else
        global pred_switch[ii] = -1.0;
    end
end

pred_cond = [pred_task.*(pred_switch.==1) pred_task.*(pred_switch.==-1) pred_switch]';
n_pred = size(pred_cond,1);


# build basis set
n_bases = round(Int64, n_times/splines_step);
basis = averagebasis(4, LinRange(1, n_times, n_bases));
pred_basis = ["spline" for _ in 1:n_bases];


u_dim = n_bases*(n_pred + 1);
u = zeros(u_dim, n_times, n_trials); 

for tt in axes(u,2)
    bs = bsplines(basis, tt);
    u[collect(axes(bs,1)),tt,:] .= collect(bs);
end

println("n bases: $(n_bases), breakpoints: $(round.(breakpoints(basis),sigdigits=4))")


# convolve predictors with basis set
for bb = 1:n_bases
    for uu in axes(pred_cond,1)
        u[(n_bases) + ((uu-1)*n_bases)+bb,:,:] .= u[bb,:,:] .* pred_cond[uu,:]';
    end
end


# check collinearity
ul = deepcopy(reshape(u, u_dim, n_times*n_trials)');
f=svd(ul);

println("predictor mean quartiles: $(round(median(mean(ul, dims=1)), sigdigits=4)) +/- $(round(iqr(mean(ul, dims=1))/2, sigdigits=4))")
println("predictor var quartiles: $(round(median(var(ul, dims=1)), sigdigits=4)) +/- $(round(iqr(var(ul, dims=1))/2, sigdigits=4))");
println("collinearity metric (best=1, threshold=30): $(round(f.S[1]/f.S[end], sigdigits=4))"); 
println("========================================\n")



# build initial state list
u0 = zeros(2, n_trials);
u0[1,:] .= 1.0; # constant term
u0[2,:] .= pred_task.*pred_switch; # constant term



# make matrices
u_train = deepcopy(u[:,:,train_sel]);
u0_train = deepcopy(u0[:,train_sel]);

u_test = deepcopy(u[:,:,test_sel]);
u0_test = deepcopy(u0[:,test_sel]);


n_train = size(u_train, 3);
n_test =  size(u_test, 3);
n_times = size(u_train, 2);

n_chans = size(dat_latents, 1);
u0_dim = size(u0_train, 1);
u_dim = size(u_train, 1);

y_train_orig = dat_latents[:,time_sel,train_sel];
y_test_orig = dat_latents[:,time_sel,test_sel];






S = core_struct(
    prm=param_struct(
        model_name = model_name,

        do_save = true,

        pt_list = 1:1,
        
        max_iter_em = ~Sys.isapple() ? 2e4 : 10000,
        test_iter = 10,
        early_stop = true,

        y_transform = "PCA",
        PCA_ratio = .90,

        NeSS_dir = NeSS_dir,
        save_dir = save_dir,

        ssid_fit = "fit",

        ssid_lag = 100,
        ), 

    dat=data_struct(
        x_dim = 100,
        n_train = n_train,
        n_test =  n_test,
        n_times = n_times,
        u_dim = u_dim,
        u0_dim = u0_dim,
        n_chans = n_chans,

        dt = 1.0,
        ),

    res=results_struct(),

    est=estimates_struct(),

    mdl=model_struct(),

);



# fill-in data struct
@reset S.dat.y_train_orig = y_train_orig;
@reset S.dat.u_train = u_train;
@reset S.dat.u0_train = u0_train;

@reset S.dat.y_test_orig = y_test_orig;
@reset S.dat.u_test = u_test;
@reset S.dat.u0_test = u0_test;

# whiten and init moments
S = deepcopy(NeSS.whiten_y(S)); # ======== apply PCA to y


println("ssid lag: $(S.prm.ssid_lag) \n");

# set estimates
@reset S.est = deepcopy(set_estimates(S));

# null loglik
NeSS.null_loglik!(S);

# save paths
@reset S.prm.save_name = "$(S.prm.model_name)_Pt$(S.dat.pt)_xdim$(S.dat.x_dim)_$(model_id)";
setup_dir(S);

# REPORT FIT INFO ==============================================
println("\n========== FIT INFO ==========")
println("save name: $(S.prm.save_name)")
println("using filename: $(S.prm.filename)")
println("participant: $(S.dat.pt)")
println("latent dimensions: $(S.dat.x_dim)")
println("observed dimensions: $(S.dat.y_dim)\n")

println("max EM iterations: $(S.prm.max_iter_em)");
println("SSID fitting: $(S.prm.ssid_fit)");
println("temporal bases: $(S.dat.n_bases)\n")

println("SSID type: $(S.prm.ssid_type)")
println("SSID lag: $(S.prm.ssid_lag)")
println("regressors: $(S.dat.n_pred)")
println("input dimensions: $(S.dat.u_dim)");
println("initial inputs dimensions: $(S.dat.u0_dim)");
println("training trials: $(S.dat.n_train)");
println("testing trials: $(S.dat.n_test)");
println("number of channels: $(S.dat.n_chans)")
println("Q type: $(S.prm.Q_type) / R type: $(S.prm.R_type) / P0 type: $(S.prm.P0_type)")
println("========================================\n")
#  =======================================================================



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

# save SSID
serialize("$(S.prm.save_dir)/../fit-SSID/$(S.prm.model_name)/$(S.prm.save_name)_SSID.jls", S)
# ================================================






# Expectation Maximization (EM) ==================
@reset S = NeSS.fit_EM(S);
# ================================================






try

    # plot loglik traces
    NeSS.plot_loglik_traces(S)

        
    # plot posterior predictive checks
    NeSS.plot_trial_pred(S, 20)
    
    NeSS.plot_avg_pred(S)


    # plot model
    NeSS.plot_params(S)


    # plot RT diffusion
    NeSS.plot_2input_diffusion(S; input1_name = "taskSwitch",  input2_name = "taskRepeat", norm_type="none")



 
catch
end








# SAVE FIT =======================================================================
if S.prm.do_save

    println("\n========== SAVING FIT ==========")

    save_results(S);

else
    println("\n========== *NOT* SAVING FIT ==========")
end

println("=================================\n")

#  =======================================================================












