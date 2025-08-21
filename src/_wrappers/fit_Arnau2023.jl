
# SET PATHS =============================================================
run_cluster = length(ARGS)!=0;
if run_cluster
    NeSS_dir = "CLUSTER/SSM-Paper-2025/src"
    save_dir = "CLUSTER/SSM-Paper-2025/src";
else
    NeSS_dir = "ROOT/SSM-Paper-2025/src"
    save_dir = NeSS_dir;
end

push!(LOAD_PATH, pwd());
push!(LOAD_PATH, "$(pwd())/../");
push!(LOAD_PATH, NeSS_dir);
if run_cluster
    println(LOAD_PATH)
end
# =============================================================




# LOAD PACKAGES =============================================================
using NeSS
using Accessors
using Random
using LinearAlgebra
using Dates
# =============================================================




# CONFIGURE SYSTEM ==============================================================
BLAS.set_num_threads(1)
set_zero_subnormals(true);

rand_seed = 99; # set random seed
Random.seed!(rand_seed); 

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
println("ARGS: $(ARGS)")
println("========================================\n")
# =============================================================


# SET PARAMETERS =======================================================================
S = core_struct(
    prm=param_struct(

        seed = rand_seed,
        model_name = "MODEL_NAME",
        changelog = "CHANGELOG",
        filename = "Arnau2023_ITI800_brain50_firws-p01-30_srate-125",
        pt_list = 1:26,

        max_iter_em = run_cluster ? 2e4 : 1000,
        test_iter = 100,
        early_stop = true,

        x_dim_fast = round.(Int64, 16:16:80),
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
        ssid_lag = run_cluster ? 128 : 32,
        ), 

    dat=data_struct(

        epoch_sel = 1:1,

        pt = 16, # pt defaults
        x_dim = 32, # x_dim default

        basis_name = "bspline",
        spline_gap = 5, # number of samples between spline knots
        norm_basis=false,

        pred_list = [
            "task", "prevTask", 
            "task@prevTask",
            "RT", "prevRT", 
            "cueColor", "cueTilt", "cueRepeat",
            ],

        pred0_list = [
            "prevTask", "RT", "prevRT", 
            ],


        ),

    res=results_struct(),

    est=estimates_struct(),

    mdl=model_struct(),

);
println("--- changelog: ",S.prm.changelog, " ---\n\n")
#  =======================================================================




# FIT THE MODEL =======================================================================
@reset S.res.startTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
println("Starting fit at $(S.res.startTime_all)")


# PREPROCESS ======================================
@reset S = NeSS.preprocess_model(S);

# @reset S = NeSS.init_param_rand(S); # helfpul for debugging
# =================================================



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




# PLOT DIAGNOSTICS =======================================================================

do_plots = false
if do_plots
    try

        # plot loglik traces
        NeSS.plot_loglik_traces(S)

        
        # plot posterior predictive checks
        NeSS.plot_trial_pred(S, 20)
        
        NeSS.plot_avg_pred(S)


        # plot model
        NeSS.plot_params(S)


        # plot RT diffusion
        NeSS.plot_input_diffusion(S; input_name = "task", mod_name = "task@RT", norm_type="obs")
        NeSS.plot_input_diffusion(S; input_name = "task", mod_name = "task@RT", norm_type="state")
        
        # plot task diffusion
        NeSS.plot_input_diffusion(S; input_name = "task", mod_name = "task@switch", norm_type="obs")
        NeSS.plot_input_diffusion(S; input_name = "task", mod_name = "task@switch", norm_type="state")

        # plot taskSwitch and taskRepeat
        NeSS.plot_2input_diffusion(S; input1_name = "taskSwitch",  input2_name = "taskRepeat", norm_type="obs")
        NeSS.plot_2input_diffusion(S; input1_name = "taskSwitch",  input2_name = "taskRepeat", norm_type="state")
        NeSS.plot_2input_diffusion(S; input1_name = "taskSwitch",  input2_name = "taskRepeat", norm_type="none")


        # plot resp 
        NeSS.plot_input_diffusion(S; input_name = "resp", mod_name = "", norm_type="none")
        NeSS.plot_input_diffusion(S; input_name = "color", mod_name = "", norm_type="none")
        NeSS.plot_input_diffusion(S; input_name = "tilt", mod_name = "", norm_type="none")



    catch
    end
end
#  =======================================================================




# SAVE FIT =======================================================================
if S.prm.do_save

    println("\n========== SAVING FIT ==========")

    save_results(S);

else
    println("\n========== *NOT* SAVING FIT ==========")
end

println("=================================\n")

#  =======================================================================