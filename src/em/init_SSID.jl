
# SET PATHS =============================================================
if Sys.isapple()
    NeSS_dir = "/Users/hr0283/Dropbox (Brown)/HallM_NeSS/NeSS"
    save_dir = NeSS_dir;
else
    NeSS_dir = "/home/hr0283/HallM_NeSS/NeSS"
    save_dir = "/scratch/gpfs/hr0283/HallM_NeSS/NeSS";
end

push!(LOAD_PATH, pwd());
push!(LOAD_PATH, "$(pwd())/../");
push!(LOAD_PATH, NeSS_dir);
if Sys.isapple() == 0
    println(LOAD_PATH)
end
# =============================================================




# LOAD PACKAGES =============================================================
# module
include("$(NeSS_dir)/NeSS.jl")
using Pkg
Pkg.instantiate()
using .NeSS

# packages
using Accessors
using Random
using LinearAlgebra
using Dates
    

# =============================================================




# CONFIGURE SYSYSTEM =============================================================
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
println("========================================\n")
# =============================================================




# SET PARAMETERS ==============================================================
S = core_struct(
    prm=param_struct(
        seed = rand_seed,
        model_name = "2024-06-13-18h_H19_basedBinsRew",
        changelog = "run 100ms bins; RT/reward",
        # filename = "HallMcMaster2019_Cue200-ISI400",
        filename = "HallMcMaster2019_ITI100-Cue200-ISI400-Trial300_noSel",
        pt_list = 1:30,

        max_iter_em = 2000,
        test_iter = 100,
        early_stop = true,

        x_dim_fast = round.(Int64, 40:10:100),
        x_dim_slow = round.(Int64, 55:60),
        
        do_fast = true,

        NeSS_dir = NeSS_dir,
        save_dir = save_dir,
        do_save = Sys.isapple() ? false : true, # dont save on apple, only on (linux) cluster

        y_baseline = true,
        y_transform = "PCA",
        PCA_ratio = .99,
        y_ICA = false,
        
        ssid_type = :CVA,
        ssid_ICA = true,

        ssid_lag = 128,

        ), 

    dat=data_struct(

        epoch_sel = 2:3,

        pt = 11, # pt default
        x_dim = 40, # x_dim default

        basis_name = "cueISI",
        bin_width = .100,

        pred_list = [
            "task", 
            "switch", "task@switch",
            "RT", "task@RT",
            "acc", "task@acc", "RT@acc",
            "rew", "task@rew", 
            "cueShape", "cueColor", "cueRepeat",
            ],

        pred0_list = [
            ],


        ),

    res=results_struct(),

    est=estimates_struct(),

    mdl=model_struct(),

);
#  =======================================================================




# FIT THE MODEL =======================================================================

for pp in eachindex(pt_list)

    @reset S.res.startTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
    println("Starting pt $(pp) at $(S.res.startTime_all)")


    @reset S = NeSS.preprocess_model(S); # PREPROCESS

    @reset S = NeSS.fit_model_SSID(S); # FIT SSID

    println("\n========== SAVING SSID ==========")
    save_SSID(S);
    println("=================================\n")

    @reset S.res.endTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
    println("Finished pt $(pp) at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))")


end

#  =======================================================================

