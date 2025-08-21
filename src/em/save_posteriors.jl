
# SET PATHS ==============================================================
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


# LOAD PACKAGES ==============================================================
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
using MAT
try
    using MATLAB
catch
    println("NO MATLAB")

end
using PDMats
using Serialization
# using MultivariateStats
# using PDMats

try
    using MATLAB
catch
    println("NO MATLAB")
end

if Sys.isapple()
    try
        using Plots
    catch
        println("NO PLOTS")
    end
else
    
end
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

load_dir = "/Users/hr0283/Dropbox (Brown)/HallM_NeSS/della-outputs";


# === PARAMETERS
model_name = "2024-05-06-16h_cueISI";
x_list = 60:5:80;
pt_list = 1:30;
# ============

for dd in eachindex(x_list)
    for pp in eachindex(pt_list)

        println("pt: $(pt_list[pp]) / x_dim: $(x_list[dd])")

        # load
        try
            global S = deserialize("$(load_dir)/$(model_name)/$(model_name)_Pt$(pt_list[pp])_xdim$(x_list[dd]).jls");
        catch
            println("NOT FOUND: $(model_name)_Pt$(pt_list[pp])_xdim$(x_list[dd]).jls")
            continue
        end
     

        ## get estimates (TRAIN) ========================================
        P_train = NeSS.posterior_estimates( S, 
                                            S.dat.y_train,
                                            S.dat.y_train_orig,  
                                            S.dat.u_train, 
                                            S.dat.u0_train,
                                                );

        write_matfile(  "$(load_dir)/$(model_name)/$(S.prm.save_name)_trainPPC.mat", 
                        smooth_mean = P_train.smooth_mean,
                        pred_orig_y = P_train.pred_orig_y,
        );



        ## get estimates (TEST) ========================================
        P_test = NeSS.posterior_estimates(  S, 
                                            S.dat.y_test,
                                            S.dat.y_test_orig,  
                                            S.dat.u_test, 
                                            S.dat.u0_test,
                                                );


        write_matfile(  "$(load_dir)/$(model_name)/$(S.prm.save_name)_testPPC.mat", 
                        smooth_mean = P_test.smooth_mean,
                        pred_orig_y = P_test.pred_orig_y,
        );



    end
end

