module NeSS

# import
using LinearAlgebra
using StatsFuns
using StatsBase
using Random
using Distributions
using FileIO
using MAT
using Dates
using Accessors
using LegendrePolynomials
using MultivariateStats
using PDMats
using SpecialFunctions
using Serialization
using ControlSystems
using ControlSystemIdentification

using BSplines
using OffsetArrays

try
    using MATLAB
catch
    println("NO MATLAB")
end


# add folders to path
if Sys.isapple()

    push!(LOAD_PATH, "$(pwd())/likelihoods");
    push!(LOAD_PATH, "$(pwd())/_wrappers");
    push!(LOAD_PATH, "$(pwd())/EM");

else

    push!(LOAD_PATH, "$(pwd())/../likelihoods");
    push!(LOAD_PATH, "$(pwd())/../_wrappers");
    push!(LOAD_PATH, "$(pwd())/../EM");

end




# short functions ===================


# tol_PD =============================
function tol_PD(A_sym::Union{Symmetric, Hermitian, PDMat}; tol=1e-6)::PDMat

    l, Q = eigen!(A_sym);    

    l_r = max.(l ./ l[end], 0.0);
    newl =  (l[end] - l[end]*tol).*l_r .+ l[end]*tol;
    return PDMat(X_A_Xt(PDiagMat(newl), Q));

end

tol_PD(A::Matrix; tol=1e-6)::PDMat = tol_PD(hermitianpart(A); tol=tol);


# tol_PSD =============================
function tol_PSD(A_sym::Union{Symmetric, Hermitian, PDMat})::Hermitian

    l, Q = eigen!(A_sym);
    return X_A_Xt(PDiagMat(max.(l, 0.0)), Q)

end

tol_PSD(A::Matrix)::Hermitian = tol_PSD(hermitianpart(A))::Hermitian;




# diag_PD =============================
function diag_PD(A; tol=1e-6)
    # this should be improved to match tol_PD
    # however, diagonal noise never works as well

    return PDiagMat(max.(diag(A), tol));

end


# format_noise =============================
function format_noise(X, type; tol=1e-6)

    if type == "identity"

        Xf = I(size(X,1));

    elseif type == "diagonal"

        Xf = diag_PD(X; tol=tol);

    elseif type == "full"

        Xf = tol_PD(X; tol=tol);

    else

        error("type not recognized")

    end

    return Xf

end


# misc =============================
init_PD(d) = PDMat(diagm(ones(d)));

init_PSD(d) = Hermitian(diagm(ones(d)));

zsel(x,sel) =  (x[sel] .- mean(x[sel])) ./ std(x[sel]);

zsel_tall(x,sel) =  ((x .- mean(x[sel])) ./ std(x[sel])).*sel;

zdim(x;dims=1) = (x .- mean(x, dims=dims)) ./ std(x, dims=dims);

sumsqr(x) = sum(x.*x);

split_list(x) = split(x, "@");

demix(S, y) = S.dat.W' * (y .- S.dat.mu);
remix(S, y) = (S.dat.W * y) .+ S.dat.mu;


export zsel, zsel_tall, zdim, init_PD, tol_PD, init_PSD, tol_PSD, diag_PD, format_noise, sumsqr, split_list, demix, remix



# add full functions
include("em/taskswitch.jl")
export  save_results, preprocess_model,
        fit_SSID, fit_EM, load_SSID, save_SSID 

include("simulators/sim_lds.jl")
export generate_lds_trials, generate_lrnn_trials, generate_dlds_trials

include("em/EM.jl")
export  task_EM!, task_ESTEP!, task_MSTEP, 
        estimate_cov!, filter_cov!, filter_cov_KF!, smooth_cov!,
        estimate_mean!, filter_mean!, filter_mean_KF!, smooth_mean!,
        init_moments!, estimate_moments!, init_param_rand,
        total_loglik!, total_loglik, test_loglik!, test_loglik, test_orig_loglik, null_loglik!,
        posterior_estimates


include("em/init_params.jl")
export mf_clmoesp, init_n4sid, subspaceid_trial, subspaceid_orig, subspaceid_CVA, mi_QR, mi_hankel, n4sid_copy, task_refine_EM, task_refine_MSTEP, reflectd

include("em/utils.jl")
export  read_args, setup_dir, load_data, build_inputs, whiten_y, convert_bal, test_rep_ESTEP, generate_parameters, save_results

include("em/structs.jl")
export  core_struct, param_struct, data_struct, results_struct, estimates_struct, set_estimates, model_struct, post_struct,
        set_model, transform_model, transform_model_orth

include("em/make_plots.jl")
export  generate_PPC, plot_trial_pred, plot_avg_pred, plot_loglik_traces, 
        plot_mean_traj!, report_R2, report_params, plot_params, plot_bal_params, 
        plot_input_diffusion, plot_2input_diffusion









end