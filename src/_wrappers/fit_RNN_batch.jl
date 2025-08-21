# new recover parameters


# imports
if Sys.isapple()
    NeSS_dir = "/Users/hr0283/Brown Dropbox/Harrison Ritz/HallM_NeSS/src"
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
using PDMats
using MAT
using MATLAB




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





# SET PARAMETERS ==============================================================
Random.seed!(99)


splines_step = 4;
basis_name = "bspline";
x_dim = 112;
ssid_lag = 112;
pca_pct = 0.99;
save_SSID = false;


# process arguements
println("ARGS: $(ARGS)")
if Sys.isapple()
    load_dir="NET-GRU_nsingle-0_nswitch-150_nreps-125_ntrials-3__TASK-iti-20_cuegain-100"
    save_id="test";
    model_range = 9;
else
    
    load_dir = ARGS[1];                     #"2024-08-20__FIT-GRU_init-xavier__TASK-iti-0";
    save_id = ARGS[2];                      #"GRU-xavier_ITI-0__bspline$(splines_step)_pca$( round(Int64, pca_pct*100) )";
    model_range = parse(Int64, ARGS[3]);

    if length(ARGS) > 3

        x_dim = parse(Int64, ARGS[4]);
        ssid_lag = 112;
        println("\n\n\n=========== setting x_dim to $(x_dim) ===========\n\n\n")

    end
    
end

trial = 2
if occursin("trial3", save_id)
    trial = 2
elseif occursin("trial2", save_id)
    trial = 1
end



println("========== PARAMS ==========")
println("spline step: $(splines_step)")
println("basis name: $(basis_name)")
println("x dim: $(x_dim)")
println("pca pct: $(pca_pct)")
println("model range: $(model_range)")
println("load dir: $(load_dir)")
println("save id: $(save_id)")
println("========================================\n")





# RUN FITTING ==============================================================
for mm = model_range


    println("\n\n\n\n\n ============ MODEL: $(save_id) -- $(mm) ============ \n\n\n\n\n")
 

    # load SSM
    if Sys.isapple()
        dat = matread("$(save_dir)/../della-outputs/_RNN/$(load_dir)_ssm/$(load_dir)__$(mm).mat")
    else
        dat = matread("$(save_dir)/../RNN_TS/task-optimized/results/ssm/$(load_dir)_ssm/$(load_dir)__$(mm).mat")
    end
    dat_latents = permutedims(convert(Array{Float64}, dat["sim_latents"]), (3,2,1));
    dat_inputs = permutedims(convert(Array{Float64}, dat["sim_inputs"]), (3,2,1));
    dat_conds =  dat["sim_conditions"];

    println("loaded data: latents: $(size(dat_latents)), inputs: $(size(dat_inputs))")


    # get time
    time_sel = dat_conds["ssm_epoch$(trial)"];
    n_times = sum(Int64, time_sel);
    println("epoch duration = $n_times")
        

    train_sel = randperm!(collect(1:1536));
    test_sel = randperm!(collect(1537:2048));
    # train_sel = collect(1:512);
    # test_sel = collect(513:768);
    n_trials = 2048;

    
    
    # check timing
    # time_hot = zeros(size(dat_inputs,2));
    # time_hot[time_sel] .= 1;
    # plot(dat_inputs[:,:,1]', label=false)
    # plot!(time_hot, color="black", label=false)

    # set up conditions
    pred_cond = [dat_conds["ssm_taskSwitch$(trial)"][1:n_trials] dat_conds["ssm_taskRepeat$(trial)"][1:n_trials] dat_conds["ssm_switch$(trial)"][1:n_trials]]';
    pred_list = ["taskSwitch", "taskRepeat", "switch"];
    n_pred = size(pred_cond,1);


    # build basis set
    if basis_name == "bspline"
        n_bases = round(Int64, n_times/splines_step);
        basis = averagebasis(4, LinRange(1, n_times, n_bases));
    elseif basis_name == "cue"
        n_bases = 1;
    end

    pred_basis = [basis_name for _ in 1:n_bases];
    u_dim = n_bases*(n_pred + 1);
    u = zeros(u_dim, n_times, n_trials); 


    for tt in axes(u,2)
        bs = bsplines(basis, tt);
        u[collect(axes(bs,1)),tt,:] .= collect(bs);
    end

    println("BSPLINE: n bases: $(n_bases), breakpoints: $(round.(breakpoints(basis),sigdigits=4))")


    # convolve predictors with basis set
    for bb = 1:n_bases
        for uu in axes(pred_cond,1)
            u[(n_bases) + ((uu-1)*n_bases)+bb,:,:] .= u[bb,:,:] .* pred_cond[uu,:]';
        end
    end

    pred_name = [pred_basis; vec(repeat(pred_list, inner=(n_bases,1)))];
    pred0_name = ["init", "prevTask"];



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
    u0[2,:] .= dat_conds["ssm_task$(trial)"][1:n_trials].*dat_conds["ssm_switch$(trial)"][1:n_trials]; # constant term



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

    y_train_orig = dat_latents[:,vec(time_sel.==1),train_sel];
    y_test_orig = dat_latents[:,vec(time_sel.==1),test_sel];





    # BUILD STRUCT ==============================================================
    S = core_struct(
        prm=param_struct(
            model_name = save_id,

            do_save = true,

            pt_list = 1:1,
            
            max_iter_em = ~Sys.isapple() ? 2e4 : 2000,
            test_iter = 10,
            early_stop = true,

            y_transform = "PCA",
            PCA_ratio = pca_pct,

            NeSS_dir = NeSS_dir,
            save_dir = save_dir,

            ssid_fit = "fit",

            ssid_lag = ssid_lag,
            ), 

        dat=data_struct(
            x_dim = x_dim,
            n_train = n_train,
            n_test =  n_test,
            n_times = n_times,
            u_dim = u_dim,
            u0_dim = u0_dim,
            n_chans = n_chans,
            pred_name = pred_name,
            pred0_name = pred0_name,
            pred_list = pred_list,
            n_bases = n_bases,
            basis_name = basis_name,
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
    @reset S.prm.save_name = "$(save_id)_Pt$(mm)_xdim$(S.dat.x_dim)";
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
    println("basis name: $(S.dat.basis_name)\n")
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

        # save SSID
        if save_SSID
            serialize("$(S.prm.save_dir)/../fit-results/SSID-jls/$(S.prm.model_name)/$(S.prm.save_name)_SSID.jls", S)
        end

    elseif S.prm.ssid_fit == "load"

        # load previously-fit SSID
        @reset S = NeSS.load_SSID(S);

    end

    
    # ================================================






    # Expectation Maximization (EM) ==================
    @reset S = NeSS.fit_EM(S);
    # ================================================






    # try

    #     # plot loglik traces
    #     NeSS.plot_loglik_traces(S)

            
    #     # plot posterior predictive checks
    #     NeSS.plot_trial_pred(S, 20)
        
    #     NeSS.plot_avg_pred(S)


    #     # plot model
    #     NeSS.plot_params(S)


    #     # plot RT diffusion
    #     NeSS.plot_2input_diffusion(S; input1_name = "taskSwitch",  input2_name = "taskRepeat", norm_type="none")



    
    # catch
    # end








    # SAVE FIT =======================================================================
    if S.prm.do_save

        println("\n========== SAVING FIT ==========")

        save_results(S);

    else
        println("\n========== *NOT* SAVING FIT ==========")
    end

    println("=================================\n")

    #  =======================================================================



end