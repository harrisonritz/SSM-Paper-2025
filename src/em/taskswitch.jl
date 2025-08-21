function preprocess_model(S)

    # READ ARGS ==============================================
    S = deepcopy(NeSS.read_args(S, ARGS));
    # =============================================================


    # LOAD DATA ==============================================
    # make directories
    NeSS.setup_dir(S)

    # load data, split into train & test
    S = deepcopy(NeSS.load_data(S));
    # =============================================================


    # BUILD INPUTS  ==============================================
    S = deepcopy(NeSS.build_inputs(S));
    # =============================================================


    # WHITEN DATA ==============================================
    S = deepcopy(NeSS.whiten_y(S));
    # =============================================================


    # NULL LOGLIKS ====================================
    NeSS.null_loglik!(S);
    #  =======================================================================


    # REPORT DATA ==============================================
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
    println("time inverval: $(minimum(S.dat.ts[S.dat.sel_times]))s to $(maximum(S.dat.ts[S.dat.sel_times]))s; timepoints: $(length(S.dat.ts[S.dat.sel_times]))")
    println("Q type: $(S.prm.Q_type) / R type: $(S.prm.R_type) / P0 type: $(S.prm.P0_type)")
    println("========================================\n")
    #  =======================================================================


    return S


end



function fit_SSID(S)


    # FIT SSID ==============================================
    println("\n\n\n\n========== FITTING SSID ==========")
    println("started SSID at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))");
    
    # if file already exists, kill the script
    ssid_file = "$(S.prm.save_dir)/../fit-results/SSID-jls/$(S.prm.model_name)/$(S.prm.save_name)_SSID.jls";
    if isfile(ssid_file)
        println("SSID file already exists: $(ssid_file)")
        println("exiting ...")
        exit(0);
    end


    # Subspace Identification (SSID) ==============================================
    @reset S = deepcopy(NeSS.init_ssid(S)); 
    # ================================================================

 
    println("finished SSID at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))");

    # print fit
    @reset S.est = deepcopy(set_estimates(S));
    NeSS.task_ESTEP!(S);
    @reset S.res.em_test_loglik = test_loglik(S);
    
    println("\n========== SSID FIT ==========")
    println("SSID test loglik: $(S.res.ssid_test_loglik)")
    NeSS.report_R2(S)
    println("========================================\n")

    # save R2
    P = NeSS.posterior_sse(S, S.dat.y_test, S.dat.y_test_orig, S.dat.u_test, S.dat.u0_test);
    @reset S.res.ssid_test_R2_white = 1.0 - (P.sse_white[1] / S.res.null_sse_white[end]);        
    @reset S.res.ssid_test_R2_orig = 1.0 - (P.sse_orig[1] / S.res.null_sse_orig[end]); 
    @reset S.res.ssid_fwd_R2_white = 1.0 .- (P.sse_fwd_white ./ S.res.null_sse_white[1]);        
    @reset S.res.ssid_fwd_R2_orig = 1.0 .- (P.sse_fwd_orig ./ S.res.null_sse_orig[1]);         
    
    
    
    if S.prm.ssid_save == 1
        println("saving SSID ..."); sleep(1);
        save_SSID(S); # save SSID
        exit(0); # exit
    end
    
    return S

end



function fit_EM(S)


    # FIT EM ==============================================
    println("\n\n\n\n========== FITTING EM ==========")
    println("started EM at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))");


    # if file already exists, kill the script
    em_file = "$(S.prm.save_dir)/../fit-results/EM-mat/$(S.prm.model_name)/$(S.prm.save_name).mat";
    if isfile(em_file)
        println("EM file already exists: $(em_file)")
        println("exiting ...")
        exit(0);
    end



    # Expectation Maximization (EM) ==============================================
    @reset S = deepcopy(NeSS.task_EM(S));
    # ================================================================



    println("finished EM at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))");

    # print fit
    @reset S.est = deepcopy(set_estimates(S));
    @reset S.res.em_test_loglik = test_loglik(S);
    
    println("\n========== EM FIT ==========")
    println("EM test loglik: $(S.res.em_test_loglik)")
    NeSS.report_R2(S)
    println("========================================\n")


    return S

end



function load_SSID(S)


    # LOAD SSID ==============================================
    println("\n\n\n\n========== LOADING SSID ==========")

    # sleep until file is found
    ssid_file = "$(S.prm.save_dir)/../fit-results/SSID-jls/$(S.prm.model_name)/$(S.prm.model_name)_Pt$(S.dat.pt)_xdim$(S.prm.ssid_lag)_SSID.jls";
    found_file = false

    println("searching for saved SSID file: $(ssid_file)\n")
    while ~found_file

        # check whether file exists
        file_stat = stat(ssid_file);
        (file_stat.size > 1024) ? found_file = true : nothing;
        
        # sleep for 10 seconds
        sleep(10)
        print(".")

    end

    # load file
    Sl = deserialize("$(S.prm.save_dir)/../fit-results/SSID-jls/$(S.prm.model_name)/$(S.prm.model_name)_Pt$(S.dat.pt)_xdim$(S.prm.ssid_lag)_SSID.jls");


    # take first x_dim dimensions of estimated parameters
    dim_sel = zeros(S.dat.x_dim, Sl.dat.x_dim)
    dim_sel[1:S.dat.x_dim, 1:S.dat.x_dim] .= I(S.dat.x_dim);

    # init parameters
    @reset S.mdl  = transform_model(Sl.mdl, dim_sel);

    # recover memory
    Sl = nothing
    if Sys.islinux() 
        ccall(:malloc_trim, Cvoid, (Cint,), 0);
        ccall(:malloc_trim, Int32, (Int32,), 0);
    end
    GC.gc(true);
   

    # SSID test loglik
    @reset S.est = deepcopy(set_estimates(S));
    @reset S.res.ssid_test_loglik = test_loglik(S);
    println("SSID test loglik: $(S.res.ssid_test_loglik)")

    # SSID test R2
    # @reset S.est = deepcopy(set_estimates(S));
    # NeSS.task_ESTEP!(S);
    NeSS.report_R2(S)
    println("========================================")

    # save R2
    P = NeSS.posterior_sse(S, S.dat.y_test, S.dat.y_test_orig, S.dat.u_test, S.dat.u0_test);
    @reset S.res.ssid_test_R2_white = 1.0 - (P.sse_white[1] / S.res.null_sse_white[end]);        
    @reset S.res.ssid_test_R2_orig = 1.0 - (P.sse_orig[1] / S.res.null_sse_orig[end]);       
    
    return S

end



function save_SSID(S)
    

    # CORE STRUCT (JLS) ========================================
    try
        serialize("$(S.prm.save_dir)/../fit-results/SSID-jls/$(S.prm.model_name)/$(S.prm.save_name)_SSID.jls", S)
    catch err
        println(err)
        println("COULD NOT SAVE JLS")
    end

end



    
function save_results(S)


        # run GC
        GC.gc(true);
    

        # CORE STRUCT (JLS) ========================================
        try
            serialize("$(S.prm.save_dir)/../fit-results/EM-jls/$(S.prm.model_name)/$(S.prm.save_name).jls", S)
        catch err
            println(err)
            println("COULD NOT SAVE JLS")
        end



        # CORE STRUCT (MAT) ========================================
        try
            write_matfile(  "$(S.prm.save_dir)/../fit-results/EM-mat/$(S.prm.model_name)/$(S.prm.save_name).mat", 
                            prm = S.prm, dat = S.dat, res = S.res, est = S.est, mdl = S.mdl);

        catch err
            println(err)            
            println("COULD NOT SAVE MAT")
        end

        
        ## POSTERIOR MEAN (TRAIN) ========================================
        P_train = NeSS.posterior_mean( 
            S, 
            S.dat.y_train,
            S.dat.y_train_orig,  
            S.dat.u_train, 
            S.dat.u0_train,
        );

        write_matfile(  "$(S.prm.save_dir)/../fit-results/PPC-mat/$(S.prm.model_name)_PPC/$(S.prm.save_name)_trainPPC.mat", 
            smooth_mean = P_train.smooth_mean,
            filt_mean = P_train.filt_mean,
            pred_mean = P_train.pred_mean,
        );
        P_train = nothing;





        ## POSTERIOR MEAN (TEST) ========================================
        P_test = NeSS.posterior_mean(
            S, 
            S.dat.y_test,
            S.dat.y_test_orig,  
            S.dat.u_test, 
            S.dat.u0_test,
        );

        write_matfile(  "$(S.prm.save_dir)/../fit-results/PPC-mat/$(S.prm.model_name)_PPC/$(S.prm.save_name)_testPPC.mat", 
            smooth_mean = P_test.smooth_mean,
            filt_mean = P_test.filt_mean,
            pred_mean = P_test.pred_mean,
        );
        P_test = nothing;


        
    end

