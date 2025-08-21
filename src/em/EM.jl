


function task_EM(S); 
    """
    run EM for individual participants
    """

    @reset S.res.startTime_em = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");

    # run tests
    if S.prm.max_iter_em > 100
        # run tests
        NeSS.run_tests(S)
    end


    # main EM loop ===================================================================
    for ii = 1:S.prm.max_iter_em

        # ==== E-STEP ================================================================
        @inline NeSS.task_ESTEP!(S);

        # ==== M-STEP ================================================================
        @reset S.mdl = deepcopy(NeSS.task_MSTEP(S));

        # ==== TOTAL LOGLIK ==========================================================
        NeSS.total_loglik!(S)
        

        # checks & cleaning ==========================================================

        # confirm loglik is increasing
        if (ii > 1)  && (S.res.total_loglik[ii] < S.res.total_loglik[ii-1])
            println("warning: total loglik decreased (Δll: $(round(S.res.total_loglik[end] - S.res.total_loglik[end-1],digits=3)))")
        end

        # test total loglik every N iters
        if mod(ii,S.prm.test_iter) == 0

            @reset S.est = deepcopy(set_estimates(S));
            NeSS.test_loglik!(S);
            push!(S.res.test_R2_white, LL_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    

            if length(S.res.test_loglik) > 1
                println("[$(ii)] total ll: $(round(S.res.total_loglik[ii],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)), Δll: $(round(S.res.total_loglik[end] - S.res.total_loglik[end-1],digits=2)) // test R2:$(round(S.res.test_R2_white[end],digits=4))")
            else
                println("[$(ii)] total ll: $(round(S.res.total_loglik[ii],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)), test R2:$(round(S.res.test_R2_white[end],digits=4))")
            end

        end

        # check for convergence
        if (length(S.res.test_loglik) > 1) &&
            (
                ((S.res.total_loglik[end] - S.res.total_loglik[end-1]) < 1) ||                        # training fit converged
                (S.prm.early_stop && ((S.res.test_loglik[end] - S.res.test_loglik[end-1]) < 1e-3))    # testing fit converged
            );

            println("")
            println("----- converged! -----")
            println("Δ total (training) loglik: $(S.res.total_loglik[end] - S.res.total_loglik[end-1])")
            println("Δ test loglik: $(S.res.test_loglik[end] - S.res.test_loglik[end-1])")
            println("")
            println("")
            break
        end

        # garbage collect every 5 iter
        if (mod(ii,5) == 0) && Sys.islinux() 
            ccall(:malloc_trim, Cvoid, (Cint,), 0);
            ccall(:malloc_trim, Int32, (Int32,), 0);
            GC.gc(true);
        end


    end


    # final test fit ===========================================================
    @reset S.est = deepcopy(set_estimates(S));        
    NeSS.test_loglik!(S);
    P = NeSS.posterior_sse(S, S.dat.y_test, S.dat.y_test_orig, S.dat.u_test, S.dat.u0_test);

    push!(S.res.test_R2_white, LL_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    
    push!(S.res.test_R2_orig, 1.0 - (P.sse_orig[1] / S.res.null_sse_orig[end]));
    
    @reset S.res.fwd_R2_white = 1.0 .- (P.sse_fwd_white ./ S.res.null_sse_white[1]);            
    @reset S.res.fwd_R2_orig = 1.0 .- (P.sse_fwd_orig ./ S.res.null_sse_orig[1]);

    push!(S.res.test_sse_white, P.sse_white[1]);    
    push!(S.res.test_sse_orig, P.sse_orig[1]);
    # ===========================================================

     
    

    println("[END] total ll: $(round(S.res.total_loglik[end],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)) // test R2: white:$(round(S.res.test_R2_white[end],digits=4)), orig:$(round(S.res.test_R2_orig[end],digits=4))")
    println("")


    @reset S.res.mdl_em = deepcopy(S.mdl);
    @reset S.res.endTime_em = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");



    return S

end



function run_tests(S)

    # check for initialized moments
    @assert ~all(S.est.xx_init .== 0) && ~all(S.est.yy_obs .== 0);

    # check that E-steps don't carry over
    @assert norm(test_rep_ESTEP(S)) ≈ 0.0;


    println("[passed all tests]\n")

end






# ===== E-STEP =================================================================



function task_ESTEP!(S)
    
    """
    run E-step for individual participants

    """

    # estimate latent covariance ==================
    @inline estimate_cov!(S);


    # initialize moments ==========================
    init_moments!(S);


    # estimate latent mean  ======================
    # init
    @inline estimate_mean!(S);   

end







# ===== ESTIMATE LATENT COVARIANCE =================================================================

function estimate_cov!(S)

    # filter cov ================================
    S.est.pred_cov[1] = deepcopy(S.mdl.P0);
    S.est.pred_icov[1] = deepcopy(S.mdl.iP0);
    S.est.filt_cov[1] = inv(S.mdl.CiRC + S.mdl.iP0); 

    filter_cov!(S);


    # smooth cov  ===============================
    S.est.smooth_xcov .= zeros(S.dat.x_dim, S.dat.x_dim);
    S.est.smooth_cov[end] = S.est.filt_cov[end];

    smooth_cov!(S);

end


function filter_cov!(S)

    # filter covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[2:end]

        S.est.pred_cov[tt] = PDMat(X_A_Xt(S.est.filt_cov[tt-1], S.mdl.A) + S.mdl.Q);
        S.est.pred_icov[tt] = inv(S.est.pred_cov[tt]);
        S.est.filt_cov[tt] = inv(S.mdl.CiRC + S.est.pred_icov[tt]);

    end
   
end


function filter_cov_KF!(S)
    # standard KF

    # filter covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[2:end]

        S.est.pred_cov[tt] = PDMat(X_A_Xt(S.est.filt_cov[tt-1], S.mdl.A) + S.mdl.Q);
        S.est.pred_icov[tt] = inv(S.est.pred_cov[tt]);

        S.est.K[:,:,tt] = S.est.pred_cov[tt]*S.mdl.C' / 
                                    tol_PD(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);


        S.est.filt_cov[tt] =  tol_PD(X_A_Xt(S.est.pred_cov[tt], I - S.est.K[:,:,tt]*S.mdl.C) .+ 
                                        X_A_Xt(S.mdl.R, S.est.K[:,:,tt]));

    end
   
end



@inline function smooth_cov!(S)



    # smooth covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[end-1:-1:1]

        # reverse kalman gain
        mul!(S.est.G[:,:,tt], S.est.filt_cov[tt], S.mdl.A', 1.0, 0.0);
        S.est.G[:,:,tt] /= S.est.pred_cov[tt+1];

        # smoothed covariancess
        mul!(S.est.xdim2_temp, S.est.G[:,:,tt], S.mdl.A, 1.0, 0.0);
        S.est.smooth_cov[tt] = PDMat(X_A_Xt(S.est.smooth_cov[tt+1] + S.mdl.Q, S.est.G[:,:,tt]) .+ 
                                     X_A_Xt(S.est.filt_cov[tt], I - S.est.xdim2_temp));

        # smoothed cross-cov
        mul!(S.est.smooth_xcov, S.est.G[:,:,tt], S.est.smooth_cov[tt+1], 1.0, 1.0);

    end

end






# ===== ESTIMATE LATENT MEAN =================================================================

function estimate_mean!(S)

    @inbounds @views for tl in axes(S.dat.y_train,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_train[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= S.dat.u_train[:,1:end-1,tl];
        S.est.u0_cur .= S.dat.u0_train[:,tl];
        mul!(S.est.Bu, S.mdl.B, S.dat.u_train[:,:,tl], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_train[:,:,tl], 1.0, 0.0);
        S.est.y_cur .= S.dat.y_train[:,:,tl];


        # filter mean ===================================
        mul!(S.est.xdim_temp, S.mdl.iP0, S.est.pred_mean[:,1], 1.0, 0.0);
        S.est.xdim_temp .+= S.est.CiRY[:,1];
        mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);


        # estimate moments ==============================
        mul!(S.est.xy_obs, S.est.smooth_mean, S.dat.y_train[:,:,tl]', 1.0, 1.0);

        estimate_moments!(S);

    end

    # format moments
    S.est.xx_dyn_PD[1] = tol_PD(S.est.xx_dyn);
    S.est.xx_obs_PD[1] = tol_PD(S.est.xx_obs);

end



@inline function filter_mean!(S)

    # filter mean [slow]
    @inbounds @views for tt in eachindex(S.est.pred_icov)[2:end]

        mul!(S.est.pred_mean[:,tt], S.mdl.A, S.est.filt_mean[:,tt-1], 1.0, 0.0);
        S.est.pred_mean[:,tt] .+= S.est.Bu[:,tt-1];

        mul!(S.est.xdim_temp, S.est.pred_icov[tt], S.est.pred_mean[:,tt], 1.0, 0.0);
        S.est.xdim_temp .+= S.est.CiRY[:,tt];

        mul!(S.est.filt_mean[:,tt], S.est.filt_cov[tt], S.est.xdim_temp, 1.0, 0.0);

    end


end





function filter_mean_KF!(S)

    # filter mean [slow]
    @inbounds @views for tt in eachindex(S.est.pred_icov)[2:end]

        mul!(S.est.pred_mean[:,tt], S.mdl.A, S.est.filt_mean[:,tt-1], 1.0, 0.0);
        S.est.pred_mean[:,tt] .+= S.est.Bu[:,tt-1];

        S.est.y_cur[:,tt] .-= S.mdl.C*S.est.pred_mean[:,tt]
        mul!(S.est.filt_mean[:,tt], S.est.K[:,:,tt], S.est.y_cur[:,tt], 1.0, 0.0);
        S.est.filt_mean[:,tt] .+= S.est.pred_mean[:,tt];

    end


end




@inline function smooth_mean!(S)

    # smooth mean
    @inbounds @views for tt in eachindex(S.est.pred_icov)[end-1:-1:1]

        S.est.xdim_temp .= S.est.smooth_mean[:,tt+1] .- S.est.pred_mean[:,tt+1];
        @inline mul!(S.est.smooth_mean[:,tt], S.est.G[:,:,tt], S.est.xdim_temp, 1.0, 0.0);
        S.est.smooth_mean[:,tt] .+= S.est.filt_mean[:,tt];

    end


end





# ===== ESTIMATE MODEL MOMENTS =================================================================

function init_moments!(S)


    # init ===============================================
    S.est.xy_init .= zeros(S.dat.u0_dim, S.dat.x_dim);
    S.est.yy_init .= S.est.smooth_cov[1] .* S.dat.n_train;
    S.est.n_init .= copy(S.dat.n_train);


    # dyn ===============================================
    S.est.xx_dyn .= zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim + S.dat.u_dim);
    S.est.xx_dyn[1:S.dat.x_dim,1:S.dat.x_dim] .= sum(S.est.smooth_cov[1:end-1]) .* S.dat.n_train;
    S.est.xx_dyn[(S.dat.x_dim+1):end, (S.dat.x_dim+1):end] .= copy(S.est.uu_dyn);
    
    S.est.xy_dyn .= zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim);
    S.est.xy_dyn[1:S.dat.x_dim,:] .= S.est.smooth_xcov*S.dat.n_train;

    S.est.yy_dyn .= sum(S.est.smooth_cov[2:end]) * S.dat.n_train;

    S.est.n_dyn .= (S.dat.n_times-1) * S.dat.n_train;


    # obs ===============================================
    S.est.xx_obs .= sum(S.est.smooth_cov) * S.dat.n_train;
    S.est.xy_obs .= zeros(S.dat.x_dim, S.dat.y_dim);
    S.est.n_obs .= S.dat.n_times * S.dat.n_train;


end


@views function estimate_moments!(S)
    
    
    # convienence variables =======================
    S.est.x_cur .= S.est.smooth_mean[:,1:end-1];
    S.est.x_next .= S.est.smooth_mean[:,2:end];


    # # initials moments =======================
    mul!(S.est.xy_init, S.est.u0_cur, S.est.x_cur[:,1]', 1.0, 1.0);
    mul!(S.est.yy_init, S.est.x_cur[:,1], S.est.x_cur[:,1]', 1.0, 1.0);


    # # dynamics moments =======================
    # # x_dyn * x_dyn
    mul!(S.est.xx_dyn[1:S.dat.x_dim,1:S.dat.x_dim], S.est.x_cur, S.est.x_cur', 1.0, 1.0);
    mul!(S.est.xx_dyn[1:S.dat.x_dim,(S.dat.x_dim+1):end], S.est.x_cur, S.est.u_cur', 1.0, 1.0);
    mul!(S.est.xx_dyn[(S.dat.x_dim+1):end, 1:S.dat.x_dim], S.est.u_cur, S.est.x_cur', 1.0, 1.0);

    # # x_dyn * y_dyn
    mul!(S.est.xy_dyn[1:S.dat.x_dim,:], S.est.x_cur, S.est.x_next', 1.0, 1.0);
    mul!(S.est.xy_dyn[S.dat.x_dim+1:end,:], S.est.u_cur, S.est.x_next', 1.0, 1.0);

    # # y_dyn * y_dyn
    mul!(S.est.yy_dyn, S.est.x_next, S.est.x_next', 1.0, 1.0);


    # # emissions moments =======================
    mul!(S.est.xx_obs, S.est.smooth_mean, S.est.smooth_mean', 1.0, 1.0);


end






# ===== M-STEP =================================================================

function task_MSTEP(S)::model_struct
    
    # initials ===============================================
    # Mean
    W = ((S.est.xx_init + S.prm.lam_B0) \ S.est.xy_init)';
    B0 = W[:, 1:S.dat.u0_dim]

    # Covariance
    Wxy = W*S.est.xy_init;
    P0e = (S.est.yy_init .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_init, W) .+ W*S.prm.lam_B0*W' + (S.prm.df_P0 * S.prm.mu_P0)) / 
            ((S.est.n_init[1] + S.prm.df_P0) - size(S.est.xx_init,1));


    P0 = format_noise(P0e, S.prm.P0_type);

    


    # latents ===============================================
    # Mean
    W = ((S.est.xx_dyn_PD[1] + S.prm.lam_AB) \ S.est.xy_dyn)';
    A = W[:, 1:S.dat.x_dim];
    B = W[:, (S.dat.x_dim+1):end];

    # Covariance
    Wxy = W*S.est.xy_dyn;
    Qe = (S.est.yy_dyn .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_dyn_PD[1], W) .+ W*S.prm.lam_AB*W' + (S.prm.df_Q * S.prm.mu_Q)) / 
        ((S.est.n_dyn[1] + S.prm.df_Q) - size(S.est.xx_dyn,1));

    Q = format_noise(Qe, S.prm.Q_type);




    # emissions ===============================================
    # Mean
    W = ((S.est.xx_obs_PD[1] + S.prm.lam_C) \ S.est.xy_obs)';
    C = deepcopy(W);

    # Covariance
    Wxy = W*S.est.xy_obs;
    Re = (S.est.yy_obs .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_obs_PD[1], W) .+ W*S.prm.lam_C*W' + (S.prm.df_R * S.prm.mu_R)) / 
            ((S.est.n_obs[1] + S.prm.df_R) - size(S.est.xx_obs,1));

    R = format_noise(Re, S.prm.R_type);



    # reconstruct model
    mdl = set_model(
        A = A,
        B = B,
        Q = Q,
        C = C,
        R = R,
        B0 = B0,
        P0 = P0,
        );


    return mdl

end




function init_param_rand(S)


    @reset S.mdl.A = randn(S.dat.x_dim, S.dat.x_dim);
    @reset S.mdl.B = randn(S.dat.x_dim, S.dat.u_dim);
    @reset S.mdl.Q = tol_PD(randn(S.dat.x_dim, S.dat.x_dim));

    @reset S.mdl.C = randn(S.dat.y_dim, S.dat.x_dim);
    @reset S.mdl.R = tol_PD(randn(S.dat.y_dim, S.dat.y_dim));

    @reset S.mdl.B0 = randn(S.dat.x_dim, S.dat.u0_dim);
    @reset S.mdl.P0 = tol_PD(randn(S.dat.x_dim, S.dat.x_dim));

    return S

end





# ===== LOGLIK =================================================================

log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) = -0.5*n*v*log(2pi) .+
                                            0.5*v*logdet(lam0) .+ 
                                            -0.5*v*logdet(lamN) .+
                                            0.5*v0*logdet(0.5 .* Sig0) .+
                                            -0.5*vN*logdet(0.5 .* SigN) .+
                                            SpecialFunctions.loggamma(0.5 .* v0) .+ 
                                            -SpecialFunctions.loggamma(0.5 .* vN);


log_post(n,v,vN,lam0,lamN,SigN) =  -0.5*n*v*log(2pi) .+
                                        0.5*v*logdet(lam0) .+ 
                                        -0.5*v*logdet(lamN) .+
                                        -0.5*vN*logdet(0.5 .* SigN) .+
                                        -SpecialFunctions.loggamma(0.5 .* vN);







function init_lik(S)

    n = S.est.n_init[1];
    v = S.dat.x_dim;
    p = S.dat.u0_dim;

    v0 = S.prm.df_P0;
    vN = v0 + (n - p);

    lam0 = S.prm.lam_B0(p);
    lamN = lam0 + S.est.xx_init;

    Sig0 = Matrix(S.prm.mu_P0(v) * v0);
    SigN = S.mdl.P0 * vN;

    if v0 > 0
        log_p = log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) 
    else
        log_p = log_post(n,v,vN,lam0,lamN,SigN) 
    end

    return log_p

end



function dyn_lik(S)

    n = S.est.n_dyn[1];
    v = S.dat.x_dim;
    p = S.dat.x_dim + S.dat.u_dim;

    v0 = S.prm.df_Q;
    vN = v0 + (n - p);

    lam0 = Matrix(S.prm.lam_AB(p));
    lamN = lam0 + S.est.xx_dyn;

    Sig0 = Matrix(S.prm.mu_Q(v) * v0);
    SigN = S.mdl.Q * vN;
    
    if v0 > 0
        log_p = log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) 
    else
        log_p = log_post(n,v,vN,lam0,lamN,SigN) 
    end

    return log_p

end



function obs_lik(S)

    n = S.est.n_obs[1];
    v = S.dat.y_dim;
    p = S.dat.x_dim;

    v0 = S.prm.df_R;
    vN = v0 + (n - p);

    lam0 = Matrix(S.prm.lam_C(p));
    lamN = lam0 + S.est.xx_obs;

    Sig0 = Matrix(S.prm.mu_R(v) * v0);
    SigN = S.mdl.R * vN;

    if v0 > 0
        log_p = log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) 
    else
        log_p = log_post(n,v,vN,lam0,lamN,SigN) 
    end

    return log_p

end




                  


function total_loglik!(S)

    # advance total loglik
    push!(S.res.init_loglik, 0.0);
    push!(S.res.dyn_loglik, 0.0);
    push!(S.res.obs_loglik, 0.0);
    push!(S.res.total_loglik, 0.0);

    # get logliks
    S.res.init_loglik[end] = init_lik(S);
    S.res.dyn_loglik[end] = dyn_lik(S);
    S.res.obs_loglik[end] = obs_lik(S);

    # total loglik
    S.res.total_loglik[end] = S.res.init_loglik[end] .+ S.res.dyn_loglik[end] .+ S.res.obs_loglik[end];

end

function total_loglik(S)

    total_loglik = 0.0;
    total_loglik += init_lik(S)
    total_loglik += dyn_lik(S)
    total_loglik += obs_lik(S)

    return total_loglik

end





function test_loglik!(S);

    # advance test loglik 
    push!(S.res.test_loglik, 0.0);
    len_test_loglik = length(S.res.test_loglik);

    # == filter covariance ==
    filter_cov!(S);

    # get mean & loglik
    @inbounds @views for tl in axes(S.dat.y_test,3)

        # set X0
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_test[:,tl], true , false);

        # transform data
        mul!(S.est.Bu, S.mdl.B, S.dat.u_test[:,:,tl]);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_test[:,:,tl]);

        # filter mean =========
        @inline filter_mean!(S);

        # get loglik
        mul!(S.est.test_mu, S.mdl.C, S.est.pred_mean, 1.0, 0.0);
        @inbounds @views for tt in axes(S.est.filt_mean,2)

            S.est.test_sigma[1] = PDMat(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);
            S.res.test_loglik[len_test_loglik] += logpdf(MvNormal(S.est.test_mu[:,tt], S.est.test_sigma[1]), S.dat.y_test[:,tt,tl]);

        end

    end

end





function test_loglik(S);

    # init test loglik 
    test_loglik = 0.0;

    # == filter covariance ==
    filter_cov!(S);

    # get mean & loglik
    @inbounds @views for tl in axes(S.dat.y_test,3)

        # set X0
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_test[:,tl], true , false);

        # transform data
        mul!(S.est.Bu, S.mdl.B, S.dat.u_test[:,:,tl], true , false);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_test[:,:,tl], true , false);

        # filter mean =========
        @inline filter_mean!(S);

        # get loglik
        mul!(S.est.test_mu, S.mdl.C, S.est.pred_mean, 1.0, 0.0);

        @inbounds @views for tt in axes(S.est.filt_mean,2)

            S.est.test_sigma[1] = tol_PD(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);
            test_loglik += logpdf(MvNormal(S.est.test_mu[:,tt], S.est.test_sigma[1]), S.dat.y_test[:,tt,tl]);

        end

    end

    return test_loglik

end





function test_orig_loglik(S);

    # init test loglik 
    test_loglik = 0.0;

    # == filter covariance ==
    filter_cov!(S);

    # get mean & loglik
    @inbounds @views for tl in axes(S.dat.y_test,3)

        # set X0
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_test[:,tl], true , false);

        # transform data
        mul!(S.est.Bu, S.mdl.B, S.dat.u_test[:,:,tl], true , false);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_test[:,:,tl], true , false);

        # filter mean =========
        @inline filter_mean!(S);

        # get loglik
        mul!(S.est.test_mu, S.mdl.C, S.est.pred_mean, 1.0, 0.0);
        @inbounds @views for tt in axes(S.est.filt_mean,2)

            S.est.test_sigma[1] = PDMat(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);
            test_loglik += logpdf(MvNormal(NeSS.remix(S, S.est.test_mu[:,tt]), X_A_Xt(S.est.test_sigma[1], S.dat.W)), S.dat.y_orig_test[:,tt,tl]);

        end

    end

    return test_loglik

end








function null_loglik!(S)

    """ Compute null log-likelihood
    args:
        y_train: training data
        y_test: test data
    return:
        cov_ll: predict from mean
        rel_ll: predict from time-resolved mean/cov
        diff_ll: predict using temporal difference
        ar1_ll: predict using AR1 regression
    """

    # long format y
    yl_train = reshape(permutedims(S.dat.y_train, (2,3,1)), S.dat.n_times*S.dat.n_train, S.dat.y_dim); # convert to long format (times x trials, channels)
    yl_test = reshape(permutedims(S.dat.y_test, (2,3,1)), S.dat.n_times*S.dat.n_test, S.dat.y_dim); # convert to long format (times x trials, channels)

    # average first timepoint
    yl1_train = mean(S.dat.y_train[:,1,:], dims=2)';
    yl1_test = mean(S.dat.y_train[:,1,:], dims=2)';

    # previous timepoints
    ylp_train = deepcopy(S.dat.y_train);
    ylp_train[:,2:end,:] .= ylp_train[:,1:end-1,:];
    ylp_train[:,1,:] .= yl1_train';
    ylp_train = reshape(permutedims(ylp_train, (2,3,1)), S.dat.n_times*S.dat.n_train, S.dat.y_dim);

    ylp_test = deepcopy(S.dat.y_test);
    ylp_test[:,2:end,:] .= ylp_test[:,1:end-1,:];
    ylp_test[:,1,:] .= yl1_test';
    ylp_test = reshape(permutedims(ylp_test, (2,3,1)), S.dat.n_times*S.dat.n_test, S.dat.y_dim);

    # previous inputs
    up_train = deepcopy(S.dat.u_train);
    up_train[:,2:end,:] .= up_train[:,1:end-1,:];
    up_train[:,1,:] .= 0.0;
    up_train = reshape(permutedims(up_train, (2,3,1)), S.dat.n_times*S.dat.n_train, S.dat.u_dim);

    up_test = deepcopy(S.dat.u_test);
    up_test[:,2:end,:] .= up_test[:,1:end-1,:];
    up_test[:,1,:] .= 0.0;
    up_test = reshape(permutedims(up_test, (2,3,1)), S.dat.n_times*S.dat.n_test, S.dat.u_dim);

    calc_b(x,y) = [x; 1e-3I(size(x,2))] \ [y; zeros(size(x,2), size(y,2))];
    calc_res(x,y,b) = y .- x*b;
    calc_cov(r,df) = (r'*r) ./ (size(r,1) - df);

    n_models = 4;
    pred_train = Vector{Array}(undef, n_models)
    pred_test = Vector{Array}(undef, n_models)

    # models
    pred_train[1] = ones(size(yl_train,1));
    pred_test[1] = ones(size(yl_test,1));

    pred_train[2] = [ones(size(yl_train,1)) ylp_train]; 
    pred_test[2] = [ones(size(yl_test,1)) ylp_test]; 

    pred_train[3] = [ones(size(yl_train,1)) up_train];
    pred_test[3] = [ones(size(yl_test,1)) up_test];

    pred_train[4] = [ones(size(yl_train,1)) ylp_train up_train];
    pred_test[4] = [ones(size(yl_test,1)) ylp_test up_test];


    # get test likelihood
    test_zeros = zeros(S.dat.y_dim);

    for mm = 1:n_models

        pred_res = calc_res(pred_test[mm], yl_test, calc_b(pred_train[mm], yl_train));
        pred_cov = tol_PD(calc_cov(pred_res, size(pred_train[mm],2)));

        null_ll = 0.0;
        for tt in axes(yl_test,1)
                null_ll += logpdf(MvNormal(pred_res[tt,:], pred_cov), test_zeros);
        end

        S.res.null_sse_white[mm] = sumsqr(pred_res);
        S.res.null_mse_white[mm] = sumsqr(pred_res)./length(pred_res);

        remix_res = NeSS.remix(S,pred_res');
        S.res.null_sse_orig[mm] = sumsqr(remix_res);
        S.res.null_mse_orig[mm] = sumsqr(remix_res)./length(remix_res);

        S.res.null_loglik[mm] = null_ll;

    end

end





# GENERATE POSTERIORS  ========================================================
function posterior_all(S, y, y_orig, u, u0)
    # all posteriors


    # initialize output ==========================
    n_trials = size(y,3);

    P = post_all(

            pred_mean = zeros(S.dat.x_dim, S.dat.n_times,n_trials),
            filt_mean = zeros(S.dat.x_dim, S.dat.n_times,n_trials),
            smooth_mean = zeros(S.dat.x_dim, S.dat.n_times,n_trials),

            pred_cov = [[init_PD(S.dat.x_dim) for _ in 1:S.dat.n_times] for _ in 1:n_trials],
            filt_cov = [[init_PD(S.dat.x_dim) for _ in 1:S.dat.n_times] for _ in 1:n_trials],
            smooth_cov = [[init_PD(S.dat.x_dim) for _ in 1:S.dat.n_times] for _ in 1:n_trials],

            obs_white_y = zeros(S.dat.y_dim, S.dat.n_times,n_trials),
            pred_white_y = zeros(S.dat.y_dim, S.dat.n_times,n_trials),
            filt_white_y = zeros(S.dat.y_dim, S.dat.n_times,n_trials),
            smooth_white_y = zeros(S.dat.y_dim, S.dat.n_times,n_trials),

            obs_orig_y = zeros(S.dat.n_chans, S.dat.n_times,n_trials),
            pred_orig_y = zeros(S.dat.n_chans, S.dat.n_times,n_trials),
            filt_orig_y = zeros(S.dat.n_chans, S.dat.n_times,n_trials),
            smooth_orig_y = zeros(S.dat.n_chans, S.dat.n_times,n_trials),

            sse_white = [0.0],
            sse_orig = [0.0],

        );


    # estimate cov ===================================
    estimate_cov!(S)



    # estimate mean ==================================
    # @inbounds @views for tl in axes(y,3)   
    @inbounds for tl in axes(y,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, u0[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= u[:,1:end-1,tl][:,:,1];
        S.est.u0_cur .= u0[:,tl][:,1];
        mul!(S.est.Bu, S.mdl.B, u[:,:,tl][:,:,1], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, y[:,:,tl][:,:,1], 1.0, 0.0);


        # filter mean ===================================
        S.est.xdim_temp .= S.est.CiRY[:,1] .+ S.mdl.iP0*S.est.pred_mean[:,1];
        @views mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);




        # save results ==================================
        P.pred_mean[:,:,tl] .= S.est.pred_mean;
        P.pred_cov[tl] .= S.est.pred_cov;

        P.filt_mean[:,:,tl] .= S.est.filt_mean;
        P.filt_cov[tl] .= S.est.filt_cov;

        P.smooth_mean[:,:,tl] .= S.est.smooth_mean;
        P.smooth_cov[tl] .= S.est.smooth_cov;

        P.obs_white_y[:,:,tl] .= y[:,:,tl];
        P.pred_white_y[:,:,tl] .= S.mdl.C * S.est.pred_mean;
        P.filt_white_y[:,:,tl] .= S.mdl.C * S.est.filt_mean;
        P.smooth_white_y[:,:,tl] .= S.mdl.C * S.est.smooth_mean;
        
        if S.dat.W == zeros(0,0)
            P.obs_orig_y[:,:,tl] .= P.obs_white_y[tl];
            P.pred_orig_y[:,:,tl] .= P.pred_white_y[tl];
            P.filt_orig_y[:,:,tl] .= P.filt_white_y[tl];
            P.smooth_orig_y[:,:,tl] .= P.smooth_white_y[tl];
        else
            P.obs_orig_y[:,:,tl] .= y_orig[:,:,tl];
            P.pred_orig_y[:,:,tl] .= NeSS.remix(S, S.mdl.C * S.est.pred_mean);
            P.filt_orig_y[:,:,tl] .= NeSS.remix(S, S.mdl.C * S.est.filt_mean);
            P.smooth_orig_y[:,:,tl] .= NeSS.remix(S, S.mdl.C * S.est.smooth_mean);
        end

        P.sse_white[1] += sumsqr(P.obs_white_y[tl] .-  P.pred_white_y[tl]);
        P.sse_orig[1] += sumsqr(P.obs_orig_y[tl].-  P.pred_orig_y[tl]);

        for ii in 0:25

            pred_y = S.mdl.C * S.mdl.A^(ii) * S.est.pred_mean[:, 1:end-ii];

            if ii >0
                for bb in 1:ii
                    pred_y .+= S.mdl.C * S.mdl.A^(bb-1) * S.mdl.B * S.dat.u_train[:, (ii-bb+1):end-bb, tl];
                end
            end

            P.sse_fwd_white[ii+1] += sumsqr(pred_y .-  P.obs_white_y[:, (ii+1):end,tl]);
            P.sse_fwd_orig[ii+1] += sumsqr(NeSS.remix(S, pred_y) .-  P.obs_orig_y[:, (ii+1):end,tl]);

        end

    end

    return P

end


function posterior_mean(S, y, y_orig, u, u0)
    # just posterior means

    # initialize output ==========================
    P = post_mean(
            pred_mean = zeros(S.dat.x_dim, S.dat.n_times, size(y,3)),
            filt_mean = zeros(S.dat.x_dim, S.dat.n_times, size(y,3)),
            smooth_mean = zeros(S.dat.x_dim, S.dat.n_times, size(y,3)),
        );


    # estimate cov ===================================
    estimate_cov!(S)


    # estimate mean ==================================
    # @inbounds @views for tl in axes(y,3)   
    @inbounds @views for tl in axes(y,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, u0[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= u[:,1:end-1,tl][:,:,1];
        S.est.u0_cur .= u0[:,tl][:,1];
        mul!(S.est.Bu, S.mdl.B, u[:,:,tl][:,:,1], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, y[:,:,tl][:,:,1], 1.0, 0.0);


        # filter mean ===================================
        S.est.xdim_temp .= S.est.CiRY[:,1] .+ S.mdl.iP0*S.est.pred_mean[:,1];
        mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);


        # save results ==================================
        P.pred_mean[:,:,tl] .= S.est.pred_mean;
        P.filt_mean[:,:,tl] .= S.est.filt_mean;
        P.smooth_mean[:,:,tl] .= S.est.smooth_mean;

    end

    return P

end




function posterior_sse(S, y, y_orig, u, u0)
    # just posteriors SSE

    # initialize output ==========================
    n_trials = size(y,3);

    P = post_sse(
            sse_white = [0.0],
            sse_orig = [0.0],
            sse_fwd_white = zeros(26),
            sse_fwd_orig = zeros(26),
        );


    # estimate cov ===================================
    estimate_cov!(S)



    # estimate mean ==================================
    # @inbounds @views for tl in axes(y,3)   
    @inbounds @views for tl in axes(y,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, u0[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= u[:,1:end-1,tl][:,:,1];
        S.est.u0_cur .= u0[:,tl][:,1];
        mul!(S.est.Bu, S.mdl.B, u[:,:,tl][:,:,1], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, y[:,:,tl][:,:,1], 1.0, 0.0);


        # filter mean ===================================
        S.est.xdim_temp .= S.est.CiRY[:,1] .+ S.mdl.iP0*S.est.pred_mean[:,1];
        mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);



        # SAVE RESULTS ==================================
        P.sse_white[1] += sumsqr(y[:,:,tl] .- S.mdl.C*S.est.pred_mean);
        P.sse_orig[1] += sumsqr(y_orig[:,:,tl] .-  NeSS.remix(S, S.mdl.C * S.est.pred_mean));

        for ii in 0:25

            pred_y = S.mdl.C * S.mdl.A^(ii) * S.est.pred_mean[:, 1:end-ii];

            if ii >0
                for bb in 1:ii
                    pred_y .+= S.mdl.C * S.mdl.A^(bb-1) * S.mdl.B * S.dat.u_train[:, (ii-bb+1):end-bb, tl];
                end
            end

            P.sse_fwd_white[ii+1] += sumsqr(pred_y .-  y[:,(ii+1):end,tl]);
            P.sse_fwd_orig[ii+1] += sumsqr(NeSS.remix(S, pred_y) .-  y_orig[:,(ii+1):end,tl]);

        end

    end

    return P

end


