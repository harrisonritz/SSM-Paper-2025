


function information_filter_meancov(
    A::Matrix{Float64}, B::Matrix{Float64}, Q,
    C::Matrix{Float64}, R,
    X0, P0,
    y::Array{Float64}, u::Matrix{Float64};
    invCtR=[], CtRinvC=[], 
    pred_mean=[], pred_cov=[],
    filt_cov=[], invP1=[], invP=[])

   """ Information filter (forward algorithm): predictions  
   Args:
        A: state dynamics matrix of shape (T,x_dim,x_dim)
        B: input matrix of shape (T,x_dim,m)
        Q: state noise covariance of shape (T,x_dim,x_dim)
        C: observation matrix of shape (T,n,x_dim)
        R: observation noise covariance of shape (T,n,n)
        X0: initial state mean of shape (x_dim,)
        P0: initial state covariance of shape (x_dim,x_dim)
        y: observations
        u: inputs
    Returns:
        filt_mean: filtered means E[x_t | y_{1:t}]
        filt_cov: filtered covariances Cov[x_t | y_{1:t}]
        pred_mean: predicted means E[x_t | y_{1:t-1}]
        pred_cov: predicted covariances Cov[x_t | y_{1:t-1}]

    Adapated from pillowlab-simplelds
    """

    x_dim = size(A,1)
    n_times = size(y,2)

    # initialize
    # filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
    # filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

    # pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
    # pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

    # precompute

    compute_local = true

    invCtR = C'*inv(cholesky(R));
    CtRinvC = Hermitian(invCtR*C);

 
    Bu = B*u;
    invCtRY = invCtR*y;

    # inital timepoint
    filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]
    pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
    pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

    pred_mean[:,1]  .= X0;
    pred_cov[:,:,1] .= P0;

    invP1 = inv(cholesky(P0));

    @views filt_cov[:,:,1] .= inv(cholesky(CtRinvC + invP1));


    filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
    @views mul!(filt_mean[:,1], filt_cov[:,:,1], invCtRY[:,1] + invP1*pred_mean[:,1]);


    # future timepoints
    @inbounds @views for t in axes(filt_mean,2)[2:end]

        mul!(pred_mean[:,t], A, filt_mean[:,t-1]);
        pred_mean[:,t] .+= Bu[:,t-1];

        mul!(pred_cov[:,:,t], A, filt_cov[:,:,t-1]*A');
        pred_cov[:,:,t] += Q;
        
        invP = Hermitian(inv(cholesky(Hermitian(pred_cov[:,:,t]))));
        filt_cov[:,:,t] .= inv(cholesky(CtRinvC + invP));

        mul!(filt_mean[:,t], filt_cov[:,:,t], invCtRY[:,t] + invP*pred_mean[:,t]);

        

    end

    return filt_mean, filt_cov, pred_mean, pred_cov
end





function information_filter_meancov_precompute( A, B, Q,
                                                C, R,
                                                X0, P0,
                                                y, u,
                                                invCtR, CtRinvC, 
                                                pred_mean, pred_cov,
                                                filt_cov, invP1, invP)

   """ Information filter (forward algorithm): predictions  
   Args:
        A: state dynamics matrix of shape (T,x_dim,x_dim)
        B: input matrix of shape (T,x_dim,m)
        Q: state noise covariance of shape (T,x_dim,x_dim)
        C: observation matrix of shape (T,n,x_dim)
        R: observation noise covariance of shape (T,n,n)
        X0: initial state mean of shape (x_dim,)
        P0: initial state covariance of shape (x_dim,x_dim)
        y: observations
        u: inputs
    Returns:
        filt_mean: filtered means E[x_t | y_{1:t}]
        filt_cov: filtered covariances Cov[x_t | y_{1:t}]
        pred_mean: predicted means E[x_t | y_{1:t-1}]
        pred_cov: predicted covariances Cov[x_t | y_{1:t-1}]

    Adapated from pillowlab-simplelds
    """

    x_dim = size(A,1)
    n_times = size(y,2)

    # initialize
    # filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
    # filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

    # pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
    # pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

    # precompute
    Bu = B*u;
    invCtRY = invCtR*y;

    # inital timepoint
    filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
    @views mul!(filt_mean[:,1], filt_cov[:,:,1], invCtRY[:,1] + invP1*pred_mean[:,1]);


    # future timepoints
    @inbounds @views for t in axes(filt_mean,2)[2:end]

        mul!(pred_mean[:,t], A, filt_mean[:,t-1]);
        pred_mean[:,t] .+= Bu[:,t-1];

        mul!(filt_mean[:,t], filt_cov[:,:,t], invCtRY[:,t] .+ invP[:,:,t]*pred_mean[:,t]);

    end

    return filt_mean, pred_mean

end












function information_filter_loglik( A::Matrix{Float64}, B::Matrix{Float64}, Q,
                                    C::Matrix{Float64}, R,
                                    X0, P0,
                                    y::Array{Float64}, u::Matrix{Float64})

   """ Information filter (forward algorithm): LogLik  
   args:
        A: state dynamics matrix of shape (T,x_dim,x_dim)
        B: input matrix of shape (T,x_dim,m)
        Q: state noise covariance of shape (T,x_dim,x_dim)
        C: observation matrix of shape (T,n,x_dim)
        R: observation noise covariance of shape (T,n,n)
        X0: initial state mean of shape (x_dim,)
        P0: initial state covariance of shape (x_dim,x_dim)
        y: observations
        u: inputs
    Adapated from pillowlab-simplelds
    """

    x_dim = size(A,1);
    y_dim = size(C,1);
    u_dim = size(B,2);
     
    n_times = size(y,2);
    ll = 0.0;

    # initialize
    filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
    filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

    pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
    pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

    # initialize
    invCtR = C' * inv(cholesky(R));
    CtRinvC = Hermitian(invCtR * C)
    Bu = B * u;
    invCtRY = invCtR * y;


    # Run the information filter
    pred_mean[:,1] .= X0
    pred_cov[:,:,1] .= P0

    local invP1 = inv(cholesky(P0));

    @views filt_cov[:,:,1] .= inv(cholesky(CtRinvC + invP1));
    @views mul!(filt_mean[:,1], filt_cov[:,:,1], invCtR*y[:,1] + invP1*pred_mean[:,1]);

    @views local mu1 = C*X0[:,1];
    @views local sigma1 = SymPD((C * P0 * C') + R);
    ll += logpdf(MvNormal(mu1, sigma1), y[:,1]);

    @inbounds @views for tt in axes(filt_mean,2)[2:end]
       
        mul!(pred_cov[:,:,tt], A, filt_cov[:,:,tt-1]*A');
        pred_cov[:,:,tt] += Q;

        mul!(pred_mean[:,tt], A, filt_mean[:,tt-1]);
        pred_mean[:,tt] .+= Bu[:,tt-1];
        
        local invP = Hermitian(inv(cholesky(Hermitian(pred_cov[:,:,tt]))));
        filt_cov[:,:,tt] .= inv(cholesky(CtRinvC + invP));
        
        mul!(filt_mean[:,tt], filt_cov[:,:,tt], invCtRY[:,tt] + invP*pred_mean[:,tt]);

        # compute log-likelihood using the logpdf function
        @views local mu = C*pred_mean[:,tt];
        @views local sigma = SymPD(C * pred_cov[:,:,tt] * C' + R);
        ll += logpdf(MvNormal(mu, sigma), y[:,tt]);

    end

    return ll
end







function kalman_filter_gain(
    A::Matrix{Float64}, B::Matrix{Float64}, Q,
    C::Matrix{Float64}, R,
    X0, P0,
    y::Array{Float64}, u::Matrix{Float64})

   """ Kalman filter (forward algorithm): predictions  
   args:
        A: state dynamics matrix of shape (T,x_dim,x_dim)
        B: input matrix of shape (T,x_dim,m)
        Q: state noise covariance of shape (T,x_dim,x_dim)
        C: observation matrix of shape (T,n,x_dim)
        R: observation noise covariance of shape (T,n,n)
        X0: initial state mean of shape (x_dim,)
        P0: initial state covariance of shape (x_dim,x_dim)
        y: observations
        u: inputs
    Adapated from pillowlab-simplelds
    """

    x_dim = size(A,1);
    y_dim = size(C,1);
    u_dim = size(B,2);
     
    n_times = size(y,2);

    # initialize
    filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
    filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

    pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
    pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

    # initialize
    # predicted_means[:,1] = deepcopy(X0);
    # predicted_covariance[:,:,1] = deepcopy(P0);

    # CiR = C' * inv(cholesky(R));
    # CiRC = Hermitian(CiR*C);
    Bu = B*u;

    # Run the kalman filter
    pred_mean[:,1] .= X0;
    pred_cov[:,:,1] .= P0;

    local CVCR1 = herm_PD(C*P0*C' + R);
    local K1 = (P0 * C') / cholesky(CVCR1);
    filt_cov[:,:,1] = pred_cov[:,:,1] - K1*CVCR1*K1';
    # mul!(filt_mean[:,1], filt_cov[:,:,1], (pred_cov[:,:,1] \ pred_mean[:,1] + C'*(R \ y[:,1]))) mul!(filt_mean[:,1], K1, y[:,1] - C*X0);
    filt_mean[:,1] = pred_mean[:,1] + K1*(y[:,1] - C*pred_mean[:,1]);


    # pred_mean[:,1]  .= X0;
    # pred_cov[:,:,1] .= P0;

    # local invP1 = inv(cholesky(P0));

    # @views filt_cov[:,:,1] .= inv(cholesky(CiRC + invP1));
    # @views mul!(filt_mean[:,1], filt_cov[:,:,1], CiR*y[:,1] + invP1*pred_mean[:,1]);

    all_K = zeros(Float64, x_dim, y_dim, n_times);
    all_K[:,:,1] = K1;

    # CRY = C'*inv(cholesky(R))*y;


    @inbounds @views for tt in axes(filt_mean,2)[2:end]
       
        pred_mean[:,tt] = A*filt_mean[:,tt-1] + Bu[:,tt-1];
        pred_cov[:,:,tt] = herm_PD(A*filt_cov[:,:,tt-1]*A' + Q);



        CVCR = Sym(C*pred_cov[:,:,tt]*C' + R);
        K = (pred_cov[:,:,tt] * C') / cholesky(CVCR);
        filt_cov[:,:,tt] = Sym(pred_cov[:,:,tt] - K*CVCR*K');
        filt_mean[:,tt] = pred_mean[:,tt] + K*(y[:,tt] - C*pred_mean[:,tt]);


        # mul!(filt_mean[:,tt], K, y[:,tt] - C*pred_mean[:,tt]);
        # filt_mean[:,tt] += pred_mean[:,tt];



        # K = (pred_cov[:,:,tt] * C') / cholesky(Sym(C*pred_cov[:,:,tt]*C' + R));
        # # jos = I - K*C;
        # # filt_cov[:,:,tt] .= jos*pred_cov[:,:,tt]*jos' + K*R*K';

        # filt_cov[:,:,tt] = pred_cov[:,:,tt] - K*C*pred_cov[:,:,tt]
        # mul!(filt_mean[:,tt], filt_cov[:,:,tt], (pred_cov[:,:,tt] \ pred_mean[:,tt]) + CRY[:,tt])


        all_K[:,:,tt] = K;


    end

    return all_K, pred_mean, pred_cov

end















function rts_smoother_moments(  A::Matrix{Float64}, B::Matrix{Float64}, Q,
                                C::Matrix{Float64}, R,
                                X0, P0,
                                y::Array{Float64}, u::Matrix{Float64};
                                invCtR=[], CtRinvC=[], 
                                pred_mean=[], pred_cov=[],
                                filt_cov=[], invP1=[], invP=[],
                                G=[], smooth_cov=[], smooth_xcov=[])
    """ Kalman smoother (forward-backward algorithm) 
   args:
        A: state dynamics matrix of shape (T,x_dim,x_dim)
        B: input matrix of shape (T,x_dim,m)
        Q: state noise covariance of shape (T,x_dim,x_dim)
        C: observation matrix of shape (T,n,x_dim)
        R: observation noise covariance of shape (T,n,n)
        X0: initial state mean of shape (x_dim,)
        P0: initial state covariance of shape (x_dim,x_dim)
        y: observations
        u: inputs
    """

    # get shape of y
    x_dim = size(A,1);
    y_dim = size(C,1);
    u_dim = size(B,2);

    n_times = size(y,2);


    # preallocate return
    x_init = zeros(Float64, x_dim);
    xx_init = zeros(Float64, x_dim, x_dim);

    xx_dyn = zeros(Float64, x_dim+u_dim, x_dim+u_dim);
    xy_dyn = zeros(Float64, x_dim+u_dim, x_dim);
    yy_dyn = zeros(Float64, x_dim, x_dim);

    xx_obs = zeros(Float64, x_dim, x_dim);
    xy_obs = zeros(Float64, x_dim, y_dim);

    # get kalman estimates
    if isempty(CtRinvC)

        # compute_local = true;

        # @inline filt_mean, filt_cov, pred_mean, pred_cov  = information_filter_meancov(A, B, Q, C, R, X0, P0, y, u);

    else

        compute_local = false;

        @inline filt_mean, filt_cov, pred_mean, pred_cov  = information_filter_meancov_precompute(  A, B, Q, 
                                                                                                    C, R, 
                                                                                                    X0, P0, 
                                                                                                    y, u,
                                                                                                    invCtR, CtRinvC, 
                                                                                                    pred_mean, pred_cov,
                                                                                                    filt_cov, invP1, invP);
    end

    # intialize means and covariances
    # if compute_local

    #     # smoothed covariances Cov[x_t | y_{1:T}]
    #     smooth_cov = zeros(Float64, x_dim, x_dim, n_times) 
    #     smooth_cov[:,:,n_times] .= filt_cov[:, :, n_times];

    #     # smoothed cross-covariances Cov[x_t, x_{t+1} | y_{1:T}]
    #     smooth_xcov = zeros(Float64, x_dim, x_dim, n_times) 

    # end

    # smoothed means E[x_t | y_{1:T}]
    smooth_mean = zeros(Float64, x_dim, n_times) 
    smooth_mean[:,n_times] .= filt_mean[:, n_times];


    # loop backwards
    @inbounds @views for t in (n_times-1):-1:1

        # let's first get the reverse kalman gain
        if compute_local

            # G = filt_cov[:,:,tt]*A';
            # rdiv!(G, cholesky(Hermitian(pred_cov[:,:,tt+1])));

            # # compute smoothed covariances
            # mul!(smooth_cov[:,:,tt], G, (smooth_cov[:,:,tt+1] .- pred_cov[:,:,tt+1])*G');
            # smooth_cov[:,:,tt] .+= filt_cov[:,:,tt];
    
            # mul!(smooth_xcov[:,:,tt], G, smooth_cov[:,:,tt+1]);

            # # compute smoothed means
            # mul!(smooth_mean[:,tt], G, (smooth_mean[:,tt+1] .- pred_mean[:,tt+1]));
            # smooth_mean[:,tt] .+= filt_mean[:,tt];

        else

              # compute smoothed means
              mul!(smooth_mean[:,tt], G[:,:,tt], (smooth_mean[:,tt+1] .- pred_mean[:,tt+1]));
              smooth_mean[:,tt] .+= filt_mean[:,tt];

        end

    end



    # compute moments for EM
    x_past = smooth_mean[:,1:end-1]
    x_next = smooth_mean[:,2:end]
    u_past = u[:,1:end-1]


    # initials moments =======================
    x_init = smooth_mean[:,1];
    # xx_init = Sym(smooth_cov[:,:,1] + x_init*x_init'); 
    if compute_local
        # xx_init = (smooth_cov[:,:,1]); 
    else
        xx_init = [];
    end


    # dynamics moments =======================
    @views xx_dyn[1:x_dim,1:x_dim] = x_past*x_past' + sum(smooth_cov[:,:,1:end-1], dims=3)[:,:,1];
    @views xx_dyn[1:x_dim,(x_dim+1):end] = x_past*u_past';
    @views xx_dyn[(x_dim+1):end,1:x_dim] = u_past*x_past';
    @views xx_dyn[x_dim+1:end,x_dim+1:end] = u_past*u_past';

    @views xy_dyn[1:x_dim,:] = x_past*x_next' + sum(smooth_xcov, dims=3)[:,:,1];
    @views xy_dyn[x_dim+1:end,:] = u_past*x_next';

    @views yy_dyn = sum(smooth_cov[:,:,2:end], dims=3)[:,:,1] + x_next*x_next';


    # emissions moments =======================
    @views xx_obs = smooth_mean*smooth_mean' + sum(smooth_cov, dims=3)[:,:,1];
    @views mul!(xy_obs, smooth_mean, y');

    
    
    # return moments
    return  x_init, xx_init, 1, xx_dyn, xy_dyn, yy_dyn, n_times-1, xx_obs, xy_obs, n_times

end












function rts_smoother_moments_precompute(   A, B, Q,
                                            C, R,
                                            X0, P0,
                                            y, u,
                                            invCtR, CtRinvC, 
                                            pred_mean, pred_cov,
                                            filt_cov, invP1, invP,
                                            G, sum_cov_past, sum_cov_next, sum_cov_all, sum_xcov_all)
    """ Kalman smoother (forward-backward algorithm) 
    args:
    A: state dynamics matrix of shape (T,x_dim,x_dim)
    B: input matrix of shape (T,x_dim,m)
    Q: state noise covariance of shape (T,x_dim,x_dim)
    C: observation matrix of shape (T,n,x_dim)
    R: observation noise covariance of shape (T,n,n)
    X0: initial state mean of shape (x_dim,)
    P0: initial state covariance of shape (x_dim,x_dim)
    y: observations
    u: inputs
    """

    # get shape of y
    x_dim = size(A,1);
    y_dim = size(C,1);
    u_dim = size(B,2);

    n_times = size(y,2);


    # preallocate return
    x_init = zeros(Float64, x_dim);
    xx_init = zeros(Float64, x_dim, x_dim);

    xx_dyn = zeros(Float64, x_dim+u_dim, x_dim+u_dim);
    xy_dyn = zeros(Float64, x_dim+u_dim, x_dim);
    yy_dyn = zeros(Float64, x_dim, x_dim);

    xx_obs = zeros(Float64, x_dim, x_dim);
    xy_obs = zeros(Float64, x_dim, y_dim);


    @inline filt_mean, pred_mean = information_filter_meancov_precompute(   A, B, Q, 
                                                                            C, R, 
                                                                            X0, P0, 
                                                                            y, u,
                                                                            invCtR, CtRinvC, 
                                                                            pred_mean, pred_cov,
                                                                            filt_cov, invP1, invP);



    # smoothed means E[x_t | y_{1:T}]
    smooth_mean = zeros(Float64, x_dim, n_times) 
    smooth_mean[:,n_times] .= filt_mean[:, n_times];

    # loop backwards
    @inbounds @views for tt in (n_times-1):-1:1

        # compute smoothed means
        mul!(smooth_mean[:,tt], G[:,:,tt], (smooth_mean[:,tt+1] .- pred_mean[:,tt+1]));

        smooth_mean[:,tt] .+= filt_mean[:,tt];

    end



    # compute moments for M-step
    @inbounds @views x_past = copy(smooth_mean[:,1:end-1]);
    @inbounds @views x_next = copy(smooth_mean[:,2:end]);
    @inbounds @views u_past = copy(u[:,1:end-1]);

    # initials moments =======================
    @inbounds @views x_init .= smooth_mean[:,1];


    # dynamics moments =======================
    @inbounds @views mul!(xx_dyn[1:x_dim,1:x_dim], x_past, x_past');
    @inbounds @views xx_dyn[1:x_dim,1:x_dim] .+= sum_cov_past;
    @inbounds @views mul!(xx_dyn[1:x_dim,(x_dim+1):end], x_past, u_past');
    @inbounds @views mul!(xx_dyn[(x_dim+1):end, 1:x_dim], u_past, x_past');
    @inbounds @views mul!(xx_dyn[(x_dim+1):end, (x_dim+1):end], u_past, u_past');

    @inbounds @views mul!(xy_dyn[1:x_dim,:], x_past, x_next');
    @inbounds @views xy_dyn[1:x_dim,:] .+= sum_xcov_all;
    @inbounds @views mul!(xy_dyn[x_dim+1:end,:], u_past, x_next');

    @inbounds @views mul!(yy_dyn, x_next, x_next')
    @inbounds @views yy_dyn .+= sum_cov_next;


    # emissions moments =======================
    @inbounds @views mul!(xx_obs, smooth_mean, smooth_mean')
    @inbounds @views xx_obs .+= sum_cov_all;

    @inbounds @views mul!(xy_obs, smooth_mean, y');


    # return moments
    return  x_init, xx_init, 1, xx_dyn, xy_dyn, yy_dyn, n_times-1, xx_obs, xy_obs, n_times

end


function sum3(x)
    x_out = zeros(size(x, 1), size(x, 2))
    @inbounds @views x_out = dropdims(sum(x, dims=3), dims=3);
    return x_out
end


























function rts_smoother_meancov(
    A::Matrix{Float64}, B::Matrix{Float64}, Q,
    C::Matrix{Float64}, R,
    X0, P0,
    y::Array{Float64}, u::Matrix{Float64})
    """ Kalman smoother (forward-backward algorithm) 
   args:
        A: state dynamics matrix of shape (T,x_dim,x_dim)
        B: input matrix of shape (T,x_dim,m)
        Q: state noise covariance of shape (T,x_dim,x_dim)
        C: observation matrix of shape (T,n,x_dim)
        R: observation noise covariance of shape (T,n,n)
        X0: initial state mean of shape (x_dim,)
        P0: initial state covariance of shape (x_dim,x_dim)
        y: observations
        u: inputs
    """

    # get shape of y
    x_dim = size(A,1);
    y_dim = size(C,1);
    u_dim = size(B,2);
    
    n_times = size(y,2);


    # preallocate return

    # get kalman estimates
    @inline filt_mean, filt_cov, pred_mean, pred_cov  = information_filter_meancov(A, B, Q, C, R, X0, P0, y, u)

    # intialize means and covariances
    smooth_mean = zeros(Float64, x_dim, n_times) # smoothed means E[x_t | y_{1:T}]
    smooth_cov = zeros(Float64, x_dim, x_dim, n_times) # smoothed covariances Cov[x_t | y_{1:T}]
    smooth_xcov = zeros(Float64, x_dim, x_dim, n_times) # smoothed cross-covariances Cov[x_t, x_{t+1} | y_{1:T}]


    # initialize
    smooth_mean[:,n_times] .= filt_mean[:, n_times];
    smooth_cov[:,:,n_times] .= filt_cov[:, :, n_times];


    # loop backwards
    @inbounds for tt in (n_times-1):-1:1

        # let's first get the reverse kalman gain
        local G = filt_cov[:,:,tt]*A';
        rdiv!(G, cholesky(Hermitian(pred_cov[:,:,tt+1])));

        # compute smoothed means
        mul!(smooth_mean[:,tt], G, (smooth_mean[:,tt+1] .- pred_mean[:,tt+1]));
        smooth_mean[:,tt] .+= filt_mean[:,tt];

        # compute smoothed covariances
        mul!(smooth_cov[:,:,tt], G, (smooth_cov[:,:,tt+1] .- pred_cov[:,:,tt+1])*G');
        smooth_cov[:,:,tt] .+= filt_cov[:,:,tt];

        mul!(smooth_xcov[:,:,tt], G, smooth_cov[:,:,tt+1]);

    end

    # return moments
    return  smooth_mean, smooth_cov


end






function fit_linear_regression(xx::Hermitian, xy, yy::Hermitian, n_times, beta_lam, cov_mu, cov_df)
    # Solve a linear regression given sufficient statistics
    # bayesian notes:
    # B = (X'X + Lam0) \ (X'X + Lam0*Mu0)
    # Sigma = V0 + (Y-XB)'(Y-XB) + (B-Mu0)'*Lam0*(B-Mu0)

    # Regression weight
    W = (cholesky(xx + beta_lam*I) \ xy)' # ridge regression
    
    # Error Covariance
    Wxy = W*xy;
    Sigma = (yy - Wxy - Wxy' + W*xx*W' + beta_lam*W*W' + (cov_df * cov_mu)) / ((n_times + cov_df) - size(yy,1));   # without prior


    return W, Sigma

end












function compute_complete_data_ll(  A, B, Q, 
                                    C, R, 
                                    X0, P0, 
                                    x_init, xx_init, x_init_cov, n_init, 
                                    xx_dyn, xy_dyn, yy_dyn, n_dyn, 
                                    xx_obs, xy_obs, yy_obs, n_obs)::Float64
    """ computes the complete data LL for all trials"""

    x_dim = size(A,1);
    y_dim = size(B,1);
    


    # initials ================================
    if isa(P0, UniformScaling)
        P0c = P0(x_dim);
    else
        P0c = P0;
    end

    @show ll_initials =  -0.5.*tr(cholesky(P0c)\(xx_init + x_init_cov)) - 
                    0.5.*n_init.*logdet(2*pi*P0c);
    # ll_initials = 0;




    # latents ================================
    AB = cat(A, B, dims=2);

    if isa(Q, UniformScaling)
        Qc = Q(x_dim);
        Qinv = Q(x_dim);
    else
        Qc = Q;
        Qinv = inv(cholesky(Q));
    end


    @show ll_dynamics =  -0.5.*tr(Qinv*yy_dyn) - 
                    0.5.*tr(Qinv*A*xx_dyn[1:x_dim,1:x_dim]*A') - 
                    0.5.*tr(Qinv*B*xx_dyn[x_dim+1:end, x_dim+1:end]*B') + 
                    tr(Qinv*A*xy_dyn[1:x_dim,:]) + 
                    tr(Qinv*B*xy_dyn[x_dim+1:end,:]) - 
                    tr(Qinv*B*xx_dyn[x_dim+1:end, 1:x_dim]*A') - 
                    0.5.*n_dyn.*logdet(2*pi*Qc) -
                    0.5*1e-6*tr(AB*AB');





    # emissions ================================
    if isa(R, UniformScaling)
        Rc = R(y_dim);
        Rinv = R(y_dim);
    else
        Rc = R;
        Rinv = inv(cholesky(R));
    end

    # Rinv = inv(cholesky(R));
    @show ll_emission =   -0.5.*tr(Rinv*yy_obs) - 
                    0.5.*tr(Rinv*C*xx_obs*C') + 
                    tr(Rinv*C*xy_obs) - 
                    0.5.*n_obs.*logdet(2*pi*Rc) -
                    0.5*1e-6*tr(C*C');




    ll = ll_initials + ll_emission + ll_dynamics;

    return ll

end
















# function kalman_filter_meancov(
#     A::Matrix{Float64}, B::Matrix{Float64}, Q,
#     C::Matrix{Float64}, R,
#     X0, P0,
#     y::Array{Float64}, u::Matrix{Float64})

#    """ Kalman filter (forward algorithm): predictions  
#    args:
#         A: state dynamics matrix of shape (T,x_dim,x_dim)
#         B: input matrix of shape (T,x_dim,m)
#         Q: state noise covariance of shape (T,x_dim,x_dim)
#         C: observation matrix of shape (T,n,x_dim)
#         R: observation noise covariance of shape (T,n,n)
#         X0: initial state mean of shape (x_dim,)
#         P0: initial state covariance of shape (x_dim,x_dim)
#         y: observations
#         u: inputs
#     Adapated from pillowlab-simplelds
#     """

#     x_dim = size(A,1);
#     y_dim = size(C,1);
#     u_dim = size(B,2);
     
#     n_times = size(y,2);

#     # initialize
#     filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
#     filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

#     pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
#     pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

#     # initialize
#     # predicted_means[:,1] = deepcopy(X0);
#     # predicted_covariance[:,:,1] = deepcopy(P0);

#     # CiR = C' * inv(cholesky(R));
#     # CiRC = Hermitian(CiR*C);
#     Bu = B*u;

#     # Run the kalman filter
#     pred_mean[:,1] .= X0;
#     pred_cov[:,:,1] .= P0;

#     local CVCR1 = Hermitian(C*P0*C' + R);
#     local K1 = (P0 * C') / cholesky(CVCR1);
#     filt_cov[:,:,1] = pred_cov[:,:,1] - K1*CVCR1*K1';
#     # mul!(filt_mean[:,1], filt_cov[:,:,1], (pred_cov[:,:,1] \ pred_mean[:,1] + C'*(R \ y[:,1]))) mul!(filt_mean[:,1], K1, y[:,1] - C*X0);
#     filt_mean[:,1] += X0;


#     # pred_mean[:,1]  .= X0;
#     # pred_cov[:,:,1] .= P0;

#     # local invP1 = inv(cholesky(P0));

#     # @views filt_cov[:,:,1] .= inv(cholesky(CiRC + invP1));
#     # @views mul!(filt_mean[:,1], filt_cov[:,:,1], CiR*y[:,1] + invP1*pred_mean[:,1]);

#     @inbounds @views for t in axes(filt_mean,2)[2:end]
       
#         pred_cov[:,:,t] = A*filt_cov[:,:,t-1]*A' + Q;
#         pred_mean[:,t] = A*filt_mean[:,t-1] + Bu[:,t-1];

#         local CVCR = Hermitian(C*pred_cov[:,:,1]*C' + R);
#         local K = (pred_cov[:,:,1] * C') / cholesky(CVCR);
#         filt_cov[:,:,1] = pred_cov[:,:,1] - K*CVCR*K';

#         mul!(filt_mean[:,t], K, y[:,t] - C*pred_mean[:,t]);
#         filt_mean[:,t] += pred_mean[:,t];


#         # local K = (pred_cov[:,:,t] * C') / cholesky(Hermitian(C*pred_cov[:,:,t]*C' + R));
#         # local jos = I - K*C;
#         # filt_cov[:,:,t] .= jos*pred_cov[:,:,t]*jos' + K*R*K';

#         # filt_cov[:,:,t] = pred_cov[:,:,t] - K*C*pred_cov[:,:,t]
#         # mul!(filt_mean[:,t], filt_cov[:,:,t], (pred_cov[:,:,t] \ pred_mean[:,t] + C'*(R \ y[:,t])))

#     end

#     return filt_mean, filt_cov, pred_mean, pred_cov

# end





# function kalman_filter_matrix_loglik(
#     A::Matrix{Float64}, B::Matrix{Float64}, Q,
#     C::Matrix{Float64}, R,
#     X0, P0,
#     y::Array{Float64}, u::Matrix{Float64})

#    """ Information filter (forward algorithm): LogLik  
#    args:
#         A: state dynamics matrix of shape (T,x_dim,x_dim)
#         B: input matrix of shape (T,x_dim,m)
#         Q: state noise covariance of shape (T,x_dim,x_dim)
#         C: observation matrix of shape (T,n,x_dim)
#         R: observation noise covariance of shape (T,n,n)
#         X0: initial state mean of shape (x_dim,)
#         P0: initial state covariance of shape (x_dim,x_dim)
#         y: observations
#         u: inputs
#     Adapated from pillowlab-simplelds
#     """

#     x_dim = size(A,1);
#     y_dim = size(y,1);
#     u_dim = size(u,1);
     
#     n_times = size(y,2);
#     ll = 0.0;

#     # initialize
#     filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
#     filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

#     pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
#     pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

#     # initialize
#     invCtR = C' * inv(cholesky(R));
#     CtRinvC = Hermitian(invCtR * C)
#     Bu = B * u

#     # Run the information filter
#     pred_mean[:,1] .= X0
#     pred_cov[:,:,1] .= P0

#     local invP1 = inv(cholesky(P0));

#     @views filt_cov[:,:,1] .= inv(cholesky(CtRinvC + invP1));
#     @views mul!(filt_mean[:,1], filt_cov[:,:,1], invCtR*y[:,1] + invP1*pred_mean[:,1]);

#     @views local mu1 = C*X0[:,1];
#     @views local sigma1 = Hermitian((C * P0 * C') + R);
#     ll += logpdf(MvNormal(mu1, sigma1), y[:,1]);

#     @inbounds @views for t in axes(filt_mean,2)[2:end]
       
#         # pred_cov[:,:,t] =  A*filt_cov[:,:,t-1]*A' + Q;
#         mul!(pred_cov[:,:,t], A, filt_cov[:,:,t-1]*A');
#         pred_cov[:,:,t] += Q;

#         # pred_mean[:,t] = A*filt_mean[:,t-1] + Bu[:,t-1];
#         mul!(pred_mean[:,t], A, filt_mean[:,t-1]);
#         pred_mean[:,t] .+= Bu[:,t-1];
        
#         local invP = Hermitian(inv(cholesky(Hermitian(pred_cov[:,:,t]))));
#         filt_cov[:,:,t] .= inv(cholesky(CtRinvC + invP));
        
#         mul!(filt_mean[:,t], filt_cov[:,:,t], invCtR*y[:,t] + invP*pred_mean[:,t]);

#         # compute log-likelihood using the logpdf function
#         @views local mu = C*pred_mean[:,t];
#         @views local sigma = Hermitian(C * pred_cov[:,:,t] * C' + R);
#         ll += logpdf(MvNormal(mu, sigma), y[:,t]);

#     end

#     return ll
# end






















# function information_filter_meancov(
#     A::Matrix{Float64}, B::Matrix{Float64}, Q::UniformScaling,
#     C::Matrix{Float64}, R,
#     X0, P0,
#     y::Array{Float64}, u::Matrix{Float64})

#    """ Information filter (forward algorithm): predictions  
#    args:
#         A: state dynamics matrix of shape (T,x_dim,x_dim)
#         B: input matrix of shape (T,x_dim,m)
#         Q: state noise covariance of shape (T,x_dim,x_dim)
#         C: observation matrix of shape (T,n,x_dim)
#         R: observation noise covariance of shape (T,n,n)
#         X0: initial state mean of shape (x_dim,)
#         P0: initial state covariance of shape (x_dim,x_dim)
#         y: observations
#         u: inputs
#     Adapated from pillowlab-simplelds
#     """

#     x_dim = size(A,1);
#     # y_dim = size(y,1);
#     # u_dim = size(u,1);
#     n_times = size(y,2);

#     # initialize
#     filt_mean = zeros(Float64, x_dim, n_times) # filtered means E[x_t | y_{1:t}]
#     filt_cov = zeros(Float64, x_dim, x_dim, n_times) # filtered covariances Cov[x_t | y_{1:t}]

#     pred_mean = zeros(Float64, x_dim, n_times) # predicted means E[x_t | y_{1:t-1}]
#     pred_cov = zeros(Float64, x_dim, x_dim, n_times) # predicted covariances Cov[x_t | y_{1:t-1}]

#     # initialize
#     CtRinv = C'/R;
#     CtRinvC = Hermitian(CtRinv*C);
#     Bu = B*u;

#     # Run the kalman filter
#     pred_mean[:,1] = X0;
#     pred_cov[:,:,1] = P0;

#     @views invP = inv(cholesky(pred_cov[:,:,1]));
#     @views filt_cov[:,:,1] = inv(cholesky(CtRinvC + invP));  
#     @views mul!(filt_mean[:,1], filt_cov[:,:,1], (CtRinv*y[:,1] + invP*pred_mean[:,1]));

#     @inbounds for t in axes(filt_mean,2)[2:end]
       
#         @views pred_cov[:,:,t] = Hermitian(A*filt_cov[:,:,t-1]*A' + Q);
#         @views pred_mean[:,t] = A*filt_mean[:,t-1] + Bu[:,t];

#         @views invP = inv(cholesky(pred_cov[:,:,t]))
#         @views filt_cov[:,:,t] = inv(cholesky(CtRinvC + invP));   # KF cov for time bin t
#         @views mul!(filt_mean[:,t], filt_cov[:,:,t], (CtRinv*y[:,t] + invP*pred_mean[:,t])); # KF mean

#     end

#     return filt_mean, filt_cov, pred_mean, pred_cov

# end






# function rts_smoother_moments(
#     A::Matrix{Float64}, B::Matrix{Float64}, Q::UniformScaling,
#     C::Matrix{Float64}, R,
#     X0, P0,
#     y::Array{Float64}, u::Matrix{Float64})
#     """ Kalman smoother (forward-backward algorithm) 
#    args:
#         A: state dynamics matrix of shape (T,x_dim,x_dim)
#         B: input matrix of shape (T,x_dim,m)
#         Q: state noise covariance of shape (T,x_dim,x_dim)
#         C: observation matrix of shape (T,n,x_dim)
#         R: observation noise covariance of shape (T,n,n)
#         X0: initial state mean of shape (x_dim,)
#         P0: initial state covariance of shape (x_dim,x_dim)
#         y: observations
#         u: inputs
#     """

#     # get shape of y
#     x_dim = size(A,1);
#     y_dim = size(y,1);
#     u_dim = size(u,1);
     
#     n_times = size(y,2);


#     # preallocate return
#     x_init = zeros(Float64, x_dim);
#     xx_init = zeros(Float64, x_dim, x_dim);

#     xx_dyn = zeros(Float64, x_dim+u_dim, x_dim+u_dim);
#     xy_dyn = zeros(Float64, x_dim+u_dim, x_dim);
#     yy_dyn = zeros(Float64, x_dim, x_dim);

#     xx_obs = zeros(Float64, x_dim, x_dim);
#     xy_obs = zeros(Float64, x_dim, y_dim);

#     # get kalman estimates
#     # if x_dim > y_dim
#     #     @inline filt_mean, filt_cov, pred_mean, pred_cov  = kalman_filter_meancov(A, B, Q, C, R, X0, P0, y, u)
#     # else
#     @inline filt_mean, filt_cov, pred_mean, pred_cov  = information_filter_meancov(A, B, Q, C, R, X0, P0, y, u)
#     # end

#     # intialize means and covariances
#     smooth_mean = zeros(x_dim, n_times) # smoothed means E[x_t | y_{1:T}]
#     smooth_cov = zeros(x_dim, x_dim, n_times) # smoothed covariances Cov[x_t | y_{1:T}]
#     smooth_xcov = zeros(x_dim, x_dim, n_times) # smoothed cross-covariances Cov[x_t, x_{t+1} | y_{1:T}]


#     # initialize
#     smooth_mean[:,n_times] = filt_mean[:, n_times];
#     smooth_cov[:,:,n_times] = filt_cov[:, :, n_times];

#     G = zeros(x_dim, x_dim) # reverse gain


#     # loop backwards
#     @inbounds for t in (n_times-1):-1:1

#         # let's first get the reverse kalman gain
#         @views G = (filt_cov[:,:,t]*A') / cholesky(pred_cov[:,:,t+1]);

#         # compute smoothed means
#         @views smooth_mean[:,t] = filt_mean[:,t] + G*(smooth_mean[:,t+1] - pred_mean[:,t+1]);

#         # compute smoothed covariances
#         @views smooth_cov[:,:,t] = Hermitian(filt_cov[:,:,t] + G*(smooth_cov[:,:,t+1] - pred_cov[:,:,t+1])*G');
#         @views mul!(smooth_xcov[:,:,t], G, smooth_cov[:,:,t+1]);

#     end



#     # compute moments for EM
#     x_past = smooth_mean[:,1:end-1]
#     x_next = smooth_mean[:,2:end]
#     u_past = u[:,1:end-1]


#     # initials moments =======================
#     x_init = smooth_mean[:,1];
#     xx_init = smooth_cov[:,:,1] + x_init*x_init'; 


#     # dynamics moments =======================
#     xx_dyn = Hermitian([x_past*x_past' + sum(smooth_cov[:,:,1:end-1], dims=3)[:,:,1]    x_past*u_past'
#                         u_past*x_past'                                                  u_past*u_past' ]);

#     xy_dyn = [x_past*x_next' + sum(smooth_xcov, dims=3)[:,:,1]; u_past*x_next'];

#     yy_dyn = Hermitian(sum(smooth_cov[:,:,2:end], dims=3)[:,:,1] + x_next*x_next');


#     # emissions moments =======================
#     xx_obs = Hermitian(smooth_mean*smooth_mean' + sum(smooth_cov, dims=3)[:,:,1]);
    
#     mul!(xy_obs, smooth_mean, y');


    
#     # return moments
#     return  x_init, xx_init, 1, xx_dyn, xy_dyn, yy_dyn, n_times-1, xx_obs, xy_obs, n_times

# end



