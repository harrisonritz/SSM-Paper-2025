# fit setup functions


function read_args(S, ARGS)
    
    # load arguements from command line
    println("ARGS: $ARGS")

    if isempty(ARGS)

        arg_num = 0;
        do_fast = true;

    else

        arg_num = parse(Int64, ARGS[1]);
        do_fast = parse(Bool, ARGS[2]); 

    end

    # get conditions
    if do_fast
        arg_list = [collect(S.prm.pt_list), S.prm.x_dim_fast];
    else
        arg_list = [collect(S.prm.pt_list), S.prm.x_dim_slow];
    end


    # get conditions
    if arg_num > 0

        indices = []
        for entry in arg_list
            push!(indices, mod(arg_num-1, length(entry)) + 1)
            arg_num = div(arg_num-1, length(entry)) + 1
        end
        conds = [entry[index] for (entry, index) in zip(arg_list, indices)];

        # assign conditions
        @reset S.dat.pt = conds[1];
        @reset S.dat.x_dim = conds[2];
        @reset S.prm.arg_num = parse(Int64, ARGS[1]);

    end

    if S.prm.ssid_save == 1
        @reset S.dat.x_dim = S.prm.ssid_lag;
    end
    if S.prm.ssid_lag == -1
        @reset S.prm.ssid_lag = S.dat.x_dim;
    end

    if do_fast
        println("FAST=[$(minimum(S.prm.x_dim_fast))-$(maximum(S.prm.x_dim_fast))], CONDITION=$(S.prm.arg_num)/$(prod(length.(arg_list))): pt=$(S.dat.pt), x_dim=$(S.dat.x_dim)")
    else
        println("SLOW=[$(minimum(S.prm.x_dim_slow))-$(maximum(S.prm.x_dim_slow))], CONDITION=$(S.prm.arg_num)/$(prod(length.(arg_list))): pt=$(S.dat.pt), x_dim=$(S.dat.x_dim)")
    end

   
    @reset S.prm.save_name = "$(S.prm.model_name)_Pt$(S.dat.pt)_xdim$(S.dat.x_dim)";
    @reset S.prm.do_fast = do_fast;

    return S

end




function setup_dir(S)

    # make output folders
    mkpath("$(S.prm.save_dir)/../figures/$(S.prm.model_name)")
    mkpath("$(S.prm.save_dir)/../fit-results/SSID-jls/$(S.prm.model_name)")
    mkpath("$(S.prm.save_dir)/../fit-results/EM-jls/$(S.prm.model_name)")
    mkpath("$(S.prm.save_dir)/../fit-results/EM-mat/$(S.prm.model_name)")
    mkpath("$(S.prm.save_dir)/../fit-results/PPC-mat/$(S.prm.model_name)_PPC")

end





function load_data(S)
    # load data from file

    # load data
    raw_data = matread("$(S.prm.save_dir)/../data/NeSS-formatted/$(S.prm.filename)/$(S.prm.filename)_$(S.dat.pt).mat")

    # get basic info
    if S.prm.decimate
        println("decimating data")
        raw_data["y"] = raw_data["y"][:,1:2:end,:];
        @reset S.dat.ts = vec(raw_data["ts"][1:2:end]);
        @reset S.dat.dt = raw_data["dt"]*2;
        @reset S.dat.epoch = vec(raw_data["epoch"][1:2:end]);
    else    
        @reset S.dat.ts = vec(raw_data["ts"]);
        @reset S.dat.dt = raw_data["dt"];
        @reset S.dat.epoch = vec(raw_data["epoch"]);
    end
    @reset S.dat.n_chans = size(raw_data["y"],1);
    @reset S.dat.n_trials = size(raw_data["y"],3);


    @reset S.dat.trial = raw_data["trial"];
    try
        @reset S.dat.chanLocs = raw_data["chanLocs"];
    catch
        println("couldn't load chanLocs")
    end

    # split test and train (median block)
    unique_blocks = unique(S.dat.trial["block"][:,1]);
    test_block = unique_blocks[round(Int64, length(unique_blocks)/2)];

    if S.prm.do_trial_sel
        println("custom trial sel")
        @reset S.dat.sel_trial = vec(S.dat.trial["acc"] .== 1) .& vec(S.dat.trial["prevAcc"] .== 1) .& vec(isfinite.(S.dat.trial["RT"])) .& vec(isfinite.(S.dat.trial["block"])); # current and previous accurate
    else
        println("custom trial sel")
        @reset S.dat.sel_trial = vec(isfinite.(S.dat.trial["RT"])) .& vec(isfinite.(S.dat.trial["block"]));
    end


    @reset S.dat.sel_train = S.dat.sel_trial .& vec(S.dat.trial["block"]  .!= test_block);
    @reset S.dat.sel_test = S.dat.sel_trial .& vec(S.dat.trial["block"] .== test_block);
    @reset S.dat.sel_times = vec(any(in.(S.dat.epoch', S.dat.epoch_sel),dims=1));
    @reset S.dat.n_times = sum(S.dat.sel_times);
    @reset S.dat.epoch = S.dat.epoch[S.dat.sel_times];


    # setup EEG data
    # train
    if S.prm.y_baseline
        println("baselining y with first timepoint")
        @reset S.dat.y_train_orig = raw_data["y"][:,S.dat.sel_times,S.dat.sel_train] .- permutedims(raw_data["y"][:,findfirst(S.dat.sel_times),S.dat.sel_train][:,:,:], (1,3,2));
    else
       @reset S.dat.y_train_orig = raw_data["y"][:,S.dat.sel_times,S.dat.sel_train];
    end
    @reset S.dat.n_train = size(S.dat.y_train_orig,3);

    # test
    if S.prm.y_baseline
        @reset S.dat.y_test_orig = raw_data["y"][:,S.dat.sel_times,S.dat.sel_test] .- permutedims(raw_data["y"][:,findfirst(S.dat.sel_times),S.dat.sel_test][:,:,:], (1,3,2));
    else
        @reset S.dat.y_test_orig =raw_data["y"][:,S.dat.sel_times,S.dat.sel_test];
    end
    @reset S.dat.n_test = size(S.dat.y_test_orig,3);

    # setup predictors
    @reset S.dat.n_pred = length(S.dat.pred_list);
    @reset S.dat.n_pred0 = length(S.dat.pred0_list);
    @reset S.dat.u0_dim = 1 + S.dat.n_pred0;
 


    return S

end






function build_inputs(S)

    println("")

    for fold in ["train", "test"]


        if fold == "train"

            sel = deepcopy(S.dat.sel_train);
            n_trials = deepcopy(S.dat.n_train);

        elseif fold == "test"

            sel = deepcopy(S.dat.sel_test);
            n_trials = deepcopy(S.dat.n_test);

        end



        # build within-trial design matrix        
        u_list = map(split_list, S.dat.pred_list);
        pred_cond = zeros(S.dat.n_pred, n_trials);

        center(A,d=1) = A .- mean(A,dims=d)

        for pp in eachindex(u_list)

            z_cond = ones(sum(sel));
            for uu in eachindex(u_list[pp])
                if u_list[pp][uu] == "switchSel"
                    z_cond .*= vec(S.dat.trial["switch"][sel].==1);
                elseif u_list[pp][uu] == "repeatSel"
                    z_cond .*= vec(S.dat.trial["switch"][sel].==0);
                else
                    z_cond .*= zsel(vec(S.dat.trial[u_list[pp][uu]]), sel);
                end
            end

            pred_cond[pp,:] .= z_cond;

        end
      

        # build basis set
        pred_misc = [];
        println("basis: $(S.dat.basis_name)")
        


        if S.dat.basis_name == "bspline"


            # set up basis
            if S.dat.n_splines > 0

                @reset S.dat.n_bases = S.dat.n_splines;
                basis = averagebasis(4, LinRange(1, S.dat.n_times, S.dat.n_bases));
                pred_basis = ["spline" for _ in 1:S.dat.n_bases];

            elseif S.dat.spline_gap > 0

                @reset S.dat.n_bases = round(S.dat.n_times/S.dat.spline_gap);
                basis = averagebasis(4, LinRange(1, S.dat.n_times, S.dat.n_bases));
                pred_basis = ["spline" for _ in 1:S.dat.n_bases];

            else
                error("n_splines or spline_gap must be greater than 0")
            end
            @reset S.dat.u_dim = S.dat.n_misc + S.dat.n_bases + (S.dat.n_bases*S.dat.n_pred);

            # construct basis
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials); 

            normalize_col(A,d=1) = A ./ (sqrt.(sum(abs2,A,dims=d)))
            for tt in axes(u,2)
                bs = bsplines(basis, tt);
                u[collect(axes(bs,1)),tt,:] .= S.dat.norm_basis ? normalize_col(collect(bs)) : collect(bs);
            end

            if S.dat.norm_basis
                println("normalized basis")
            else
                println("unnormalized basis")
            end

            println("n bases: $(S.dat.n_bases), breakpoints: $(round.(breakpoints(basis),sigdigits=4))")


        elseif S.dat.basis_name == "bins"

            ts = S.dat.ts;
            first_t = ts[findfirst(S.dat.epoch .== 2)];
            last_t =  ts[findlast(S.dat.epoch .== 3)];

            nbins = ceil(Int64, (last_t-first_t)/S.dat.bin_width)
            @reset S.dat.n_bases = nbins;
            @reset S.dat.bin_skip = round(Int, nbins/2);

            bin_edges = collect(first_t:S.dat.bin_width:(first_t + S.dat.bin_width*nbins));
            bin_edges[end] = last_t;
            println("nbins: $(nbins), bin edges: $(bin_edges)")
            @assert mod(nbins/2,1) == 0 "nbins must be even"

            pred_basis = ["bin$(ii)" for ii in 1:S.dat.n_bases];

            @reset S.dat.u_dim = S.dat.n_misc + S.dat.n_bases + (S.dat.n_bases*S.dat.n_pred);
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials); 

            # make bins
            for bb in 1:(nbins)
                u[bb, findall(bin_edges[bb] .<= ts .< bin_edges[bb+1]), :] .= 1.0;
            end

        elseif S.dat.basis_name == "cue"

            @reset S.dat.n_bases = 1;
            @reset S.dat.u_dim = S.dat.n_misc + S.dat.n_bases + (S.dat.n_bases*S.dat.n_pred);
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials); 

            pred_basis = ["cue"];
            u[1,:,:] .= (S.dat.epoch .== 2);


        elseif S.dat.basis_name == "cue_evoke"

            @reset S.dat.n_misc = 2;
            pred_misc = ["cueOn", "cueOff"]

            @reset S.dat.n_bases = 1;
            pred_basis = ["cue"];

            @reset S.dat.u_dim = S.dat.n_misc + S.dat.n_bases + (S.dat.n_bases*S.dat.n_pred);
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials); 


            # misc (transients)
            cue_on_idx = findfirst(S.dat.epoch .== 2);
            cue_on = cue_on_idx:findlast(S.dat.ts .<= (S.dat.ts[cue_on_idx] + .050));
            u[1,cue_on,:] .= 1.0;
            u[1,:,:] .-= mean(u[1,:,:], dims=1);

            cue_off_idx = findfirst(S.dat.epoch .== 3);
            cue_off = cue_off_idx:findlast(S.dat.ts .<= (S.dat.ts[cue_off_idx] + .050));
            u[2,cue_off,:] .= 1.0;
            u[2,:,:] .-= mean(u[2,:,:], dims=1);

            # bases 
            u[3,:,:] .= (S.dat.epoch .== 2);


        elseif S.dat.basis_name == "cue_CV"

            @reset S.dat.n_bases = 2;
            @reset S.dat.u_dim = (S.dat.n_bases*S.dat.n_pred)+S.dat.n_bases;
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials); 

            blocks = S.dat.trial["block"][sel,1];
            unique_blocks = unique(blocks);

            pred_basis = ["blk1", "blk2"];
            u[1,:,in(unique_blocks[1:2:end]).(blocks)] .= (S.dat.epoch .== 2);
            u[2,:,in(unique_blocks[2:2:end]).(blocks)] .= (S.dat.epoch .== 2);



        elseif S.dat.basis_name == "cue_linear"

            @reset S.dat.n_bases = 2;
            @reset S.dat.u_dim = (S.dat.n_bases*S.dat.n_pred)+S.dat.n_bases;
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials);
        
            pred_basis = ["cue1", "cueLin"];

            # intercept
            u[1,:,:] .= (S.dat.epoch .== 2);

            # linear
            cue_lin = convert(Vector{Float64}, S.dat.epoch .== 2);
            cue_lin[cue_lin.==true] = LinRange(-1,1,sum(S.dat.epoch .== 2));
            u[2,:,:] .= deepcopy(cue_lin);




        elseif S.dat.basis_name == "cue_ISI"

            @reset S.dat.n_bases = 2;
            @reset S.dat.u_dim = (S.dat.n_bases*S.dat.n_pred)+S.dat.n_bases;
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials);

            pred_basis = ["cue","ISI"];

            u[1,:,:] .= S.dat.epoch .== 2;
            u[2,:,:] .= S.dat.epoch .== 3;


        elseif S.dat.basis_name == "cue_ISI_trial"

            @reset S.dat.n_bases = 3;
            @reset S.dat.u_dim = (S.dat.n_bases*S.dat.n_pred)+S.dat.n_bases;
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials);

            pred_basis = ["cue","ISI","trial"];

            u[1,:,:] .= S.dat.epoch .== 2;
            u[2,:,:] .= S.dat.epoch .== 3;
            u[3,:,:] .= S.dat.epoch .== 4;

        elseif S.dat.basis_name == "cueISI_linear"

            @reset S.dat.n_bases = 4;
            @reset S.dat.u_dim = (S.dat.n_bases*S.dat.n_pred)+S.dat.n_bases;
            u = zeros(S.dat.u_dim, S.dat.n_times, n_trials);
        
            pred_basis = ["cue", "cueLin", "ISI", "ISILin"];

            # cue intercept
            u[1,:,:] .= (S.dat.epoch .== 2);

            # cue linear
            cue_lin = convert(Vector{Float64}, S.dat.epoch .== 2);
            cue_lin[cue_lin.==true] = LinRange(-1,1,sum(S.dat.epoch .== 2));
            u[2,:,:] .= deepcopy(cue_lin);


            # ISI intercept
            u[3,:,:] .= (S.dat.epoch .== 3);

            # cue linear
            isi_lin = convert(Vector{Float64}, S.dat.epoch .== 3);
            isi_lin[isi_lin.==true] = LinRange(-1,1,sum(S.dat.epoch .== 3));
            u[4,:,:] .= deepcopy(isi_lin);



        else
            error("basis_name not recognized")
        end


        
        # convolve predictors with basis set
        for bb = 1:S.dat.n_bases
            for uu in axes(pred_cond,1)
                u[(S.dat.n_misc + S.dat.n_bases) + ((uu-1)*S.dat.n_bases)+bb,:,:] .= u[S.dat.n_misc+bb,:,:] .* pred_cond[uu,:]';
            end
        end

        
        # check collinearity
        ul = deepcopy(reshape(u, S.dat.u_dim, S.dat.n_times*n_trials)');
        f=svd(ul);



        # build initial state list
        u0_list = map(split_list, S.dat.pred0_list);
        u0 = zeros(S.dat.u0_dim, n_trials);
        u0[1,:] .= 1.0; # constant term

        for pp in eachindex(u0_list)

            z_cond = ones(sum(sel));
            for uu in eachindex(u0_list[pp])
                z_cond .*= zsel(vec(S.dat.trial[u0_list[pp][uu]]), sel);
            end

            u0[pp+1,:] .= z_cond;

        end






        if fold == "train"

            @reset S.dat.u_train = u;
            @reset S.dat.n_train = n_trials;
            @reset S.dat.pred_collin_train = f.S ./ f.S[end];
            @reset S.dat.pred_name = [pred_misc; pred_basis; vec(repeat(S.dat.pred_list, inner=(S.dat.n_bases,1)))];
       
            @reset S.dat.u0_train = u0;
            @reset S.dat.pred0_name = ["bias"; S.dat.pred0_list];

            @reset S.dat.u_train_cor = cor(ul);


            println("========== train fold inputs ==========")


        elseif fold == "test"

            @reset S.dat.u_test = u;
            @reset S.dat.n_test = n_trials;
            @reset S.dat.pred_collin_test = f.S ./ f.S[end];

            @reset S.dat.u0_test = u0;

            println("========== test fold fold inputs ==========")

        end


    
        println("predictor mean quartiles: $(round(median(mean(ul, dims=1)), sigdigits=4)) +/- $(round(iqr(mean(ul, dims=1))/2, sigdigits=4))")
        println("predictor var quartiles: $(round(median(var(ul, dims=1)), sigdigits=4)) +/- $(round(iqr(var(ul, dims=1))/2, sigdigits=4))");
        println("collinearity metric (best=1, threshold=30): $(round(f.S[1]/f.S[end], sigdigits=4))"); 
        println("========================================\n")

    end 



    return S

end





function whiten_y(S)
# orthogonalize data


    if S.prm.remove_ERP
        
        println("removing ERP")

        ERP = mean(S.dat.y_train_orig, dims=(1,3));
        S.dat.y_train_orig .-= ERP;
        S.dat.y_test_orig .-= ERP;

    end

    yl = reshape(S.dat.y_train_orig, S.dat.n_chans, S.dat.n_times*S.dat.n_train);

    if  uppercase(S.prm.y_transform) == "PCA"

        # fit PCA
        pca = fit(PCA, yl, pratio=S.prm.PCA_ratio, maxoutdim=S.prm.PCA_maxdim); # ==== do PCA

        # enforce sign convention (largest component is positive)
        W = pca.proj;
        for cc in axes(W,2)
            W[:,cc] .*= sign(W[findmax(abs.(W[:,cc]))[2],cc]);
        end

        # save
        @reset S.dat.W = W;
        @reset S.dat.mu = pca.mean;
        @reset S.dat.pca_R2 = pca.tprinvar/pca.tvar;
        @reset S.dat.y_dim = size(S.dat.W, 2);


        println("\n========== PCA with dim=$(S.dat.y_dim) ==========")

        if S.prm.y_ICA
            
            println("fitting ICA to y")

             # fit ICA
             ica = fit(ICA, demix(S,yl), S.dat.y_dim;  maxiter=1000000, do_whiten=false, mean=0); # ==== do ICA
             @reset S.dat.W = pca.proj*ica.W;

        end

        println("variance included: $(S.dat.pca_R2)");
        println("eigenvalues: $(round.(principalvars(pca)'))");

    elseif  uppercase(S.prm.y_transform) == "WHITEN"

        pca = fit(Whitening, yl, regcoef=1e-6); # ==== do whitening

        @reset S.dat.W = pca.W;
        @reset S.dat.mu = pca.mean;
        @reset S.dat.pca_R2 = 1.0;
        @reset S.dat.y_dim = deepcopy(S.dat.n_chans);

        println("\n========== Whitening with dim=$(S.dat.y_dim) ==========")


    elseif  uppercase(S.prm.y_transform) == "CCA"

        cca = fit(CCA, yl[:,1:end-1], yl[:,2:end]); # ==== do CCA
        @reset S.dat.y_dim = findfirst(x -> x < S.prm.PCA_ratio, cca.corrs)-1;
        @reset S.dat.W = cca.xproj[:,1:S.dat.y_dim];
        @reset S.dat.mu = cca.xmean;
        @reset S.dat.pca_R2 = S.prm.PCA_ratio;
      

        println("\n========== Whitening with dim=$(S.dat.y_dim) ==========")

    elseif  uppercase(S.prm.y_transform) == "NONE"

        @reset S.dat.W = Matrix(1.0I(S.dat.n_chans));
        @reset S.dat.mu = vec(mean(yl, dims=2));
        @reset S.dat.pca_R2 = 1.0;
        @reset S.dat.y_dim = deepcopy(S.dat.n_chans);

        # get possible C matrix
        pca = fit(PCA, yl, pratio=1.0, maxoutdim=S.dat.x_dim);
        @reset S.dat.pca_C = pca.proj;

        println("\n========== No Transform with dim=$(S.dat.y_dim) ==========")


    else
        error("invalid y_transform")
    end
    println("========================================\n")




    # whiten train ==================================
    y_train = zeros(S.dat.y_dim, S.dat.n_times, S.dat.n_train);
    for tt in axes(y_train,3)
        y_train[:,:,tt] = NeSS.demix(S, S.dat.y_train_orig[:,:,tt]);
    end

    @reset S.dat.y_train = y_train;

    # whiten test ==================================
    y_test = zeros(S.dat.y_dim, S.dat.n_times, S.dat.n_test);
    for tt in axes(y_test,3)
        y_test[:,:,tt] = NeSS.demix(S, S.dat.y_test_orig[:,:,tt]);
    end

    @reset S.dat.y_test = y_test;


    return S

end




function convert_bal(S)



    sys = ss(S.mdl.A, S.mdl.B, S.mdl.C, 0, S.dat.dt);
    _,_,T = balreal(sys);


    Ab = T*S.mdl.A/T;
    Bb = T*S.mdl.B;
    Qb = format_noise(X_A_Xt(S.mdl.Q, T), S.prm.Q_type);

    Cb = S.mdl.C/T;

    B0b = T*S.mdl.B0;
    P0b = format_noise(X_A_Xt(S.mdl.P0,T), S.prm.P0_type);


    # reconstruct model
    mdl = set_model(
        A = Ab,
        B = Bb,
        Q = Qb,
        C = Cb,
        R = S.mdl.R,
        B0 = B0b,
        P0 = P0b,
        );


    return mdl



end





















function test_rep_ESTEP(S)
#  run ESTEP twice and compare results (should be zero)
   

    NeSS.task_ESTEP!(S);
    M1 = deepcopy(S.est);

    NeSS.task_ESTEP!(S);
    M2 = deepcopy(S.est);


    rep_norm = zeros(0);

    push!(rep_norm, norm(M1.xx_init .- M2.xx_init))
    push!(rep_norm, norm(M1.xy_init .- M2.xy_init))
    push!(rep_norm, norm(M1.yy_init .- M2.yy_init));

    push!(rep_norm, norm(M1.xx_dyn .- M2.xx_dyn));
    push!(rep_norm, norm(M1.xy_dyn .- M2.xy_dyn));
    push!(rep_norm, norm(M1.yy_dyn .- M2.yy_dyn));

    push!(rep_norm, norm(M1.xx_obs .- M2.xx_obs))
    push!(rep_norm, norm(M1.xy_obs .- M2.xy_obs))
    push!(rep_norm, norm(M1.yy_obs .- M2.yy_obs));


    return rep_norm

end



function generate_lds_parameters(S, Q_noise, R_noise, P0_noise)::NamedTuple

    A = randn(S.dat.x_dim, S.dat.x_dim) + 5I;
    s,u = eigen(A); # get eigenvectors and eigenvals
    s = s/maximum(abs.(s))*.95; # set largest eigenvalue to lie inside unit circle (enforcing stability)
    s[real.(s) .< 0] = -s[real.(s) .< 0]; #set real parts to be positive (encouraging smoothness)
    A_sim = real(u*(Diagonal(s)/u));  # reconstruct A from eigs and eigenvectors

    # diagonal Q
    # Q =  Matrix(sqrt(Q_noise) * I(S.dat.x_dim)); 
    Q = sqrt(Q_noise) .* randn(S.dat.x_dim, S.dat.x_dim); 
    Q_sim = tol_PD(Q'*Q; tol=.1);

    B_sim = 0.5*randn(S.dat.x_dim, S.dat.u_dim);
    C_sim = 0.5*randn(S.dat.n_chans, S.dat.x_dim);

    R =  sqrt(R_noise) .* randn(S.dat.n_chans, S.dat.n_chans); 
    R_sim = tol_PD(R'*R; tol=.1);

    B0_sim = randn(S.dat.x_dim, S.dat.u0_dim);

    P = sqrt(P0_noise) .* randn(S.dat.x_dim, S.dat.x_dim);
    P0_sim = tol_PD(P'*P; tol=.1);
    

    sim = (A = A_sim, B = B_sim, C = C_sim, Q = Q_sim, R = R_sim, B0 = B0_sim, P0 = P0_sim);
    return sim

end