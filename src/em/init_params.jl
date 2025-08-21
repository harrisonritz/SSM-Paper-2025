

 


function init_ssid(S)

    trial_sel = 1:S.dat.n_train;

    # format data ==================================================
    y = deepcopy(S.dat.y_train[:,:,trial_sel]);
    yl = reshape(y, size(y,1), size(y,2)*size(y,3));


    # use subset of predictors to keep SSID well-posed + reduce computation
    @assert (S.dat.basis_name != "") "basis_name must be set"
    if (S.dat.basis_name == "cueISI") 

        println("init for cueISI")
        u = deepcopy(S.dat.u_train[1:2:end,:,trial_sel]);

    elseif (S.dat.basis_name == "bspline")

        println("init for bspline")


        if S.prm.ssid_new_init
            println("new init -- HALF INPUT")
            u = deepcopy(S.dat.u_train[1:1:end,:,trial_sel]);
        else
            u = deepcopy(S.dat.u_train[2:S.dat.n_bases:end,:,trial_sel]);
            # u = sum([S.dat.u_train[ii:S.dat.n_bases:end,:,trial_sel] for ii in 1:5]);
            # u = sum([S.dat.u_train[ii:S.dat.n_bases:end,:,trial_sel] for ii in 1:5]);

        end
        u_orig = deepcopy(u); #reshape(deepcopy(S.dat.u_train[:,:,trial_sel]), size(S.dat.u_train,1), size(S.dat.u_train,2)*size(S.dat.u_train,3));


        println("u size: ", size(u))
        println("u_orig size: ", size(u_orig))

    else

        u = deepcopy(S.dat.u_train[:,:,trial_sel]);

    end
    ul = reshape(u, size(u,1), size(u,2)*size(u,3));
    u_origl = reshape(u_orig, size(u_orig,1), size(u_orig,2)*size(u_orig,3));

    # make sys
    # id = prefilter(iddata(yl, ul, S.dat.dt, .01, 40);
    id = iddata(yl, ul, S.dat.dt);
    println(id);


    # run subspace id ==================================================
    @views @inbounds @inline sys = subspaceid_orig(id, S.dat.x_dim; 
                                            r=S.prm.ssid_lag,
                                            u_orig=u_origl,
                                            stable=true,
                                            verbose=true, 
                                            scaleU=false, 
                                            zeroD=true,
                                            Aestimator = ridge_estimator,
                                            Bestimator = ridge_estimator,
                                            W=S.prm.ssid_type,
                                            new_init=S.prm.ssid_new_init);

    # sys = subspaceid_orig(id, nx; r=k, verbose=true, scaleU=false, zeroD=true, W=:IVM)
    # sys = subspaceid_trial(y,u, nx; r=k, verbose=true, scaleU=false, zeroD=true, W=:IVM)

    # balanced realization
    # sysr, G, T  = balreal(sys);


    # format system ==================================================
    
    # dynamics terms
    Ad = deepcopy(sys.A);

    # re-add ISI predictors
    if (S.dat.basis_name == "cueISI") 

        sysB = deepcopy(sys.B);

        Bd = zeros(S.dat.x_dim, S.dat.u_dim);
        Bd[:,1:2:end] = sysB;
        Bd[:,2:2:end] = sysB;


    elseif (S.dat.basis_name == "cueISI_linear") 

        sysB = deepcopy(sys.B);

        Bd = zeros(S.dat.x_dim, S.dat.u_dim);
        Bd[:,1:4:end] = sysB;
        Bd[:,3:4:end] = sysB;


    elseif (S.dat.basis_name == "bspline") 

        sysB = deepcopy(sys.B);
     

        if S.prm.ssid_new_init
            println("new init -- HALF INPUT")
            Bd = zeros(S.dat.x_dim, S.dat.u_dim);
            Bd[:,1:1:end] .= sysB;
            # Bd[:,2:2:end] = sysB;
        else
            Bd = zeros(S.dat.x_dim, S.dat.u_dim);
            for ii in 1:S.dat.n_bases
                Bd[:,ii:S.dat.n_bases:end] = sysB/S.dat.n_bases;
            end
        end

       
    elseif  (S.dat.basis_name == "bins")


        sysB = deepcopy(sys.B);
        n_bases = deepcopy(S.dat.n_bases);
        bin_skip = deepcopy(S.dat.bin_skip);

        Bd = zeros(S.dat.x_dim, S.dat.u_dim);
        Bd[:,1:bin_skip:end] = sysB;

        bin_mix = collect((1:(bin_skip-1))./bin_skip)

        for ii in 2:bin_skip
            Bd[:,ii:n_bases:end] = (1-bin_mix[ii-1]).*Bd[:,1:n_bases:end] .+ bin_mix[ii-1].*Bd[:,(bin_skip+1):n_bases:end];
            Bd[:,(bin_skip+ii):n_bases:end] = Bd[:,(bin_skip+1):n_bases:end];
        end

    else

        Bd = deepcopy(sys.B);

    end

    Cd = deepcopy(sys.C);

    x0 = Cd\y[:,1,:];
    B0d = x0/S.dat.u0_train[:,trial_sel];

    # noise terms
    Qd = format_noise(sys.Q, S.prm.Q_refine_type);
    Rd = format_noise(sys.R, S.prm.R_refine_type);
    P0d = format_noise(sys.P, S.prm.P0_refine_type);
      

    # save model ==================================================
    @reset S.mdl = set_model(
                            A = Ad,
                            B = Bd,
                            Q = Qd,
                            C = Cd,
                            R = Rd,
                            B0 = B0d,
                            P0 = P0d,
                            );

    @reset S.res.mdl_ssid = deepcopy(S.mdl)
    @reset S.res.ssid_sv = sys.s.S;


    return S
  
end



# use ridge regression for SSID
function ridge_estimator(x::AbstractArray{Float64},y::AbstractArray{Float64})
    
    p = size(x,2);

    xr = [x; sqrt(1e-3)I(p)];
    yr = [y; zeros(p, size(y,2))];
    br = xr\yr;

    return br;
end


function ridge_estimator(x::QRPivoted,y::AbstractArray{Float64})

    diagR = Diagonal(x.R);
    x.R .+= sqrt.(diagR.^2 + 1e-3I(size(x.R,2))) - diagR;
    br = x \ y;
    
    return br;

end




function subspaceid_orig(
    data::InputOutputData,
    nx = :auto;
    u_orig=Matrix{Float64}(undef, 0, 0),
    verbose = false,
    r = nx === :auto ? min(length(data) ÷ 20, 50) : 2nx + 10, # the maximal prediction horizon used
    s1 = r, # number of past outputs
    s2 = r, # number of past inputs
    γ = nothing, # discarded, aids switching from n4sid
    W = :CVA,
    zeroD = false,
    stable = true, 
    focus = :prediction,
    svd::F1 = svd!,
    scaleU = true,
    Aestimator::F2 = \,
    Bestimator::F3 = \,
    weights = nothing,
    new_init = true,
) where {F1,F2,F3}

    # nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
    y, u = transpose(copy(output(data))), transpose(copy(input(data)))
    if isempty(u_orig)
        println("u_orig is empty")
        u_orig = deepcopy(u)
    end

    # println("size y = $(size(y)), size u = $(size(u))")
    if scaleU
        CU = std(u, dims=1)
        u ./= CU
    end
    t, p = size(y, 1), size(y, 2)
    m = size(u, 2)
    t0 = max(s1,s2)+1
    s = s1*p + s2*m
    N = t - r + 1 - t0

    @views @inbounds function hankel(u::AbstractArray, t0, r)
        d = size(u, 2)
        H = zeros(eltype(u), r * d, N)
        for ri = 1:r, Ni = 1:N
            H[(ri-1)*d+1:ri*d, Ni] = u[t0+ri+Ni-2, :] # TODO: should start at t0
        end
        H
    end

    # 1. Form G  (10.103). (10.100). (10.106). (10.114). and (10.108).

    Y = hankel(y, t0, r) # these go forward in time
    U = hankel(u, t0, r) # these go forward in time
    # @assert all(!iszero, Y) # to be turned off later
    # @assert all(!iszero, U) # to be turned off later
    @assert size(Y) == (r*p, N)
    @assert size(U) == (r*m, N)
    φs(t) = [ # 10.114
        y[t-1:-1:t-s1, :] |> vec # QUESTION: not clear if vec here or not, Φ should become s × N, should maybe be transpose before vec, but does not appear to matter
        u[t-1:-1:t-s2, :] |> vec
    ]

    Φ = reduce(hcat, [φs(t) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value
    @assert size(Φ) == (s, N)

    println("ssid lq...")
    UΦY = [U; Φ; Y]
    l = lq!(UΦY)
    L = l.L
    if W ∈ (:MOESP, :N4SID)
        Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    end
    l = nothing # free memory

    # @assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q)), if this fails, you may need to lower the prediction horizon r which is currently set to $r"
    Uinds = 1:size(U,1)
    Φinds = (1:size(Φ,1)) .+ Uinds[end]
    Yinds = (1:size(Y,1)) .+ (Uinds[end]+s)
    @assert Yinds[end] == p*r+m*r+s

    L1 = L[Uinds, Uinds]
    L2 = L[s1*p+(r+s2)*m+1:end, 1:s1*p+(r+s2)*m+p]
   

    # 2. Select weighting matrices W1 (rp × rp)
    # and W2 (p*s1 + m*s2 × α) = (s × α)
    # @assert size(Ĝ, 1) == r*p
    if W ∈ (:MOESP, :N4SID)

        L21 = L[Φinds, Uinds]
        L22 = L[Φinds, Φinds]
        L32 = L[Yinds, Φinds]
       
        Q1 = Q[Uinds, :]
        Q2 = Q[Φinds, :]
        Ĝ = L32*(L22\[L21 L22])*[Q1; Q2] # this G is used for N4SID weight, but also to form Yh for all methods

        if W === :MOESP
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ*ΠUt
            G = L32*Q2 #* 1/N# QUESTION: N does not appear to matter here
        elseif W === :N4SID
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ
            G = deepcopy(Ĝ) #* 1/N
        end

    elseif W ∈ (:IVM, :CVA)

        if W === :IVM

            UY = [U; Y]
            Yinds = (1:size(Y,1)) .+ size(U,1)
            YΠUt = proj_hr(UY, Yinds)
            
            G = YΠUt*Φ' #* 1/N # 10.109, pr×s # N does not matter here
            @assert size(G) == (p*r, s)
            W1 = sqrt(Symmetric(pinv(inv(N) * (YΠUt*Y')))) |> real
            W2 = sqrt(Symmetric(pinv(inv(N) * Φ*Φ'))) |> real
            G = W1*G*W2
            @assert size(G, 1) == r*p

        elseif W === :CVA        

            L32 = L[Yinds, Φinds]
            W1 = L[Yinds,[Φinds; Yinds]]

            println("ssid svd...")
            ull1,sll1 = svd!(W1)
            sll1 = Diagonal(sll1[1:r*p])
            # Or,Sn = svd(pinv(sll1)*ull1'*L32)
            cva_svd = svd!(sll1\(ull1'*L32))
            Or = ull1*sll1*cva_svd.U
            # ΦΠUt = proj(Φ, U)
            # W1 = pinv(sqrt(1/N * (YΠUt*Y'))) |> real
            # W2 = pinv(sqrt(1/N * ΦΠUt*Φ')) |> real
            # G = W1*G*W2

        end

        # @assert size(W1) == (r*p, r*p)
        # @assert size(W2, 1) == p*s1 + m*s2

    else
        throw(ArgumentError("Unknown choice of W"))
    end

    # 3. Select R and define Or = W1\U1*R
    sv = W === :CVA ? svd!(L32) : svd!(G)
    if nx === :auto
        nx = sum(sv.S .> sqrt(sv.S[1] * sv.S[end]))
        verbose && @info "Choosing order $nx"
    end
    n = nx
    S1 = sv.S[1:n]
    R = Diagonal(sqrt.(S1))
    if W !== :CVA
        U1 = sv.U[:, 1:n]
        V1 = sv.V[:, 1:n]
        Or = W1\(U1*R)
    end
    
    fve = sum(S1) / sum(sv.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    C = Or[1:p, 1:n]
    A = Aestimator(Or[1:p*(r-1), 1:n] , Or[p+1:p*r, 1:n])
    if !all(e->abs(e)<=1, eigvals(A))
        if stable
            verbose && @info "A matrix unstable -- stabilizing by reflection"
            A = reflectd(A)
        else
            verbose && @info "A matrix unstable -- NOT stabilizing"
        end
    end





    if new_init


        println("NEW INIT ...")
        println("estimating Q&R ...")

        @views @inbounds @inline Qc, Rc, Pc = find_PK_hr(L1,L2,Or,n,p,m,r,s1,s2,A,C)
       
        # set parameters
        P = Pc
        K = zeros(n, p)
        Sc = zeros(p, m)
                
        D=0;

        B=ones(size(A,1),size(u_orig,1));
        B/=sum(B)

        x0=ones(size(A,1));
        x0/=sum(x0)


    else

        println("estimating noise & gain ...")
        @views @inbounds @inline P, K, Qc, Rc, Sc = find_PK(L1,L2,Or,n,p,m,r,s1,s2,A,C)
    

        println("estimating B & D ...")
        pred_K = (focus === :prediction)*K
        ut = u_orig
        ut = transpose(u)
        mt = size(u_orig, 1)
        println("size u_orig $(size(u_orig))")
        yt = transpose(y)

        @views @inbounds @inline B, D, x0 = find_BD_hr(A, pred_K, C, u_orig, yt, mt, zeroD, Bestimator, weights)
    
    end



    # TODO: iterate find C/D and find B/D a couple of times

    if scaleU
        B ./= CU
        D ./= CU
    end


    
    N4SIDStateSpace(ss(A,  B,  C,  D, data.Ts), Qc,Rc,Sc,K,P,x0,sv,fve)

end





function find_BD_hr_new(A,K,C,U,Y,m, zeroD=false, estimator=\, weights=nothing)
    T = eltype(A)
    nx = size(A, 1)
    p = size(C, 1)
    N = size(U, 2)
    A = A-K*C
    y_hat = lsim(ss(A,K,C,0,1), Y)[1] # innovation sequence
    φB = zeros(Float64, p, N, m*nx)
    @inbounds @views for (j,k) in Iterators.product(1:nx, 1:m)
        E = zeros(nx)
        E[j] = 1
        fsys = ss(A, E, C, 0, 1)
        u = U[k:k,:]
        uf = lsim(fsys, u)[1]
        r = (k-1)*nx+j
        @inbounds φB[:,:,r] .= uf 
    end
    φx0 = zeros(p, N, nx)
    x0u = zeros(1, N)
    @inbounds @views for (j,k) in Iterators.product(1:nx, 1:1)
        E = zeros(nx)
        x0 = zeros(nx); x0[j] = 1
        fsys = ss(A, E, C, 0, 1)
        uf = lsim(fsys, x0u; x0)[1]
        r = (k-1)*nx+j
        φx0[:,:,r] = uf 
    end
    if !zeroD
        φD = zeros(Float64, p, N, m*p)
        for (j,k) in Iterators.product(1:p, 1:m)
            E = zeros(p)
            E[j] = 1
            fsys = ss(E, 1)
            u = U[k:k,:]
            uf = lsim(fsys, u)[1]
            r = (k-1)*p+j
            φD[:,:,r] = uf 
        end
    end

    if zeroD

        φ3 = zeros(Float64, p, N, (m+1)*nx);
        @inbounds φ3[:,:,1:(m*nx)] .= φB;
        @inbounds φ3[:,:,((m*nx)+1):end] .= φx0;
        # φ3 = cat(φB, φx0, dims=Val(3));
        
        φ = reshape(φ3, (p*N, (m+1)*nx));
        φqr = qr!(φ, ColumnNorm());

        φ = nothing;
        φ3 = nothing;

    else
        φ3 = cat(φB, φx0, φD, dims=Val(3))
        φqr = reshape(φ3, p*N, :)
    end

    # @inbounds φ3 = zeroD ? cat(φB, φx0, dims=Val(3)) : cat(φB, φx0, φD, dims=Val(3))
    # φ4 = permutedims(φ3, (1,3,2))

    if weights === nothing
        BD = estimator(φqr, vec(Y .- y_hat))
    else
        BD = estimator(φqr, vec(Y .- y_hat), weights)
    end
    B = copy(reshape(BD[1:m*nx], nx, m))
    x0 = BD[m*nx .+ (1:nx)]
    if zeroD
        D = zeros(T, p, m)
    else
        D = copy(reshape(BD[end-p*m+1:end], p, m))
        B .+= K*D
    end
    B,D,x0
end








function find_BD_hr(A,K,C,U,Y,m, zeroD=false, estimator=\, weights=nothing)
    T = eltype(A)
    nx = size(A, 1)
    p = size(C, 1)
    N = size(U, 2)
    A = A-K*C
    y_hat = lsim(ss(A,K,C,0,1), Y)[1] # innovation sequence
    φB = zeros(Float64, p, N, m*nx)
    @inbounds @views for (j,k) in Iterators.product(1:nx, 1:m)
        E = zeros(nx)
        E[j] = 1
        fsys = ss(A, E, C, 0, 1)
        u = U[k:k,:]
        uf = lsim(fsys, u)[1]
        r = (k-1)*nx+j
        @inbounds φB[:,:,r] .= uf 
    end
    φx0 = zeros(p, N, nx)
    x0u = zeros(1, N)
    @inbounds @views for (j,k) in Iterators.product(1:nx, 1:1)
        E = zeros(nx)
        x0 = zeros(nx); x0[j] = 1
        fsys = ss(A, E, C, 0, 1)
        uf = lsim(fsys, x0u; x0)[1]
        r = (k-1)*nx+j
        φx0[:,:,r] = uf 
    end
    if !zeroD
        φD = zeros(Float64, p, N, m*p)
        for (j,k) in Iterators.product(1:p, 1:m)
            E = zeros(p)
            E[j] = 1
            fsys = ss(E, 1)
            u = U[k:k,:]
            uf = lsim(fsys, u)[1]
            r = (k-1)*p+j
            φD[:,:,r] = uf 
        end
    end

    if zeroD

        φ3 = zeros(Float64, p, N, (m+1)*nx);
        @inbounds φ3[:,:,1:(m*nx)] .= φB;
        @inbounds φ3[:,:,((m*nx)+1):end] .= φx0;
        # φ3 = cat(φB, φx0, dims=Val(3));
        
        φ = reshape(φ3, (p*N, (m+1)*nx));
        φqr = qr!(φ, ColumnNorm());

        φ = nothing;
        φ3 = nothing;

    else
        φ3 = cat(φB, φx0, φD, dims=Val(3))
        φqr = reshape(φ3, p*N, :)
    end

    # @inbounds φ3 = zeroD ? cat(φB, φx0, dims=Val(3)) : cat(φB, φx0, φD, dims=Val(3))
    # φ4 = permutedims(φ3, (1,3,2))

    if weights === nothing
        BD = estimator(φqr, vec(Y .- y_hat))
    else
        BD = estimator(φqr, vec(Y .- y_hat), weights)
    end
    B = copy(reshape(BD[1:m*nx], nx, m))
    x0 = BD[m*nx .+ (1:nx)]
    if zeroD
        D = zeros(T, p, m)
    else
        D = copy(reshape(BD[end-p*m+1:end], p, m))
        B .+= K*D
    end
    B,D,x0
end

function find_BDf(A, C, U, Y, λ, zeroD, Bestimator, estimate_x0)
    nx = size(A,1)
    ny, nw = size(Y)
    nu = size(U, 1)
    if estimate_x0
        ue = [U; transpose(λ)] # Form "extended input"
        nup1 = nu + 1
    else
        ue = U
        nup1 = nu
    end

    sys0 = ss(A,I(nx),C,0) 
    F = evalfr2(sys0, λ)
    # Form kron matrices
    if zeroD      
        AA = similar(U, nw*ny, nup1*nx)
        for i in 1:nw
            r = ny*(i-1) + 1:ny*i
            for j in 1:nup1
                @views AA[r, ((j-1)nx) + 1:j*nx] .= ue[j, i] .* (F[:, :, i])
            end
        end
    else
        AA = similar(U, nw*ny, nup1*nx+nu*ny) 
        for i in 1:nw
            r = (ny*(i-1) + 1) : ny*i
            for j in 1:nup1
                @views AA[r, (j-1)nx + 1:j*nx] .= ue[j, i] .* (F[:, :, i])
            end
            for j in 1:nu
                @views AA[r, nup1*nx + (j-1)ny + 1:nup1*nx+ny*j] = ue[j, i] * I(ny)
            end
        end
    end
    vy = vec(Y)
    YY = [real(vy); imag(vy)]
    AAAA = [real(AA); imag(AA)]
    BD = Bestimator(AAAA, YY)
    e = YY - AAAA*BD
    B = reshape(BD[1:nx*nup1], nx, :)
    D = zeroD ? zeros(eltype(B), ny, nu) : reshape(BD[nx*nup1+1:end], ny, nu)
    if estimate_x0
        x0 = B[:, end]
        B = B[:, 1:end-1]
    else
        x0 = zeros(eltype(B), nx)
    end
    return B, D, x0, e
end

function find_CDf(A, B, U, Y, λ, x0, zeroD, Bestimator, estimate_x0)
    nx = size(A,1)
    ny, nw = size(Y)
    nu = size(U, 1)
    if estimate_x0
        Ue = [U; transpose(λ)] # Form "extended input"
        Bx0 = [B x0]
    else
        Ue = U
        Bx0 = B
    end

    sys0 = ss(A,Bx0,I(nx),0)
    F = evalfr2(sys0, λ, Ue)
    # Form kron matrices
    if zeroD      
        AA = F
    else
        AA = [F; U]
    end


    YY = [real(transpose(Y)); imag(transpose(Y))]
    AAAA = [real(AA) imag(AA)]
    CD = Bestimator(transpose(AAAA), YY) |> transpose
    e = YY - transpose(AAAA)*transpose(CD)
    C = CD[:, 1:nx]
    D = zeroD ? zeros(eltype(C), ny, nu) : CD[:, nx+1:end]
    return C, D, e
end

function proj_hr(UY, Yinds)
    # UY = [U; Yi]
    l = lq!(UY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    # Uinds = 1:size(U,1)
    # Yinds = (1:size(Yi,1)) .+ Uinds[end]
    # if Yi === Y
        # @assert size(Q) == (p*r+m*r, N) "size(Q) == $(size(Q))"
        # @assert Yinds[end] == p*r+m*r
    # end
    L22 = L[Yinds, Yinds]
    Q2 = Q[Yinds, :]
    L22*Q2
end


function find_PK_hr(L1,L2,Or,n,p,m,r,s1,s2,A,C)
    
    println("pk setup...")
    X1 = L2[p+1:r*p, 1:m*(s2+r)+p*s1+p]
    X2 = [L2[1:r*p,1:m*(s2+r)+p*s1] zeros(r*p,p)]
    vl = [Or[1:(r-1)*p, 1:n]\X1; L2[1:p, 1:m*(s2+r)+p*s1+p]]
    hl = [Or[:,1:n]\X2 ; [L1 zeros(m*r,(m*s2+p*s1)+p)]]
    
    println("pk regression...")
    vl_hat = (vl/hl)*hl

    # W = (vl - K0*hl)*(vl-K0*hl)'
    println("pk cov...")
    # @views Q = (((vl[1:n,:] - vl_hat[1:n,:])*(vl[1:n,:] - vl_hat[1:n,:])')) / (size(vl,2)-n) |> Hermitian
    # @views R = ((vl[n+1:n+p,:] - vl_hat[n+1:n+p,:])*(vl[n+1:n+p,:] - vl_hat[n+1:n+p,:])') / (size(vl,2)-p) |> Hermitian
    @views Q = (((vl[1:n,:] - vl_hat[1:n,:])*(vl[1:n,:] - vl_hat[1:n,:])')) |> Hermitian
    @views R = ((vl[n+1:n+p,:] - vl_hat[n+1:n+p,:])*(vl[n+1:n+p,:] - vl_hat[n+1:n+p,:])') |> Hermitian

    # get P
    println("pk p...")
    a = 1/sqrt(mean(abs, Q)*mean(abs, R)) # scaling for better numerics in ared
    P, _, _, _ = ControlSystemIdentification.MatrixEquations.ared(copy(A'), copy(C'), a*R, a*Q)

    return Q, R, P
end



function find_PK(L1,L2,Or,n,p,m,r,s1,s2,A,C)
    X1 = L2[p+1:r*p, 1:m*(s2+r)+p*s1+p]
    X2 = [L2[1:r*p,1:m*(s2+r)+p*s1] zeros(r*p,p)]
    vl = [Or[1:(r-1)*p, 1:n]\X1; L2[1:p, 1:m*(s2+r)+p*s1+p]]
    hl = [Or[:,1:n]\X2 ; [L1 zeros(m*r,(m*s2+p*s1)+p)]]
    
    K0 = vl*pinv(hl)
    W = (vl - K0*hl)*(vl-K0*hl)'
    
    Q = W[1:n,1:n] |> Hermitian
    S = W[1:n,n+1:n+p]
    R = W[n+1:n+p,n+1:n+p] |> Hermitian
    
    local P, K
    try
        a = 1/sqrt(mean(abs, Q)*mean(abs, R)) # scaling for better numerics in ared
        P, _, Kt, _ = ControlSystemIdentification.MatrixEquations.ared(copy(A'), copy(C'), a*R, a*Q, a*S)
        K = Kt' |> copy
    catch e
        @error "Failed to estimate kalman gain, got error" 
        P = I(n)
        K = zeros(n, p)
    end
 
    P, K, Q, R, S
end

function reflectd(x)
    a = abs(x)
    a < .9999 && return oftype(cis(angle(x)),x)
    (.9999)/a * cis(angle(x))
end

function reflectd(A::AbstractMatrix)
    D,V = eigen(A)
    D = reflectd.(D)
    A2 = V*Diagonal(D)/V
    if eltype(A) <: Real
        return real(A2)
    end
    A2
end










function subspaceid_trial(
    yo, uo,
    nx = :auto;
    verbose = false,
    r = nx === :auto ? min(size(y,3) ÷ 20, 50) : 2nx + 10, # the maximal prediction horizon used
    s1 = r, # number of past outputs
    s2 = r, # number of past inputs
    γ = nothing, # discarded, aids switching from n4sid
    W = :MOESP,
    zeroD = false,
    stable = true, 
    focus = :prediction,
    svd::F1 = svd!,
    scaleU = true,
    Aestimator::F2 = \,
    Bestimator::F3 = \,
    weights = nothing,
) where {F1,F2,F3}

    y = copy(permutedims(yo, (2,1,3)));
    u = copy(permutedims(uo, (2,1,3)));

    nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
    if scaleU
        CU = std(u, dims=1)
        u ./= CU
    end
    t, p = size(y, 1), size(y, 2)
    m = size(u, 2)
    reps = size(u,3)
    t0 = max(s1,s2)+1
    s = s1*p + s2*m
    N = t - r + 1 - t0

    # @views @inbounds function hankel(u::Matrix, t0, r)
    #     d = size(u, 2)
    #     H = zeros(eltype(u), r * d, N)
    #     for ri = 1:r, Ni = 1:N
    #         H[(ri-1)*d+1:ri*d, Ni] = u[t0+ri+Ni-2, :] # TODO: should start at t0
    #     end
    #     H
    # end

    @views @inbounds function hankel(u::Array, t0, r)
        d, reps = size(u, 2), size(u,3)

        H = zeros(r * d, N, reps)
        for rr = 1:reps
            for ri = 1:r, Ni = 1:N
                H[(ri-1)*d+1:ri*d, Ni, rr] = u[t0+ri+Ni-2, :, rr] # TODO: should start at t0
            end
        end
        rH = reshape(H, r*d, N*reps)
        return rH
    end



    # 1. Form G  (10.103). (10.100). (10.106). (10.114). and (10.108).

    Y = hankel(y, t0, r) # these go forward in time
    U = hankel(u, t0, r) # these go forward in time
    # @assert all(!iszero, Y) # to be turned off later
    # @assert all(!iszero, U) # to be turned off later
    # @assert size(Y) == (r*p, N)
    # @assert size(U) == (r*m, N)



    φs(t,rr) = [ # 10.114
        y[t-1:-1:t-s1, :,rr] |> vec # QUESTION: not clear if vec here or not, Φ should become s × N, should maybe be transpose before vec, but does not appear to matter
        u[t-1:-1:t-s2, :,rr] |> vec
    ]
    Φrr = zeros(s, N, reps)
    for rr = 1:reps
        Φrr[:,:,rr] = reduce(hcat, [φs(t,rr) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value
    end
    Φ = reshape(Φrr, s, N*reps)
    
    # Φ = reduce(hcat, [φs(t) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value
    # @assert size(Φ) == (s, N)

    # println("size Y = $(size(Y)), size U = $(size(U)), size Φ = $(size(Φ))")

    UΦY = [U; Φ; Y]
    l = lq!(UΦY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    # @assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q)), if this fails, you may need to lower the prediction horizon r which is currently set to $r"
    Uinds = 1:size(U,1)
    Φinds = (1:size(Φ,1)) .+ Uinds[end]
    Yinds = (1:size(Y,1)) .+ (Uinds[end]+s)
    @assert Yinds[end] == p*r+m*r+s
    L1 = L[Uinds, Uinds]
    L2 = L[s1*p+(r+s2)*m+1:end, 1:s1*p+(r+s2)*m+p]
    L21 = L[Φinds, Uinds]
    L22 = L[Φinds, Φinds]
    L32 = L[Yinds, Φinds]
    Q1 = Q[Uinds, :]
    Q2 = Q[Φinds, :]

    Ĝ = L32*(L22\[L21 L22])*[Q1; Q2] # this G is used for N4SID weight, but also to form Yh for all methods
    # 2. Select weighting matrices W1 (rp × rp)
    # and W2 (p*s1 + m*s2 × α) = (s × α)
    @assert size(Ĝ, 1) == r*p
    if W ∈ (:MOESP, :N4SID)
        if W === :MOESP
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ*ΠUt
            G = L32*Q2 #* 1/N# QUESTION: N does not appear to matter here
        elseif W === :N4SID
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ
            G = Ĝ #* 1/N
        end
    elseif W ∈ (:IVM, :CVA)
        if W === :IVM
            # YΠUt = proj_hr(Y, U)
            UY = [U; Y]
            Yinds = (1:size(Y,1)) .+ size(U,1)
            YΠUt = proj_hr(UY, Yinds)
            G = YΠUt*Φ' #* 1/N # 10.109, pr×s # N does not matter here
            @assert size(G) == (p*r, s)
            W1 = sqrt(Symmetric(pinv(1/N * (YΠUt*Y')))) |> real
            W2 = sqrt(Symmetric(pinv(1/N * Φ*Φ'))) |> real
            G = W1*G*W2
            @assert size(G, 1) == r*p
        elseif W === :CVA
            W1 = L[Yinds,[Φinds; Yinds]]
            ull1,sll1 = svd!(W1)
            sll1 = Diagonal(sll1[1:r*p])
            Or,Sn = svd!(pinv(sll1)*ull1'*L32)
            Or = ull1*sll1*Or
            # ΦΠUt = proj(Φ, U)
            # W1 = pinv(sqrt(1/N * (YΠUt*Y'))) |> real
            # W2 = pinv(sqrt(1/N * ΦΠUt*Φ')) |> real
            # G = W1*G*W2
        end
        # @assert size(W1) == (r*p, r*p)
        # @assert size(W2, 1) == p*s1 + m*s2
    else
        throw(ArgumentError("Unknown choice of W"))
    end

    # 3. Select R and define Or = W1\U1*R
    sv = W === :CVA ? svd(L32) : svd(G)
    if nx === :auto
        nx = sum(sv.S .> sqrt(sv.S[1] * sv.S[end]))
        verbose && @info "Choosing order $nx"
    end
    n = nx
    S1 = sv.S[1:n]
    R = Diagonal(sqrt.(S1))
    if W !== :CVA
        U1 = sv.U[:, 1:n]
        V1 = sv.V[:, 1:n]
        Or = W1\(U1*R)
    end
    
    fve = sum(S1) / sum(sv.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    C = Or[1:p, 1:n]
    A = Aestimator(Or[1:p*(r-1), 1:n] , Or[p+1:p*r, 1:n])
    if !all(e->abs(e)<=1, eigvals(A))
        verbose && @info "A matrix unstable, stabilizing by reflection"
        A = reflectd(A)
    end

    P, K, Qc, Rc, Sc = find_PK(L1,L2,Or,n,p,m,r,s1,s2,A,C)

    # 4. Estimate B, D, x0 by linear regression
    # B,D,x0 = find_BD(A, (focus === :prediction)*K, C, transpose(u), transpose(y), m, zeroD, Bestimator, weights)
    B,D,x0 = find_BD(A, (focus === :prediction)*K, C, reshape(permutedims(u, (2,1,3)), (m, t*reps)),  reshape(permutedims(y, (2,1,3)), (p, t*reps)), m, zeroD, Bestimator, weights)
    # TODO: iterate find C/D and find B/D a couple of times

    if scaleU
        B ./= CU
        D ./= CU
    end

    # 5. If noise model, form Xh from (10.123) and estimate noise contributions using (10.124)
    # Yh,Xh = let
    #     # if W === :N4SID
    #     # else
    #     # end
    #     svi = svd(Ĝ) # to form Yh, use N4SID weight
    #     U1i = svi.U[:, 1:n]
    #     S1i = svi.S[1:n]
    #     V1i = svi.V[:, 1:n]
    #     Yh = U1i*Diagonal(S1i)*V1i' # This expression only valid for N4SID?
    #     Lr = R\U1i'
    #     Xh = Lr*Yh
    #     Yh,Xh
    # end


    # CD = Yh[1:p, :]/[Xh; !zeroD*U[1:m, :]]  
    # C2 = CD[1:p, 1:n]
    # D2 = CD[1:p, n+1:end]
    # AB = Xh[:, 2:end]/[Xh[:, 1:end-1]; U[1:m, 1:end-1]]
    # A2 = AB[1:n, 1:n]
    # B2 = AB[1:n, n+1:end]
    
    # N4SIDStateSpace(ss(A,  B,  C,  D, Nothing), Qc,Rc,Sc,K,P,x0,sv,fve)

    sys = (A = A, B = B, C = C, D = D, Q = Qc, R = Rc, S = Sc, K = K, P = P, x = x0, fve = fve)
   return sys

end





function estimate_n4sid(y::Array{Float64}, u::Array{Float64}, k::Int64, nx::Int64)

    # Check if u and y has the same length
    if size(u,2) != size(y,2)
        error("Input(u) and output(y) are not the same length")
    end

    # Get the size of output
    l,ny,n_trials = size(y);
    # Get the size of input
    m,nu = size(u);


    # Create the amout of columns
    j = ny-2*k+1;

    # Create hankel matrecies
    Utt = zeros(size(u,1)*2*k, j, n_trials);
    Ytt = zeros(size(y,1)*2*k, j, n_trials);

    @inbounds @simd for tt in axes(Ytt,3)

        Utt[:,:,tt] .= mi_hankel(u[:,:,tt], 2*k, j);
        Ytt[:,:,tt] .= mi_hankel(y[:,:,tt], 2*k, j);

    end

    U = deepcopy(reshape(Utt, size(Utt,1), size(Utt,2)*size(Utt,3)));
    Y = deepcopy(reshape(Ytt, size(Ytt,1), size(Ytt,2)*size(Ytt,3)));

    # Create the past, future hankel matrices
    Up = U[1:(k*m), :];
    Uf = U[(k*m+1):(2*k*m), :];

    # For output too.
    Yp = Y[1:(k*l),:];
    Yf = Y[(k*l+1):(2*k*l),:];

    # Get the size of the hankel matrecies
    km = size(Up, 1);
    kl = size(Yp, 1);
  

    # Create the Wp matrix
    Wp = [Up; Yp];

    # Do QR decomposition
    UY = [Uf;Up;Yp;Yf];

    @inline @inbounds R32, R31, R22, R21, R11 = mi_QR(UY, km, kl);

    # Orthogonal projections
    try
        global Rdiv = R32/R22;
    catch
        @warn "R32 rdiv R22 failed"
        global Rdiv = R32*pinv(R22);
    end
    AbC  = Rdiv*Wp;


    # Do singular value decomposition
    @inline @inbounds U, s_diag, V = svd(AbC, alg=LinearAlgebra.QRIteration());
    ssid_sv = s_diag;
    Sd = diagm(s_diag);

    # Do model reduction
    U1 = U[:, 1:nx];
    S1 = Sd[1:nx, 1:nx];
    # V1 = V[:, 1:nx];

    # Create the observability matrix
    OBSV = U1*sqrt(S1);

    # Find Cd and Ad from observability matrix
    Cd = OBSV[1:l, 1:nx];
    try
        global Ad = OBSV[1:l*(k-1), 1:nx] \ OBSV[(l+1):l*k, 1:nx];
    catch
        @warn "OBSV ldiv OBSV failed"
        global Ad = pinv(OBSV[1:l*(k-1), 1:nx])*OBSV[(l+1):l*k, 1:nx];
    end

    # Find D and B matrix
    try
        global DBi = (R31 - Rdiv*R21) / R11;
    catch
        @warn "(R31 - Rdiv*R21) rdiv R11 failed"
        global DBi = (R31 - Rdiv*R21)*pinv(R11);        
    end

    # Split up so we first can find Dd and Bd
    DB = deepcopy(DBi[:,1:m]);

    # Collet CA^kB'Sd only
    DB0 = DB[1:l,:];
    DB1 = DB[l+1:2*l,:];
    DB2 = DB[2*l+1:3*l,:];
    DB3 = DB[3*l+1:4*l,:];
    DB4 = DB[4*l+1:5*l,:];

    # This is Db matrix
    Dd = deepcopy(DB0);

    # We can call this CAB
    CAB = [ DB1 DB2;
            DB2 DB3;
            DB3 DB4];

    # Extract ony the CAB...CA^kB parts of DB  
    CAB_OBSV = OBSV[1:3*l,:];

    # Create the controllability matrix
    try
        global CTRB = CAB_OBSV\CAB;
    catch
        @warn "CAB_OBSV ldiv CAB failed"
        global CTRB = pinv(CAB_OBSV)*CAB;
    end

    # Find Bd matrix now
    Bd = CTRB[:, 1:m];


    return Ad, Bd, Cd, Dd

end


function mi_QR(UY::Matrix{Float64}, km::Int64, kl::Int64)

    LQ = lq!(UY);
    L = LQ.L;

    # Split
    R11 = L[1:km, 1:km];
    L21 = L[(km+1):(2*km), 1:km];
    L22 = L[(km+1):(2*km), (km+1):(2*km)];
    L31 = L[(2*km+1):(2*km+kl), 1:km];
    L32 = L[(2*km+1):(2*km+kl), (km+1):(2*km)];
    L33 = L[(2*km+1):(2*km+kl), (2*km+1):(2*km+kl)];
    R31 = L[(2*km+kl+1):(2*km+2*kl), 1:km];
    L42 = L[(2*km+kl+1):(2*km+2*kl), (km+1):(2*km)];
    L43 = L[(2*km+kl+1):(2*km+2*kl), (2*km+1):(2*km+kl)];
    # L44 = L[(2*km+kl+1):(2*km+2*kl), (2*km+kl+1):(2*km+2*kl)];

    # R11 = L11;
    R21 = [L21; L31];
    R22 = [L22 zeros(km,kl); L32 L33];
    # R31 = L41;
    R32 = [L42 L43];
    
    return R32, R31, R22, R21, R11

end



function mi_hankel(g::Matrix{Float64}, i::Int64, j::Int64) 
    # U = mi_hankel(u, 2*k, j)

    # Get size
    l = size(g, 1);

    # Create a large hankel matrix
    H = zeros(Float64,l*i,j);


    for k=1:i
        H[((k-1)*l+1):(k*l),:] .= g[:,k:(k+j-1)];
    end

    return H

end







function estimate_clmoesp(r::Array{Float64}, y::Array{Float64}, u::Array{Float64}, k::Int64, n::Int64)
    # ● System Identification - CL-MOESP method
    # [A, B, C, D] = mf_clmoesp(r, u, y, k, n)
    # r: reference signal
    # u: input signal
    # y: output signal
    # k: number of rows in the data matrix
    # n: order of the state variables
    # A(n*n), B(n*m), C(p*n), D(p*m): matrices of the state-space representation

    # ● Generation of input and output data matrices
    m = size(u, 1)      # order of input and reference values, size of the first row of u
    p = size(y, 1)      # order of output
    km = k * m
    kp = k * p
    n_trials = size(u, 3)



    # Past data
    Rp_tt = zeros(m * k, size(r, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Rp_tt[m * (i - 1) + 1:m * i, :, tt] = r[:, i:size(r, 2) - 2 * k + 1 + i - 1, tt]
        end
    end
    Rp = deepcopy(reshape(Rp_tt, size(Rp_tt,1), size(Rp_tt,2)*size(Rp_tt,3)));


    Up_tt = zeros(m * k, size(u, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Up_tt[m * (i - 1) + 1:m * i, :, tt] = u[:, i:size(u, 2) - 2 * k + 1 + i - 1, tt]
        end
    end
    Up = deepcopy(reshape(Up_tt, size(Up_tt,1), size(Up_tt,2)*size(Up_tt,3)));



    Yp_tt = zeros(p * k, size(y, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Yp_tt[p * (i - 1) + 1:p * i, :, tt] = y[:, i:size(y, 2) - 2 * k + 1 + i - 1, tt]
        end
    end
    Yp = deepcopy(reshape(Yp_tt, size(Yp_tt,1), size(Yp_tt,2)*size(Yp_tt,3)));


    # Future data
    Rf_tt = zeros(m * k, size(r, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Rf_tt[m * (i - 1) + 1:m * i, :, tt] = r[:, i + k:size(r, 2) - 2 * k + 1 + i + k - 1, tt]
        end
    end
    Rf = deepcopy(reshape(Rf_tt, size(Rf_tt,1), size(Rf_tt,2)*size(Rf_tt,3)));


    Uf_tt = zeros(m * k, size(u, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Uf_tt[m * (i - 1) + 1:m * i, :, tt] = u[:, i + k:size(u, 2) - 2 * k + 1 + i + k - 1, tt]
        end
    end
    Uf = deepcopy(reshape(Uf_tt, size(Uf_tt,1), size(Uf_tt,2)*size(Uf_tt,3)));
    

    Yf_tt = zeros(p * k, size(y, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Yf_tt[p * (i - 1) + 1:p * i, :, tt] = y[:, i + k:size(y, 2) - 2 * k + 1 + i + k - 1, tt]
        end
    end
    Yf = deepcopy(reshape(Yf_tt, size(Yf_tt,1), size(Yf_tt,2)*size(Yf_tt,3)));



    N = size(Up, 2)

    # ● LQ decomposition
    tmp1 = [Rp; Rf; Up; Uf; Yf]
    # Q, L = qr(tmp1')
    # Q = Q'
    # L = L'

    @inbounds @inline LQ = lq!(tmp1);
    L = LQ.L;

    # com_st = L[1, 1]
    # global c1 = 0
    # for i = 1:N
    #     if any(abs.(L[1:km, i]) < abs(com_st * 1e-5))
    #         global c1 = i - 1
    #         break
    #     end
    # end
    # global c2 = 0
    # for j = i:N
    #     if any(abs.(L[km + 1:2 * km, j]) < abs(com_st * 1e-5))
    #         global c2 = j - 1
    #         break
    #     end
    # end

    c1 = size(Up, 1)
    c2 = size(Yp, 1)

    L31 = L[2 * km + 1:3 * km, 1:c1]
    L32 = L[2 * km + 1:3 * km, c1 + 1:c2]
    L41 = L[3 * km + 1:4 * km, 1:c1]
    L42 = L[3 * km + 1:4 * km, c1 + 1:c2]
    L51 = L[4 * km + 1:4 * km + kp, 1:c1]
    L52 = L[4 * km + 1:4 * km + kp, c1 + 1:c2]

    # ● Calculation of gamma
    P1 = L31 - (L31 * L41' + L32 * L42') * pinv(L41 * L41' + L42 * L42') * L41;
    P2 = L32 - (L31 * L41' + L32 * L42') * pinv(L41 * L41' + L42 * L42') * L42;
    gamma = (L51 * P1' + L52 * P2') * pinv(sqrt(P1 * P1' + P2 * P2'));

    # ● SVD (sU*sS*sV' = gamma)
    # sU, sS, sV = svd(gamma)
    @inbounds sSVD = svd(gamma, alg=LinearAlgebra.QRIteration());
    sU = sSVD.U;
    
    sU1 = sU[:, 1:n]
    # sS1 = sS[1:n, 1:n]
    # sV1 = sV[:, 1:n]  # note that this is the transpose
    sU2 = sU[:, n + 1:end]

    # ● Calculation of A, B, C, D
    C = sU1[1:p, :]
    try
        global A = sU1[1:p * (k - 1), :] \ sU1[p + 1:kp, :]
    catch
        println("U1[1:p * (k - 1), :] ldiv U1[p + 1:kp, :] failed")
        global A = pinv(sU1[1:p * (k - 1), :]) * sU1[p + 1:kp, :]
    end
    sU2T = sU2'
    sBT = sU2T * (L51 * L41' + L52 * L42') * pinv(L41 * L41' + L42 * L42')
    DB1 = [I(p) zeros(p, n); zeros((k - 1) * p, p) sU1[1:(k - 1) * p, :]]
    DB2 = zeros(size(sU2T, 1) * k, kp)
    DB3 = zeros(size(sBT, 1) * k, m)
    alpha_r = size(sU2T, 1)
    alpha_c = p
    beta_r = size(sBT, 1)
    beta_c = m
    for i = 1:k
        for j = 1:i
            ii = i - (j - 1)
            jj = j
            DB2[(ii - 1) * alpha_r + 1:ii * alpha_r, (jj - 1) * alpha_c + 1:jj * alpha_c] = sU2T[:, (i - 1) * p + 1:i * p]
        end
        DB3[(i - 1) * beta_r + 1:i * beta_r, :] = sBT[:, (i - 1) * m + 1:i * m]
    end

    try
        global DB = DB1 \ (DB2 \ DB3)
    catch
        println("DB1 ldiv (pinv(DB2) ldiv DB3) failed")
        global DB = pinv(DB1) * pinv(DB2) * DB3
    end
    D = DB[1:p, :]
    B = DB[p + 1:p + n, :]

    return A, B, C, D
end



function estimate_moesp(u, y, R, n)

    # Number of inputs and outputs:
    m, nu = size(u)
    if nu < m
        u = u'
        m, nu,_ = size(u)
    end
    
    p, ny = size(y)
    if ny < p
        y = y'
        p, ny,_ = size(y)
    end
    n_trials = size(y,3)
    
    Ncol = ny - R + 1  # Calculate number of COLUMNS BLOCK HANKEL
    
    # Make BLOCK HANKEL INPUTS U-OUTPUTS Y
    Utt = zeros(m * R, Ncol, n_trials)
    Ytt = zeros(p * R, Ncol, n_trials)
    for tt = 1:n_trials
        Utt[:,:,tt] = blkhank(u[:,:,tt], R, Ncol)
        Ytt[:,:,tt] = blkhank(y[:,:,tt], R, Ncol)
    end
    U = deepcopy(reshape(Utt, size(Utt,1), size(Utt,2)*size(Utt,3)));
    Y = deepcopy(reshape(Ytt, size(Ytt,1), size(Ytt,2)*size(Ytt,3)));

    
    km = size(U, 1)
    kp = size(Y, 1)  # Rows of U and Y

    UY = [U; Y]
    LQ = lq(UY)
    L = LQ.L  # LQ decomposition  Eq. 3.45  qr: Orthogonal-triangular decomposition.
    L11 = L[1:km, 1:km]
    L21 = L[km + 1:km + kp, 1:km]
    L22 = L[km + 1:km + kp, km + 1:km + kp]
    UU, SS, VV = svd(L22)  # Eq. 3.48  Singular Value Descomposition
    s1 = Diagonal(SS)

    # n=4;
    U1 = UU[:, 1:n]  # n is known, as you can see last equation
    Ok = U1 * sqrt.(s1[1:n, 1:n])  # SQRTM     Matrix square root.  Eq. 3.49

    # Matrices A and C
    C = Ok[1:p, 1:n]  # Eq. (6.41)
    A = pinv(Ok[1:p * (R - 1), 1:n]) * Ok[p + 1:p * R, 1:n]  # Eq.3.51

    # Matrices B and D
    U2 = UU[:, n + 1:size(UU', 1)]
    Z = U2' * L21 / L11  # Eq. 3.53
    XX = []
    RR = []

    for j = 1:R
        XX = cat(XX, Z[:, m * (j - 1) + 1:m * j], dim=3)
        Okj = Ok[1:p * (R - j), :]
        Rj = [zeros(p * (j - 1), p)     zeros(p * (j - 1), n);
              I(p)                      zeros(p, n);
              zeros(p * (R - j), p)     Okj]
        RR = [RR; U2' * Rj]
    end

    DB = RR \ XX  # Eq. 3.57
    D = DB[1:p, :]
    B = DB[p + 1:size(DB, 1), :]


    return A,B,C,D

end


function blkhank(y, i, j)
    # Make a (block)-row vector out of y
    l, nd = size(y)
    if nd < l
        y = y'
        l, nd = size(y)
    end
    
    # Check dimensions
    if i < 0
        error("blkHank: i should be positive")
    end
    if j < 0
        error("blkHank: j should be positive")
    end
    if j > nd - i + 1
        error("blkHank: j too big")
    end
    
    # Make a block-Hankel matrix
    H = zeros(l * i, j)
    for k = 1:i
        H[(k - 1) * l + 1:k * l, :] = y[:, k:k + j - 1]
    end
    
    return H
end






function estimate_n4sid2(y, u, k, n)
    #--------------------------------------------------------------------------
    # ● System Identification - N4SID Method (Supports SISO and MIMO)
    #--------------------------------------------------------------------------
    # [A, B, C, D] = estimate_n4sid2(y, u, k, n)
    # y: output
    # u: input
    # k: number of rows in the data matrix
    # n: order of the state variables
    # A(n*n), B(n*m), C(p*n), D(p*m): matrices of the state-space representation
    
    #--------------------------------------------------------------------------
    # ● Generation of Input and Output Data Matrices
    #--------------------------------------------------------------------------
    # Data matrices for SISO case
    # % Past input
    # tmp1 = u[:, 1:k]
    # tmp2 = u[:, k:end-k]
    # Up = hankel(tmp1, tmp2)
    # tmp1 = y[:, 1:k]
    # tmp2 = y[:, k:end-k]
    # Yp = hankel(tmp1, tmp2)
    # % Future input
    # tmp1 = u[:, k+1:2*k]
    # tmp2 = u[:, 2*k:end]
    # Uf = hankel(tmp1, tmp2)
    # tmp1 = y[:, k+1:2*k]
    # tmp2 = y[:, 2*k:end]
    # Yf = hankel(tmp1, tmp2)
    
    # Data matrices for MIMO case (using this one)
    m = size(u, 1)      # input dimension, size of the first row of u
    p = size(y, 1)      # output dimension
    km = k * m
    kp = k * p


    # Past data
   
    Up_tt = zeros(m * k, size(u, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Up_tt[m * (i - 1) + 1:m * i, :, tt] = u[:, i:size(u, 2) - 2 * k + 1 + i - 1, tt]
        end
    end
    Up = deepcopy(reshape(Up_tt, size(Up_tt,1), size(Up_tt,2)*size(Up_tt,3)));

    Yp_tt = zeros(p * k, size(y, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Yp_tt[p * (i - 1) + 1:p * i, :, tt] = y[:, i:size(y, 2) - 2 * k + 1 + i - 1, tt]
        end
    end
    Yp = deepcopy(reshape(Yp_tt, size(Yp_tt,1), size(Yp_tt,2)*size(Yp_tt,3)));


    # Future data
    
    Uf_tt = zeros(m * k, size(u, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Uf_tt[m * (i - 1) + 1:m * i, :, tt] = u[:, i + k:size(u, 2) - 2 * k + 1 + i + k - 1, tt]
        end
    end
    Uf = deepcopy(reshape(Uf_tt, size(Uf_tt,1), size(Uf_tt,2)*size(Uf_tt,3)));
    

    Yf_tt = zeros(p * k, size(y, 2) - 2 * k + 1, n_trials)
    for tt = 1:n_trials
        for i = 1:k
            Yf_tt[p * (i - 1) + 1:p * i, :, tt] = y[:, i + k:size(y, 2) - 2 * k + 1 + i + k - 1, tt]
        end
    end
    Yf = deepcopy(reshape(Yf_tt, size(Yf_tt,1), size(Yf_tt,2)*size(Yf_tt,3)));

    N = size(Up, 2)
    

    #--------------------------------------------------------------------------
    # ● LQ Decomposition
    #--------------------------------------------------------------------------
    Wp = [Up; Yp]
    tmp1 = [Uf; Wp; Yf]  # [Uf; Up; Yp; Yf] is also acceptable
    LQ = lq(tmp1)
    L = LQ.L
    com_st = L[1, 1]  # component_standard: reference element
    r=km + N
    # for i = km + 1:N
    #     if L[km + 1:2 * km + kp, i] < abs(com_st * 10^(-5))
    #         r = i - 1
    #         break
    #     end
    # end
    L22 = L[km + 1:2 * km + kp, km + 1:km + r]
    L32 = L[2 * km + kp + 1:2 * km + 2 * kp, km + 1:km + r]
    
    # Th = L32 * (L22' * L22)^(-1) * L22' * Wp
    Th = L32 * pinv(L22) * Wp
    
    #--------------------------------------------------------------------------
    # ● SVD
    # n: order of the state variables
    #--------------------------------------------------------------------------
    sU, sS, sV = svd(Th)
    sU1 = sU[:, 1:n]
    sS1 = sS[1:n, 1:n]
    sV1 = sV[:, 1:n]  # note that this is the transpose
    
    #--------------------------------------------------------------------------
    # ● Calculation of the Extended Observability Matrix
    #--------------------------------------------------------------------------
    Ok = sU1 * sqrt(sS1)
    
    #--------------------------------------------------------------------------
    # ● Calculation of the State Vector
    #--------------------------------------------------------------------------
    Xk = (Ok' * Ok)^(-1) * Ok' * Th
    
    #--------------------------------------------------------------------------
    # ● Calculation of A, B, C, D
    #--------------------------------------------------------------------------
    XY = [Xk[:, 2:end]; y[:, k + 1:end - k]]  # [Xk+1; Yk]
    XU = [Xk[:, 1:end - 1]; u[:, k + 1:end - k]]  # [Xk; Uk]
    res = (XY * XU') * (XU * XU')^(-1)
    A = res[1:n, 1:n]
    B = res[1:n, 1 + n:end]
    C = res[1 + n:end, 1:n]
    D = res[1 + n:end, 1 + n:end]
    
    return A, B, C, D
end

    

    




















function task_refine_EM(S)
    # run EM just on noise terms

    @reset S.res.startTime_refine = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");


    # run EM on noise terms
    for ii = 1:S.prm.n_iter_refine

        # E-step
        NeSS.task_ESTEP!(S);

        # M-step [Refine Version]
        @reset S.mdl = deepcopy(NeSS.task_refine_MSTEP(S));

        # compute the complete data log-likelihood
        push!(S.res.refine_total_loglik, total_loglik(S));
        


        # confirm loglik is increasing
        if (ii > 1)  && (S.res.refine_total_loglik[ii] < S.res.refine_total_loglik[ii-1])
            println("warning: total loglik decreased")
        end


         # print total loglik every n iters
         if mod(ii,S.prm.print_iter) == 0
            println("[refine $(ii)] total ll: $(round(S.res.refine_total_loglik[ii],digits=2))")
        end

        # check for convergence
        if (ii > 1) && (abs(S.res.refine_total_loglik[ii] - S.res.refine_total_loglik[ii-1]) < 0.1)       # reasonable expectations for convergence
            println("")
            println("----- converged! -----")
            println("")
            break
        end

        # garbage collect every 5 iterations
        if (mod(ii,5) == 0) && Sys.islinux() 
            ccall(:malloc_trim, Cvoid, (Cint,), 0);
            ccall(:malloc_trim, Int32, (Int32,), 0);
            GC.gc(true);
        end





    end

    @reset S.res.mdl_refine = deepcopy(S.mdl);
    @reset S.res.endTime_refine = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");


    return S

end






function task_refine_MSTEP(S)::model_struct

   
    # initials ===============================================
    # Mean
    # W = ((S.est.xx_init + S.prm.lam_B0) \ S.est.xy_init)';
    # B0 = W[:, 1:S.dat.u0_dim]

    # # Covariance
    # Wxy = W*S.est.xy_init;
    # P0e = (S.est.yy_init - Wxy - Wxy' + X_A_Xt(S.est.xx_init, W) + W*S.prm.lam_B0*W' + (S.prm.df_P0 * S.prm.mu_P0)) / 
    #         ((S.est.n_init[1] + S.prm.df_P0) - size(S.est.xx_init,1));
    # P0 = format_noise(P0e, S.prm.P0_type);

    B0 = deepcopy(S.res.mdl_ssid.B0);
    P0 = deepcopy(S.res.mdl_ssid.P0);


    

    # dynamics ===============================================
    # Mean
    A = deepcopy(S.res.mdl_ssid.A);
    B = deepcopy(S.res.mdl_ssid.B);
    W = [A B];
    
    # Covariance
    Wxy = W*S.est.xy_dyn;
    Qe = (S.est.yy_dyn - Wxy - Wxy' + X_A_Xt(S.est.xx_dyn_PD[1], W) + W*S.prm.lam_AB*W' + (S.prm.df_Q * S.prm.mu_Q)) / 
        ((S.est.n_dyn[1] + S.prm.df_Q) - size(S.est.xx_dyn,1));

    Q = format_noise(Qe, S.prm.Q_type);



    # observations ===============================================
    # Mean
    W = deepcopy(S.res.mdl_ssid.C);
    C = W;

    # Covariance
    Wxy = W*S.est.xy_obs;
    Re = (S.est.yy_obs - Wxy - Wxy' + X_A_Xt(S.est.xx_obs_PD[1], W) + W*S.prm.lam_C*W' + (S.prm.df_R * S.prm.mu_R)) / 
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


