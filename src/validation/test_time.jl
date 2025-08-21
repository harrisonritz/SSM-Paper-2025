# test time

ROOT = ""

# imports
NeSS_dir = "$ROOT/HallM_NeSS/src"
save_dir = "$ROOT/HallM_NeSS/src";

# add paths
push!(LOAD_PATH, pwd());
push!(LOAD_PATH, "$(pwd())/../");
push!(LOAD_PATH, NeSS_dir);
if Sys.isapple() == 0
    println(LOAD_PATH)
end

# load modules
using Revise
using NeSS

# load packages
using Accessors
using Random
using LinearAlgebra
using Dates
using Plots

using PDMats
using BenchmarkTools
using JET
using StatProfilerHTML


# disable multithreaded BLAS
BLAS.set_num_threads(1)
@show VERSION

# parameters
Random.seed!(42)


model_name = "test_time";
max_iter_em = 10;

n_train = 450;
n_test =  450;
n_times = 150;

x_dim = 50;
u_dim = 50;
n_chans = 50;
u0_dim = 1;


Q_noise = .01;
R_noise = .001;
P0_noise = .1;


S = core_struct(
    prm=param_struct(
        model_name = model_name,
        max_iter_em = max_iter_em;
        NeSS_dir = NeSS_dir,
        save_dir = save_dir,
        test_iter = 100,
        PCA_ratio = 1.0,
        pt_list = 1:30,
        ), 

    dat=data_struct(
        n_train = n_train,
        n_test =  n_test,
        n_times = n_times,
        x_dim = x_dim,
        n_chans = n_chans,
        y_dim = n_chans,
        u_dim = u_dim,
        u0_dim = u0_dim,
        ),

    res=results_struct(),

    est=estimates_struct(),

    mdl=model_struct(),

);


# Generate Paramters
sim = NeSS.generate_lds_parameters(S, Q_noise, R_noise, P0_noise);
sim2 = NeSS.generate_lds_parameters(S, Q_noise, R_noise, P0_noise);



# simulate
u_train = randn(u_dim, n_times, n_train);
u_train[1,:,:] .= 1.0;
u0_train = randn(u0_dim, n_train);
u0_train[1,:] .= 1.0;

x_sim, y_train  = NeSS.generate_lds_trials( sim.A, sim.B, sim.Q,
                                    sim.C, sim.R, 
                                    sim.B0, sim.P0,
                                    u_train,u0_train,  
                                    n_times, n_train);


u_test = randn(u_dim, n_times, n_test);
u_test[1,:,:] .= 1.0;
u0_test = randn(u0_dim, n_test);
u0_test[1,:] .= 1.0;
x_test, y_test  = NeSS.generate_lds_trials( sim.A, sim.B, sim.Q,
                                        sim.C, sim.R, 
                                        sim.B0, sim.P0,
                                        u_test,u0_test,  
                                        n_times, n_test);


                                        # y'y

@reset S.dat.y_train_orig = y_train;
@reset S.dat.y_train = y_train;
@reset S.dat.u_train = u_train;
@reset S.dat.u0_train = u0_train;

@reset S.dat.y_test_orig = y_train;
@reset S.dat.y_test = y_test;
@reset S.dat.u_test = u_test;
@reset S.dat.u0_test = u0_test;



 # pack into structures
 @reset S.mdl = deepcopy(set_model(
    A = .99*sim.A,
    B = sim2.B,
    Q = sim2.Q,

    C = sim2.C,
    R = sim2.R,

    B0 = sim2.B0,
    P0 = sim2.P0,
    ));


@reset S.est = deepcopy(NeSS.set_estimates(S));


# whiten
S = deepcopy(NeSS.whiten_y(S));










# plot loglik trajectory =============================================================
@reset S.mdl = deepcopy(set_model(
    A = .99*sim.A,
    B = sim2.B,
    Q = sim.Q,

    C = sim.C,
    R = sim.R,

    B0 = sim.B0,
    P0 = sim.P0,
    ));
@reset S.est = deepcopy(set_estimates(S));
@reset S.res=results_struct()
@reset S.prm.max_iter_em = 2;
S = deepcopy(NeSS.task_EM(S));  

p1=plot(S.res.total_loglik, legend=false, title="total loglik", xlabel="EM Iteration", ylabel="Loglik")
p2=plot(S.res.test_loglik, legend=false, title="test loglik", xlabel="EM Iteration", ylabel="Loglik")
plot(p1,p2, layout=(1,2), size=(800,400))






# BENCHMARK E-STEP =============================================================


@reset S.res=results_struct()
@reset S.est = deepcopy(set_estimates(S));
@reset S.prm.max_iter_em = 200;

# BENCHMARK
bench_estep = @benchmark NeSS.task_ESTEP!($S); # 30ms
display(bench_estep)

# PROFILE
VSCodeServer.@profview NeSS.task_ESTEP!(S)







# benchmark full EM =============================================================
@reset S.mdl = deepcopy(set_model(
    A = .99*sim.A,
    B = sim2.B,
    Q = sim2.Q,

    C = sim2.C,
    R = sim2.R,

    B0 = sim2.B0,
    P0 = sim2.P0,
    ));

@reset S.res=results_struct()
@reset S.est = deepcopy(set_estimates(S));
@reset S.prm.max_iter_em = 10;


bench = @benchmark NeSS.task_EM($S) 
display(bench)

bench = @benchmark NeSS.test_loglik($S)
display(bench)




# benchmark EM components =============================================================
@reset S.mdl = deepcopy(set_model(
    A = .99*sim.A,
    B = sim.B,
    Q = sim.Q,

    C = sim.C,
    R = sim.R,

    B0 = sim.B0,
    P0 = sim.P0,
    ));


@reset S.res=results_struct()
@reset S.est = deepcopy(set_estimates(S));
@reset S.prm.max_iter_em = 200;


bench_estep = @benchmark NeSS.task_ESTEP!($S); # 30ms
display(bench_estep)

bench_mstep = @benchmark NeSS.task_MSTEP($S);
display(bench_mstep)


bench_loglik = @benchmark NeSS.total_loglik($S);
display(bench_loglik)


bench_init = @benchmark  NeSS.init_ssid($S);
display(bench_init)





# profile EM =============================================================
@reset S.prm.max_iter_em = 200;
@reset S.res=results_struct()
@reset S.est = deepcopy(set_estimates(S));
VSCodeServer.@profview NeSS.task_EM(S)


@reset S.res=results_struct()
@reset S.est = deepcopy(set_estimates(S));
@reset S.prm.max_iter_em = 200;
VSCodeServer.@profview NeSS.task_ESTEP!(S)



# profile init
VSCodeServer.@profview NeSS.init_ssid(S)




Ah = hermitianpart(randn(50,50))
VSCodeServer.@profview begin
    for ii = 1:1000
        Apd = NeSS.tol_PD(Ah);
    end
end


@code_warntype NeSS.task_EM(S)

@report_opt vscode_console_output=stdout NeSS.smooth_mean!(S)

@descend NeSS.task_EM(S)




## profile init =============================================================

using ControlSystemIdentification
using ControlSystems




function proj(Yi, U)
    UY = [U; Yi]
    l = lq!(UY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    Uinds = 1:size(U,1)
    Yinds = (1:size(Yi,1)) .+ Uinds[end]
    # if Yi === Y
        # @assert size(Q) == (p*r+m*r, N) "size(Q) == $(size(Q))"
        # @assert Yinds[end] == p*r+m*r
    # end
    L22 = L[Yinds, Yinds]
    Q2 = Q[Yinds, :]
    L22*Q2
    # Ym = L22*Q2
    # return Ym
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
        @error "Failed to estimate kalman gain, got error" e
        P = I(n)
        K = zeros(n, p)
    end
    P, K, Q, R, S
end







function find_BD_hr(A,K,C,U,Y,m, zeroD=false, estimator=\, weights=nothing)
    T = eltype(A)
    nx = size(A, 1)
    p = size(C, 1)
    N = size(U, 2)
    A = A-K*C
    ε = lsim(ss(A,K,C,0,1), Y)[1] # innovation sequence
    φB = zeros(p, N, m*nx)
    @inbounds for (j,k) in Iterators.product(1:nx, 1:m)
        E = zeros(nx)
        E[j] = 1
        fsys = ss(A, E, C, 0, 1)
        u = U[k:k,:]
        uf = lsim(fsys, u)[1]
        r = (k-1)*nx+j
        φB[:,:,r] = uf 
    end
    φx0 = zeros(p, N, nx)
    x0u = zeros(1, N)
    @inbounds for (j,k) in Iterators.product(1:nx, 1:1)
        E = zeros(nx)
        x0 = zeros(nx); x0[j] = 1
        fsys = ss(A, E, C, 0, 1)
        uf = lsim(fsys, x0u; x0)[1]
        r = (k-1)*nx+j
        φx0[:,:,r] = uf 
    end
    if !zeroD
        φD = zeros(p, N, m*p)
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

    @inbounds φ3 = zeroD ? cat(φB, φx0, dims=Val(3)) : cat(φB, φx0, φD, dims=Val(3))
    # φ4 = permutedims(φ3, (1,3,2))
    φ = reshape(φ3, p*N, :)
    if weights === nothing
        BD = estimator(φ, vec(Y .- ε))
    else
        BD = estimator(φ, vec(Y .- ε), weights)
    end
    B = copy(reshape(BD[1:m*nx], nx, m))
    x0 = BD[m*nx .+ (1:nx)]
    if zeroD
        D = zeros(T, p, m)
    else
        D = reshape(BD[end-p*m+1:end], p, m)
        B .+= K*D
    end
    B,D,x0
end





















# setup ============
y = deepcopy(S.dat.y_train);
u = deepcopy(S.dat.u_train);

nx =  deepcopy(S.dat.x_dim);
k = deepcopy(S.prm.ssid_lag);

yl = reshape(y, size(y,1), size(y,2)*size(y,3));
ul = reshape(u, size(u,1), size(u,2)*size(u,3));
id = prefilter(iddata(yl, ul, .004), .01, Inf)


# subspaceid 

data = id
verbose = true
r = nx
s1 = r # number of past outputs
s2 = r # number of past inputs
γ = nothing # discarded, aids switching from n4sid
W = :IVM
zeroD = true
stable = true
focus = :prediction
scaleU = true
Aestimator = \
Bestimator = \
weights = nothing


nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
y, u = transpose(copy(output(data))), transpose(copy(input(data)))
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

UΦY = [U; Φ; Y]
l = lq!(UΦY)
L = l.L
Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
@assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q)), if this fails, you may need to lower the prediction horizon r which is currently set to $r"
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

        @benchmark proj($Y, $U)

        VSCodeServer.@profview proj(Y, U)

        YΠUt = proj(Y, U)
        G = YΠUt*Φ' #* 1/N # 10.109, pr×s # N does not matter here
        @assert size(G) == (p*r, s)
        W1 = sqrt(Symmetric(pinv(inv(N) * (YΠUt*Y')))) |> real
        W2 = sqrt(Symmetric(pinv(inv(N) * Φ*Φ'))) |> real
        G = W1*G*W2
        @assert size(G, 1) == r*p
    elseif W === :CVA
        W1 = L[Yinds,[Φinds; Yinds]]
        ull1,sll1 = svd(W1)
        sll1 = Diagonal(sll1[1:r*p])
        Or,Sn = svd(pinv(sll1)*ull1'*L32)
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

@views @inbounds @inline P, K, Qc, Rc, Sc = find_PK(L1,L2,Or,n,p,m,r,s1,s2,A,C)

@benchmark find_PK($L1,$L2,$Or,$n,$p,$m,$r,$s1,$s2,$A,$C)


@benchmark find_BD_hr($A, $K, $C, transpose($u), transpose($y), $m, $zeroD, $Bestimator, $weights)


@views @inbounds @inline B,D,x0 = find_BD_hr(A, (focus === :prediction)*K, C, transpose(u), transpose(y), m, zeroD, Bestimator, weights)
# TODO: iterate find C/D and find B/D a couple of times




## find_BD_hr

U = transpose(u)
Y = transpose(y)
estimator=\

T = eltype(A)
nx = size(A, 1)
p = size(C, 1)
N = size(U, 2)
A = A-K*C
ε = lsim(ss(A,K,C,0,1), Y)[1] # innovation sequence
φB = zeros(p, N, m*nx)
@inbounds for (j,k) in Iterators.product(1:nx, 1:m)
    E = zeros(nx)
    E[j] = 1
    fsys = ss(A, E, C, 0, 1)
    u = U[k:k,:]
    uf = lsim(fsys, u)[1]
    r = (k-1)*nx+j
    φB[:,:,r] = uf 
end
φx0 = zeros(p, N, nx)
x0u = zeros(1, N)
@inbounds for (j,k) in Iterators.product(1:nx, 1:1)
    E = zeros(nx)
    x0 = zeros(nx); x0[j] = 1
    fsys = ss(A, E, C, 0, 1)
    uf = lsim(fsys, x0u; x0)[1]
    r = (k-1)*nx+j
    φx0[:,:,r] = uf 
end
if !zeroD
    φD = zeros(p, N, m*p)
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



function phi_1(φB, φx0, pN)
    φ3 = cat(φB, φx0, dims=Val(3))
    φ = reshape(φ3, pN, :)
end


function phi_2(φB, φx0, pN)
    φ = reshape(cat(φB, φx0, dims=Val(3)), pN, :)
end






# φ4 = permutedims(φ3, (1,3,2))
φ = reshape(φ3, p*N, :)
if weights === nothing
    BD = estimator(φ, vec(Y .- ε))
else
    BD = estimator(φ, vec(Y .- ε), weights)
end
B = copy(reshape(BD[1:m*nx], nx, m))
x0 = BD[m*nx .+ (1:nx)]
if zeroD
    D = zeros(T, p, m)
else
    D = reshape(BD[end-p*m+1:end], p, m)
    B .+= K*D
end


@benchmark cat($φB, $φx0, dims=Val(3))


est_1(φ, r) = φ\r
est_2(φ, r) = factorize(φ)\r

face_est(φ, r)
fφ = factorize(φ)

@benchmark estimator(factorize($φ), vec($Y .- $ε))
