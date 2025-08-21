


# function generate_lds_parameters(S, Q_noise, R_noise, P0_noise)::NamedTuple

#     A = randn(S.dat.x_dim, S.dat.x_dim) + 5I;
#     s,u = eigen(A); # get eigenvectors and eigenvals
#     s = s/maximum(abs.(s))*.95; # set largest eigenvalue to lie inside unit circle (enforcing stability)
#     s[real.(s) .< 0] = -s[real.(s) .< 0]; #set real parts to be positive (encouraging smoothness)
#     A_sim = real(u*(Diagonal(s)/u));  # reconstruct A from eigs and eigenvectors

#     # diagonal Q
#     # Q =  Matrix(sqrt(Q_noise) * I(S.dat.x_dim)); 
#     Q = sqrt(Q_noise) .* randn(S.dat.x_dim, S.dat.x_dim); 
#     Q_sim = tol_PD(Q'*Q; tol=Q_noise);

#     B_sim = randn(S.dat.x_dim, S.dat.u_dim);
#     C_sim = randn(S.dat.n_chans, S.dat.x_dim);

#     R =  sqrt(R_noise) .* randn(S.dat.n_chans, S.dat.n_chans); 
#     R_sim = tol_PD(R'*R; tol=R_noise);

#     B0_sim = randn(S.dat.x_dim, S.dat.u0_dim);

#     P = sqrt(P0_noise) .* randn(S.dat.x_dim, S.dat.x_dim);
#     P0_sim = tol_PD(P'*P; tol=P0_noise);
    

#     sim = (A = A_sim, B = B_sim, C = C_sim, Q = Q_sim, R = R_sim, B0 = B0_sim, P0 = P0_sim);
#     return sim

# end





function generate_lds_trials(A, B, Q, C, R, B0, P0, u, u0, n_times, n_trials)
    """ Generate data from a linear dynamical system
    Args:
        A : state transition matrix (dim_x x dim_x)
        B : control matrix (dim_x x dim_u)
        Q : state noise covariance (dim_x x dim_x)
        C : observation matrix (dim_y x dim_x)
        R : observation noise covariance (dim_y x dim_y)
        u : u (n_trials x n_times x dim_u)
        n_times : number of time steps
        n_trials : number of trials
        X0 : initial state (dim_x x 1)
    Returns:
        x : latent states (n_trials x n_times x dim_x)
        y : observations (n_trials x n_times x dim_y)
    """


    # get dimensions
    dim_x = size(A, 1)
    dim_y = size(C, 1)
    dim_u = size(B, 2)

    # make sure if u are not None then they have the right shape
    if u !== nothing
        @assert size(u, 1) == dim_u
        @assert size(u, 2) == n_times
        @assert size(u, 3) == n_trials
    end

    # initialize latent states
    x = zeros(dim_x, n_times, n_trials)
    y = zeros(dim_y, n_times, n_trials)

    for tt in axes(y,3)

        # initialize latent state
        x[:,1,tt] = rand(MvNormal(B0*u0[:,tt], P0))
  
        # generate latent states
        for nn in axes(x,2)[2:end]
            x[:, nn, tt] = A*x[:, nn-1, tt] + B*u[:,nn-1,tt] + rand(MvNormal(zeros(dim_x), Q))
        end

        # generate observations
        for nn in axes(y,2)
            y[:, nn, tt] = C * x[:, nn, tt] + rand(MvNormal(zeros(dim_y), R))
        end

    end

    # return x and y
    return x, y

end



function generate_dlds_trials(A, B, Q, C, R, B0, P0, u, u0, n_times, n_trials::Int64=1)
    """ Generate data from a deterministic linear dynamical system
    Args:
        A : state transition matrix (dim_x x dim_x)
        B : control matrix (dim_x x dim_u)
        Q : state noise covariance (dim_x x dim_x)
        C : observation matrix (dim_y x dim_x)
        R : observation noise covariance (dim_y x dim_y)
        u : u (n_trials x n_times x dim_u)
        n_times : number of time steps
        n_trials : number of trials
        X0 : initial state (dim_x x 1)
    Returns:
        x : latent states (n_trials x n_times x dim_x)
        y : observations (n_trials x n_times x dim_y)
    """

    # seed this 
    # Random.seed!(seed)

    # get dimensions
    dim_x = size(A, 1)
    dim_y = size(C, 1)
    dim_u = size(B, 2)

    # make sure if u are not None then they have the right shape
    if u !== nothing
        @assert size(u, 1) == dim_u
        @assert size(u, 2) == n_times
        @assert size(u, 3) == n_trials
    end

    # initialize latent states
    x = zeros(dim_x, n_times, n_trials)
    y = zeros(dim_y, n_times, n_trials)

    for tt in axes(y,3)

        # initialize latent state
        x[:,1,tt] = B0*u0[:,tt];
        
        # generate latent states
        for nn in axes(x,2)[2:end]
            x[:, nn, tt] = A * x[:, nn-1, tt] + B * u[:, nn-1, tt];
        end

        # generate observations
        for nn in axes(y,2)[1:end]
            y[:, nn, tt] = C * x[:, nn, tt];
        end

    end

# return x and y
    return x, y
end







# function generate_lrnn_trials(A, B, Q, C, R, X0, P0, u, n_times, n_trials::Int64=1)
#     """ Generate data from a linear dynamical system
#     Args:
#         A : state transition matrix (dim_x x dim_x)
#         B : control matrix (dim_x x dim_u)
#         Q : state noise covariance (dim_x x dim_x)
#         C : observation matrix (dim_y x dim_x)
#         R : observation noise covariance (dim_y x dim_y)
#         u : u (n_trials x n_times x dim_u)
#         n_times : number of time steps
#         n_trials : number of trials
#         X0 : initial state (dim_x x 1)
#     Returns:
#         x : latent states (n_trials x n_times x dim_x)
#         y : observations (n_trials x n_times x dim_y)
#     """

#     # seed this 
#     # Random.seed!(seed)

#     # get dimensions
#     dim_x = size(A, 1)
#     dim_y = size(C, 1)
#     dim_u = size(B, 2)

#     # make sure if u are not None then they have the right shape
#     if u !== nothing
#         @assert size(u, 1) == dim_u
#         @assert size(u, 2) == n_times
#         @assert size(u, 3) == n_trials
#     end

#     # initialize latent states
#     x = zeros(dim_x, n_times, n_trials)
#     y = zeros(dim_y, n_times, n_trials)

#     for tt in 1:n_trials

#         # initialize latent state
#         if X0 === nothing
#             x[:,1,tt] = rand(MvNormal(zeros(dim_x), P0))
#         elsesss
#             x[:,1,tt] = rand(MvNormal(X0, P0))
#         end

#         # generate latent states
#         for nn in 2:n_times
#             if u !== nothing
#                 x[:, nn, tt] = A * x[:, nn-1, tt] + B * u[:, nn-1, tt] + rand(MvNormal(zeros(dim_x), Q))
#             else
#                 x[:, nn, tt] = A * x[:, nn-1, tt] + rand(MvNormal(zeros(dim_x), Q))
#             end
#         end

#         # generate observations
#         for nn in 1:n_times
#             y[:, nn, tt] = C * Fx.(x[:, nn, tt]) + rand(MvNormal(zeros(dim_y), R))
#         end

#     end

# # return x and y
#     return x, y
# end
