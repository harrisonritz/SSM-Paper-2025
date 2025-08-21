%% analyze single model

clear; clc;



ROOT = ""


%% MODELS


dataset = 'H19'
x_disp = 112;


switch dataset
    case 'H19'

        fn = '2024-10-31-10h_H19__prepTrial200-prevTask'
        x_list = [16:16:128]; nx = length(x_list);
        pts = 1:30; npt = length(pts);

    case 'A23'

        fn = '2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask'

        x_list = [16:16:128]; nx = length(x_list);
        pts = 1:26; npt = length(pts);

        fprintf("\n PTS: %d\n\n", npt)

end










fld = sprintf('%s/della-outputs/%s/', ROOT, fn);
ppc_fld = sprintf('%s/della-outputs/%s_PPC/', ROOT, fn);
neuroRNN_fld = sprintf('%s/della-outputs/RNN_neuro_orth', ROOT);
neuroRNN_tied_fld = sprintf('%s/della-outputs/RNN-untied_neuro_orth', ROOT);


addpath(genpath(sprintf('%s/data/utils', ROOT)))
addpath(genpath("utils"))
vik = load('vik.mat'); vik = vik.vik;
davos = load('davos.mat'); davos = davos.davos;
batlow = load('batlow.mat'); batlow = batlow.batlow;
lajolla= load('lajolla.mat'); lajolla = lajolla.lajolla;
batlowK = load('batlowK.mat'); batlowK = batlowK.batlowK;
bam = load('bam.mat'); bam = bam.bam;
berlin = load('berlin.mat'); berlin = berlin.berlin;
cork = load('cork.mat'); cork = cork.cork;

center = @(x) x-nanmean(x);
vec = @(x) x(:);
sem = @(x) nanstd(x,[],1)./sqrt(sum(isfinite(x)));
sem_wn = @(x) ((size(x,2)-1)/size(x,2))*nanstd(x-nanmean(x,2),[],1)./sqrt(sum(isfinite(x)));
symt = @(x) 0.5*(x+x');



infmt = "MM/dd/uuuu HH:mm:ss";








%% LOAD PARAMETERS



clear A B C Q_mat Q_chol R_mat R_chol B0 P0_mat P0_chol u u0 y y_orig W mu y_train y_train_orig
for pp = 1:npt

    fprintf('\npt %d', pp)



    % parameters
    fdir = dir(sprintf('%s*Pt%d_xdim%d.mat',fld, pts(pp), x_disp));

    if isempty(fdir)
        fprintf(' - skip')
        continue;
    end

    try
        d = load(fullfile(fdir(1).folder, fdir(1).name), 'mdl', 'dat');
        isempty(d.mdl);
    catch
        fprintf(' - skip')
        continue

    end


    A{pp} = d.mdl.A;
    B{pp} = d.mdl.B;
    C{pp} = d.mdl.C;

    Q_mat{pp} = d.mdl.Q.mat;
    Q_chol{pp} = chol(Q_mat{pp});

    R_mat{pp} = d.mdl.R.mat;
    R_chol{pp} = chol(R_mat{pp});


    B0{pp} = d.mdl.B0;

    P0_mat{pp} = d.mdl.P0.mat;
    P0_chol{pp} = chol(P0_mat{pp});

    u{pp} = d.dat.u_test;
    u0{pp} = d.dat.u0_test;
    y{pp} = d.dat.y_test;
    y_orig{pp} = d.dat.y_test_orig;

    y_train{pp} = d.dat.y_train;
    y_train_orig{pp} = d.dat.y_train_orig;

    W{pp} = d.dat.W;
    mu{pp} = d.dat.mu;



end



%% ESTIMATE SPECTRUM & PLOT

pt_disp = 3;

n_sims = 100;
x_z = zeros(x_disp,1);
x_sigma = eye(x_disp);
fmax = 30;

do_detrend = false;
do_orig = false;

%
%
% [spectra,freqs] =spectopo(d.y, size(d.y,2), 1/d.dt,...
%     'plot', 'off', 'verbose', 'off');


for pp = pt_disp

    fprintf('\npt %d', pp)
    if do_orig
        [n_y,n_times,n_trials] = size(y_orig{pp});
    else
        [n_y,n_times,n_trials] = size(y{pp});
    end

    A_p = A{pp};
    B_p = B{pp};
    C_p = C{pp};

    Q_mat_p = Q_mat{pp};
    Q_chol_p = Q_chol{pp};

    R_mat_p = R_mat{pp};
    R_chol_p = R_chol{pp};

    B0_p = B0{pp};
    P0_mat_p = P0_mat{pp};
    P0_chol_p = P0_chol{pp};

    u0_p = u0{pp};

    y_s_all = nan(n_y,n_times,n_trials,n_sims);
    for ss = 1:n_sims

        % run sim
        y_s = nan(n_y,n_times,n_trials);
        for tl = 1:n_trials
            if do_orig==1
                y_s(:,:,tl) = W{pp}*C_p*[B0_p*u0_p(:,tl), B_p*u{pp}(:,1:(end-1),tl)];
            else
                y_s(:,:,tl) = C_p*[B0_p*u0_p(:,tl), B_p*u{pp}(:,1:(end-1),tl)];
            end
        end

        zx = zeros(x_disp,1);

        parfor tl = 1:n_trials

            xx = nan(x_disp,1);
            for tt = 1:n_times

                if tt ==1
                    xx = mvnrnd(zx, P0_mat_p, [], P0_chol_p)';
                else
                    xx = mvnrnd(A_p*xx, Q_mat_p, [], Q_chol_p)';
                end
                if do_orig ==1
                    y_s(:,tt,tl) = mvnrnd(y_s(:,tt,tl) + mu{pp} + W{pp}*C_p*xx,  W{pp}*R_mat_p*W{pp}', [], W{pp}*R_chol_p*W{pp}')';
                else
                    y_s(:,tt,tl) = mvnrnd(y_s(:,tt,tl) + C_p*xx, R_mat_p, [], R_chol_p)';
                end
            end

        end

        y_s_all(:,:,:,ss) = y_s;

    end


    % true spectra
    fprintf(' - true spectra')
    % y_pl = reshape(y{pp}, [n_y, n_times*n_trials]);
    if do_orig==1
        [spectra_p,freqs_p] = spectopo(y_orig{pp}, size(y_orig{pp},2), 1/d.dat.dt,...
            'plot', 'off', 'verbose', 'off');
    else
        [spectra_p,freqs_p] = spectopo(y{pp}, size(y{pp},2), 1/d.dat.dt,...
            'plot', 'off', 'verbose', 'off');
    end


    % sim spectra
    fprintf(' - sim spectra')
    % y_sl = reshape(y_s, [n_y, n_times*n_trials]);
    y_s = reshape(y_s_all, [n_y, n_times, n_trials*n_sims]);
    [spectra_s,freqs_s] = spectopo(y_s, size(y_s,2), 1/d.dat.dt,...
        'plot', 'off', 'verbose', 'off');



    % plot
    f_sel = freqs_p<=fmax;
    assert(all(freqs_p==freqs_s))

     if do_detrend
        pred = [ones(sum(f_sel),1), zscore(vec(1:sum(f_sel)))];
        spectra_p(:,f_sel) = spectra_p(:,f_sel) - spectra_p(:,f_sel)/pred'*pred';
        spectra_s(:,f_sel) = spectra_s(:,f_sel) - spectra_s(:,f_sel)/pred'*pred';
    end

    col1 = batlowK;
    col2 = batlow;
    cidx = round(linspace(1,216, size(y_s,1)));

    y_std = max(std(spectra_p))*.5;
    figure; hold on;
    for yy = 1:size(y_s,1)
        if do_orig==1
            plot(freqs_p(f_sel), y_std*yy+(spectra_p(yy,f_sel)),...
                '-', 'LineWidth', 3, 'color', col1(cidx(yy),:) );

            plot(freqs_s(f_sel), y_std*yy+(spectra_s(yy,f_sel)),...
                ':', 'LineWidth', 3, 'color', col2(cidx(yy),:))
        else
            plot(freqs_p(f_sel), (spectra_p(yy,f_sel)),...
                '-', 'LineWidth', 3, 'color', col1(cidx(yy),:) );

            plot(freqs_s(f_sel), (spectra_s(yy,f_sel)),...
                ':', 'LineWidth', 3, 'color', col2(cidx(yy),:))
        end
    end

    set(gca, 'TickDir', 'out', 'LineWidth', 1)
    title(pp)


end



%% ANALYSIS


detrend_method = 'none' % channel agg, none




cos_sim = @(x,y) normalize(x,1,'norm')'*normalize(y,1,'norm');


n_sims = 10;
x_z = zeros(x_disp,1);
x_sigma = eye(x_disp);
fmax = 30;
do_orig = false;


% [spectra,freqs] =spectopo(d.y, size(d.y,2), 1/d.dt,...
%     'plot', 'off', 'verbose', 'off');

[R2_f,cos_f,R2_train] = deal(nan(npt,1));
parfor pp = 1:npt

    fprintf('\npt %d', pp)
    if do_orig
        [n_y,n_times,n_trials] = size(y_orig{pp});
    else
        [n_y,n_times,n_trials] = size(y{pp});
    end

    A_p = A{pp};
    B_p = B{pp};
    C_p = C{pp};

    Q_mat_p = Q_mat{pp};
    Q_chol_p = Q_chol{pp};

    R_mat_p = R_mat{pp};
    R_chol_p = R_chol{pp};

    B0_p = B0{pp};
    P0_mat_p = P0_mat{pp};
    P0_chol_p = P0_chol{pp};

    u0_p = u0{pp};

    W_p = W{pp};
    mu_p = mu{pp};

    y_s_all = nan(n_y,n_times,n_trials,n_sims);
    for ss = 1:n_sims

        % run sim
        y_s = nan(n_y,n_times,n_trials);
        for tl = 1:n_trials
            if do_orig==1
                y_s(:,:,tl) = W_p*C_p*[B0_p*u0_p(:,tl), B_p*u{pp}(:,1:(end-1),tl)];
            else
                y_s(:,:,tl) = C_p*[B0_p*u0_p(:,tl), B_p*u{pp}(:,1:(end-1),tl)];
            end
        end

        zx = zeros(x_disp,1);

        for tl = 1:n_trials

            xx = nan(x_disp,1);
            for tt = 1:n_times

                if tt ==1
                    xx = mvnrnd(zx, P0_mat_p, [], P0_chol_p)';
                else
                    xx = mvnrnd(A_p*xx, Q_mat_p, [], Q_chol_p)';
                end
                if do_orig ==1
                    y_s(:,tt,tl) = mvnrnd(y_s(:,tt,tl) + mu_p + W_p*C_p*xx,  W_p*R_mat_p*W_p', [], W_p*R_chol_p*W_p')';
                else
                    y_s(:,tt,tl) = mvnrnd(y_s(:,tt,tl) + C_p*xx, R_mat_p, [], R_chol_p)';
                end

            end

        end

        y_s_all(:,:,:,ss) = y_s;

    end


    % true spectra
    fprintf(' - true spectra')
    % y_pl = reshape(y{pp}, [n_y, n_times*n_trials]);
    if do_orig==1
        [spectra_p,freqs_p] = spectopo(y_orig{pp}, size(y_orig{pp},2), 1/d.dat.dt,...
            'plot', 'off', 'verbose', 'off');
    else
        [spectra_p,freqs_p] = spectopo(y{pp}, size(y{pp},2), 1/d.dat.dt,...
            'plot', 'off', 'verbose', 'off');
    end

    % train spectra
    fprintf(' - train spectra')
    if do_orig==1
        [spectra_p_train,freqs_p_train] = spectopo(y_train_orig{pp}, size(y_orig{pp},2), 1/d.dat.dt,...
            'plot', 'off', 'verbose', 'off');
    else
        [spectra_p_train,freqs_p_train] = spectopo(y_train{pp}, size(y{pp},2), 1/d.dat.dt,...
            'plot', 'off', 'verbose', 'off');
    end



    % sim spectra
    fprintf(' - sim spectra')
    % y_sl = reshape(y_s, [n_y, n_times*n_trials]);
    y_s = reshape(y_s_all, [n_y, n_times, n_trials*n_sims]);
    [spectra_s,freqs_s] = spectopo(y_s, size(y_s,2), 1/d.dat.dt,...
        'plot', 'off', 'verbose', 'off');


    % detrend
    assert(all(freqs_p==freqs_s))
    f_sel = freqs_p<=fmax;

    switch detrend_method


        case 'agg'

            pred = [ones(sum(f_sel)*n_y,1), repmat(zscore(vec(log(1+freqs_s(f_sel)))),[n_y,1])];
            spectra_pl = vec(spectra_p(:,f_sel)) - pred*(pred\vec(spectra_p(:,f_sel)));
            spectra_sl = vec(spectra_s(:,f_sel)) - pred*(pred\vec(spectra_s(:,f_sel)));

            cos_f(pp) = cos_sim(spectra_pl, spectra_sl);
            R2_f(pp) = 1 - (meansqr(spectra_pl - spectra_sl)/var(spectra_pl));


        case 'channel' % linearly remove 

            pred = [ones(sum(f_sel),1), zscore(vec(1:sum(f_sel)))];
            spectra_pl = vec(spectra_p(:,f_sel) - spectra_p(:,f_sel)/pred'*pred');
            spectra_sl = vec(spectra_s(:,f_sel) - spectra_s(:,f_sel)/pred'*pred');
            spectra_pl_train = vec(spectra_p_train(:,f_sel) - spectra_p_train(:,f_sel)/pred'*pred');


            cos_f(pp) = cos_sim(spectra_pl, spectra_sl);
            R2_f(pp) = 1 - (meansqr(spectra_pl - spectra_sl)/var(spectra_pl));
            R2_train(pp) = 1 - (meansqr(spectra_pl - spectra_pl_train)/var(spectra_pl));


        case 'none'

            cos_f(pp) = cos_sim(vec(spectra_p(:,f_sel)'), vec(spectra_s(:,f_sel)'));
            R2_f(pp) = 1 - (meansqr(vec(spectra_p(:,f_sel) - spectra_s(:,f_sel)))/var(vec(spectra_p(:,f_sel))));
            R2_train(pp) = 1 - (meansqr(vec(spectra_p(:,f_sel) - spectra_p_train(:,f_sel)))/var(vec(spectra_p(:,f_sel))));

        otherwise
            error('need detrend method')

    end


end


% bootstrap_rows(cos_f, @median,1000)
sim_R2 = bootstrap_rows(R2_f, @median,1000) % recovered
rel_R2 = bootstrap_rows(R2_train, @median,1000) % noise ceiling
simNormed_R2 = bootstrap_rows(R2_f./R2_train, @median,1000) % recovered / noise-ceiling

figure;
histogram(R2_f./R2_train,30)



