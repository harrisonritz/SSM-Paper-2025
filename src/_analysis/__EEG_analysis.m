%% analyze single model

clear; clc;
    

ROOT = '/Users/hr0283/Brown Dropbox/Harrison Ritz/HallM_NeSS'
addpath(sprintf('%s/src/analysis/utils/matplotlib', ROOT))


%% MODELS


run_model = 'A23'

report_n = 0

switch run_model

    case 'H19'

        fn = '2024-10-31-10h_H19__prepTrial200-prevTask' %=============== BEST H19
      

        x_list = [16:16:128]; nx = length(x_list);
        pts = 1:30; npt = length(pts);


    case 'A23'
        fn = '2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask'

        x_list = [16:16:128]; nx = length(x_list);
        pts = 1:26; npt = length(pts);

        fprintf("\n PTS: %d\n\n", npt)


end










fld = sprintf('%s/della-outputs/%s/',ROOT, fn);
ppc_fld = sprintf('%s/della-outputs/%s_PPC/',ROOT, fn);


mdl_short = strsplit(run_model, '_');
mdl_short = mdl_short{1}
neuroRNN_sweep_fld = sprintf('%s/della-outputs/_RNN/RNN_%s_neuro-sweep-5k', ROOT, mdl_short);
neuroRNN_untied_fld = sprintf('%s/della-outputs/_RNN/RNN-untied_%s_neuro-sweep-5k', ROOT, mdl_short);



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






% load --------------------------------








time_on = tic;


% get parameters
for pp = 1:npt
    fdir = dir(fullfile(fld, sprintf('*Pt%d_xdim%d.mat',pts(pp), x_list(1))));
    try
        d = load(fullfile(fdir.folder, fdir.name), 'prm', 'dat');
        prm = d.prm;
        dat = d.dat;

        fprintf('preproc file: %s\n', prm.filename);

    catch
        continue
    end
    disp(pts(pp))
    break;
end






tot_ll = nan(npt, nx, prm.max_iter_em);
[test_ll, test_R2] = deal(nan(npt, nx, prm.max_iter_em/prm.test_iter + 1));
[test_R2_end, test_R2_white, test_ll_end,...
    fit_dur,...
    null_mse, test_mse, ssid_R2, null_ll_white, null_mse_white, test_mse_white,...
    neuroRNN_sweep_loss, neuroRNN_untied_loss] = deal(nan(npt,nx));
% [y_dim,neuroRNN_loss,neuroRNN_untied_loss] = deal(nan(npt,1));

model_size = 0;
clear fwd n_train n_test
for pp = 1:npt

    fprintf('\npt-%d',pp)


    for dd = 1:nx

        fdir = dir(sprintf('%s*Pt%d_xdim%d.mat',fld, pts(pp), x_list(dd)));


        if isempty(fdir)
            continue;
        end

        try
            if report_n==1
                d = load(fullfile(fdir(1).folder, fdir(1).name), 'res','dat');
                n_train(pp) = d.dat.n_train;
                n_test(pp) = d.dat.n_test;
            else
                d = load(fullfile(fdir(1).folder, fdir(1).name), 'res');
            end
            isempty(d.res);
        catch
            continue
            
        end

        if x_list(dd) == 112 && pp == 2

            x_dim = size(d.res.mdl_em.B,1);
            y_dim = size(d.res.mdl_em.C,1);

            model_size = ...
                numel(d.res.mdl_em.A) + ...
                numel(d.res.mdl_em.B) + ...
                .5*x_dim*(x_dim-1) + ...
                numel(d.res.mdl_em.C) + ...
                .5*y_dim*(y_dim-1) + ...
                numel(d.res.mdl_em.B0) + ...
                .5*x_dim*(x_dim-1);

        end

        % D(pp,dd) = d.S;
        tot_ll(pp,dd, 1:length(d.res.total_loglik)) = d.res.total_loglik;
        test_ll(pp,dd, 1:length(d.res.test_loglik)) = d.res.test_loglik;
        test_R2(pp,dd, 1:length(d.res.test_R2_orig)) = d.res.test_R2_orig;
        try
            ssid_R2(pp,dd) = 1 - (d.res.preRefine_test_loglik/d.res.null_loglik(4));
        catch
            ssid_R2(pp,dd) = 1 - (d.res.ssid_test_loglik/d.res.null_loglik(4));
        end
        test_R2_end(pp,dd) = d.res.test_R2_orig(end);
        test_R2_white(pp,dd) = d.res.test_R2_white(end);
        null_ll_white(pp,dd) = d.res.null_loglik(4);

        try
            test_ll_end(pp,dd) = d.res.postEM_test_loglik;
        catch
            test_ll_end(pp,dd) = d.res.em_test_loglik;
        end

        y_dim(pp) = size(d.res.mdl_em.R.mat,1);

        fit_dur(pp,dd) = hours(datetime(d.res.endTime_em, 'InputFormat',infmt) - datetime(d.res.startTime_all, 'InputFormat',infmt));

        fwd(pp,dd,:) = d.res.fwd_R2_white;


        try
           
            neuroRNN_fn = dir(fullfile(neuroRNN_sweep_fld, sprintf('*_pt%d_xdim%d.mat', pts(pp), x_list(dd))));
            nrnn = load(fullfile(neuroRNN_fn.folder, neuroRNN_fn.name), 'min_test_loss');
            neuroRNN_sweep_loss(pp,dd) = nrnn.min_test_loss;

            if isnan(nrnn.min_test_loss)
                keyboard
            end

            neuroRNN_untied_fn = dir(fullfile(neuroRNN_untied_fld, sprintf('*_pt%d_xdim%d.mat', pts(pp), x_list(dd)) ));
            nrnn = load(fullfile(neuroRNN_untied_fn.folder, neuroRNN_untied_fn.name), 'min_test_loss');
            neuroRNN_untied_loss(pp,dd) = nrnn.min_test_loss;
            if isnan(nrnn.min_test_loss)
                keyboard
            end

        catch
        end


    end
end
fprintf('\nloaded')
fprintf('\nmodel size: %d\n', model_size)


time_off = toc(time_on)


if report_n==1
    report_n_train = [min(n_train), max(n_train),  mean(double(n_train)), std(double(n_train))]
    report_n_test = [min(n_test), max(n_test),  mean(double(n_test)), std(double(n_test))]
end



%% fit and duration

% close all;
figure;
set(gcf, 'Position', [50,200,1200,1200])


% R2
nexttile;
colormap(hot)
imagesc(test_R2_white', [-inf,1]); colorbar;
yticks(1:nx)
yticklabels(x_list)
xticks(1:npt)
title('R2')


nexttile; hold on;


vec_xlim = vec(repmat(x_list, [npt,1]));
vec_ll = vec(test_R2_white);
swarmchart(vec_xlim, vec_ll, 50, 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w')


% plot(x_list, test_R2_white', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list, nanmean(test_R2_white)', '-k', 'LineWidth', 3)
plot(x_list, nanmean(test_R2_white)', '-ok', 'LineWidth', 1, 'MarkerFaceColor', 'k', 'MarkerSize', 8)
ylim([0,1])
title('test R2')
xlim([min(x_list)-5, max(x_list)+5])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
xticks(x_list)


c_test_R2_white = test_R2_white - nanmean(test_R2_white,2);
nexttile; hold on;
plot(x_list, c_test_R2_white', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list, nanmean(c_test_R2_white)', '-k', 'LineWidth', 3)
plot(x_list, nanmean(c_test_R2_white)', '-ok', 'LineWidth', 1, 'MarkerFaceColor', 'k', 'MarkerSize', 8)
% ylim([-.2,inf])
xlim([min(x_list)-5, max(x_list)+5])
title('test R2 (pt-centered)')
xlim([min(x_list)-5, max(x_list)+5])
set(gca, 'TickDir', 'out', 'LineWidth', 1)


% forward prediction
nexttile; hold on;
fwd(fwd==0) = nan;
% plot((1:25)*dat.dt*1000, squeeze((fwd(:,7,1:25)))', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)

vec_dt = repmat(vec((1:25)*dat.dt*1000), [npt,1]);
vec_fwd = vec(squeeze((fwd(:,7,1:25)))');
swarmchart(vec_dt, vec_fwd, 30, 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w')

median_horizon = median(squeeze(fwd(:,end-1,:)))

plot((1:25)*dat.dt*1000, squeeze(nanmean(fwd(:,7,1:25))), '-ok', 'LineWidth', 3, 'MarkerFaceColor','k')
ylim([0,1])
title('test R2 (lookahead)')
set(gca, 'TickDir', 'out', 'LineWidth', 1)
xlabel('horizon (ms)')
ylabel('R^2 (cov)')
xlim([0,26*dat.dt*1000])


% ====== loglik
nexttile;
colormap(hot)
imagesc(test_ll_end'); colorbar;
yticks(1:nx)
yticklabels(x_list)
xticks(1:npt)
title('test ll')


% test loglik
nexttile; hold on;

plot(x_list, test_ll_end', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list, nanmean(test_ll_end)', '-k', 'LineWidth', 3)
plot(x_list, nanmean(test_ll_end)', '-ok', 'LineWidth', 1, 'MarkerFaceColor', 'k', 'MarkerSize', 8)
% ylim([0,1])
xlim([min(x_list)-5, max(x_list)+5])
title('test ll')
xlim([min(x_list)-5, max(x_list)+5])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
set(gca, 'YScale', 'log')

[~,sort_pt] = sort(test_ll_end(:,end-1))



% pt-centered test loglik
c_test_ll_end = test_ll_end - nanmean(test_ll_end,2);

nexttile; hold on;

vec_xlim = vec(repmat(x_list, [npt,1]));
vec_ll = vec(c_test_ll_end);
swarmchart(vec_xlim, vec_ll, 50, 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w')

% plot(x_list, c_test_ll_end', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list, nanmean(c_test_ll_end)', '-k', 'LineWidth', 3)
plot(x_list, nanmean(c_test_ll_end)', '-ok', 'LineWidth', 1, 'MarkerFaceColor', 'k', 'MarkerSize', 8)
% ylim([-.2,inf])
xlim([min(x_list)-5, max(x_list)+5])
title('test ll (pt-centered)')
xlim([min(x_list)-5, max(x_list)+5])
xticks(x_list)
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
xline(median(y_dim))
fprintf('\n median y dim: %.2g\n', median(y_dim))
fprintf('\n y dim range: %.2g - %.2g\n', min(y_dim), max(y_dim))

set(gcf, 'Renderer', 'painters')


% model sel

try
    spm_sel = all(isfinite(test_ll_end),1);
    [alpha,exp_r,xp,pxp,bor] = spm_BMS(test_ll_end(:,spm_sel));
    nexttile; hold on;
    bar(pxp)
    % bar(exp_r)
    xticks(1:sum(spm_sel))
    xticklabels(x_list(spm_sel))
    title('BMS - PXP')
    ylim([0,1])
    set(gca, 'TickDir', 'out', 'LineWidth', 1)

catch
end




% number of PCA dims
nexttile; hold on;
histogram(y_dim,14:2:28);
title('y dim')
set(gca, 'TickDir', 'out', 'LineWidth', 1)






% duration

nexttile; hold on;

% hours
plot(x_list+1, fit_dur', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list+1, nanmedian(fit_dur)', '-k', 'LineWidth', 3)
plot(x_list+1, nanmedian(fit_dur)', '-ok', 'LineWidth', 1, 'MarkerFaceColor', 'k', 'MarkerSize', 8)
yline(24, '-k', 'LineWidth', 2)
ylabel('hours')


xlim([min(x_list)-5, max(x_list)+5])
title('fit duration')
xlim([min(x_list)-5, max(x_list)+5])
xticks(x_list)
set(gca, 'TickDir', 'out', 'LineWidth', 1)


dim = ones(size(fit_dur)).*x_list;

nexttile; hold on;
plot(dim, (fit_dur), 'ok')
B = [ones(size(vec(dim))), zscore(vec(dim)), zscore(vec(dim.^2))]\vec(fit_dur)
plot(vec(dim), ([ones(size(vec(dim))), zscore(vec(dim)), zscore(vec(dim.^2))]*B), '-k');
B = [ones(size(vec(dim))), zscore(vec(dim))]\vec(fit_dur)
plot(vec(dim), ([ones(size(vec(dim))), zscore(vec(dim))]*B), '--k');

fitlm([zscore(vec(dim)), zscore(vec(dim.^2))], vec(fit_dur))
mdl = fitglm([zscore(vec(dim)), zscore(vec(dim.^2))], vec(fit_dur), 'Distribution', 'poisson')

plot(vec(dim), mdl.predict, '-b');


% iter
nexttile; hold on;
niter = (sum(isfinite(tot_ll),3)); niter(niter==0) = nan;

plot(x_list-1, niter', 'ob', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list-1, nanmedian(niter)', '-b', 'LineWidth', 3)
plot(x_list-1, nanmedian(niter)', '-ob', 'LineWidth', 1, 'MarkerFaceColor', 'b', 'MarkerSize', 8)
yline((double(prm.max_iter_em)), '-b', 'LineWidth', 2)
ylabel('iterations')
xlim([min(x_list)-5, max(x_list)+5])
title('iterations')
xlim([min(x_list)-5, max(x_list)+5])
xticks(x_list)
set(gca, 'TickDir', 'out', 'LineWidth', 1)




% niter
%
% nexttile;
% colormap(hot)
% imagesc(niter); colorbar;
% title('n iterations')
% yticks(1:nx)
% yticklabels(x_list)
% xticks(1:npt)





R2_prct = [x_list; prctile(test_R2_end, [2.5, 50, 97.5])]




% basis set
n=nexttile; hold on;
% cols = colormap(n,abyss);
colororder(n, tab10(dat.n_bases))

col_idx = round(linspace(1, 256, dat.n_bases));

plot(vecnorm(dat.u_train(1:dat.n_bases,:,1)), '--k')
for cc = 1:dat.n_bases
    % plot(dat.u_train(cc,:,1)', '-', 'color', cols(col_idx(cc),:))
    plot(dat.u_train(cc,:,1)', '-')
end
try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
title('basis set')
set(gca, 'TickDir', 'out', 'LineWidth', 1)
xlabel('time')
ylabel('u')


% trial predictors
n=nexttile; hold on;
colormap(n,'prism')
preds = unique(dat.pred_name);
for pp = 1:length(preds)
    pred_sel =find(ismember(dat.pred_name, preds(pp)),1);
    plot(3*pp+squeeze(dat.u_train(pred_sel,1,:)))
end

% title('basis set')
set(gca, 'TickDir', 'out', 'LineWidth', 1)
xlabel('time')
ylabel('B \times u_t')
yline(0)



% estimated inputs
n_gap = 3;
n=nexttile; hold on;
preds = unique(dat.pred_name);
colororder(n, tab20b(4*length(preds)))
for pp = 1:length(preds)
    pred_sel = ismember(dat.pred_name, preds(pp));
    plot(n_gap*pp + (dat.u_train(pred_sel,:,10)'*d.res.mdl_em.B(1,pred_sel)')./std(d.res.mdl_em.B(:)))
    plot(n_gap*pp + (dat.u_train(pred_sel,:,10)'*d.res.mdl_em.B(2,pred_sel)')./std(d.res.mdl_em.B(:)))
    plot(n_gap*pp + (dat.u_train(pred_sel,:,10)'*d.res.mdl_em.B(3,pred_sel)')./std(d.res.mdl_em.B(:)))
    plot(n_gap*pp + (dat.u_train(pred_sel,:,10)'*d.res.mdl_em.B(4,pred_sel)')./std(d.res.mdl_em.B(:)))
end

try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
% title('basis set')
set(gca, 'TickDir', 'out', 'LineWidth', 1)
xlabel('time')
ylabel('B \times u_t')
yline(0)

yticks(n_gap:n_gap:n_gap*length(preds))
yticklabels(preds)







%% plot AR + RNN (sweep)


bootstrap_rnn = false


tic
[null_mse, test_mse,ssid_R2, null_mse_white, test_mse_white] = deal(nan(npt,nx));
for pp = 1:npt

    parfor dd = 1:nx

        fdir = dir(sprintf('%s*Pt%d_xdim%d.mat', fld, pts(pp), x_list(dd)));


        if isempty(fdir)
            continue;
        end

        try
            d = load(fullfile(fdir(1).folder, fdir(1).name), 'res', 'dat');
            isempty(d.res);
        catch
            continue

        end

        null_mse(pp,dd) = d.res.null_mse_orig(end);
        null_mse_white(pp,dd) = d.res.null_mse_white(end);

        test_mse(pp,dd) = d.res.test_sse_orig(end) ./ numel(d.dat.y_test_orig);
        test_mse_white(pp,dd) = d.res.test_sse_white(end) ./ numel(d.dat.y_test);

    end

end
toc




denom = null_mse_white;
rnn_untied_R2 = 1 - (neuroRNN_untied_loss./denom);
rnn_sweep_R2 = 1 - (neuroRNN_sweep_loss./denom);
pt_R2  = 1 - (test_mse_white./denom);

boot_pt = bootstrap_rows(pt_R2, @mean, 10000)

if bootstrap_rnn

    boot_rnn_untied = bootstrap_rows(rnn_untied_R2, @nanmean, 10000)
    boot_rnn_sweep = bootstrap_rows(rnn_sweep_R2, @nanmean, 10000)

else
    boot_rnn_untied = prctile(rnn_untied_R2, [2.5, 50, 97.5])
    boot_rnn_sweep = prctile(rnn_sweep_R2, [2.5, 50, 97.5])

end



% main plot

figure;
nexttile; hold on;

vec_xlim = vec(repmat(x_list, [npt,1]));
vec_ll = vec(test_R2_white);
swarmchart(vec_xlim, vec_ll, 50, 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w')

% plot(x_list, test_R2_white', 'ok', 'LineWidth', 1, 'MarkerFaceColor', 'w', 'MarkerSize', 5)
plot(x_list, nanmean(test_R2_white)', '-k', 'LineWidth', 3)
plot(x_list, nanmean(test_R2_white)', '-ok', 'LineWidth', 1, 'MarkerFaceColor', 'k', 'MarkerSize', 8)

yline(0, '-k', 'LineWidth', 1.5)
title('test R2 (lik)')
ylim([-.1,1])
yticks(-.1:.1:1)
xlim([min(x_list)-5, max(x_list)+5])
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
xticks(x_list)
set(gcf, 'Renderer', 'painters')


errorarea(x_list, boot_rnn_sweep(2,:), ...
    abs(boot_rnn_sweep(1,:) - boot_rnn_sweep(2,:)),...
    abs(boot_rnn_sweep(3,:) - boot_rnn_sweep(2,:)),...
    '-b', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 5)


% errorarea(x_list, boot_rnn_untied(2,:), ...
%     abs(boot_rnn_untied(1,:) - boot_rnn_untied(2,:)),...
%     abs(boot_rnn_untied(3,:) - boot_rnn_untied(2,:)),...
%     '-og', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'MarkerSize', 5)

set(gcf, 'Renderer', 'painters')



% plot relative fit

figure
nexttile; hold on;
plot(rnn_sweep_R2(:,end-1),pt_R2(:,end-1), 'ok', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1)
plot(rnn_sweep_R2(:,end-1),test_R2_end(:,end-1), 'ob', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1)

plot([0,1], [0,1], '-k', 'LineWidth', 1)
set(gca, 'TickDir', 'out', 'LineWidth', 1)

set(gcf, 'Renderer', 'painters')
xlabel('RNN')
ylabel('SSA')

set(gcf, 'Renderer', 'painters')




%% BOTH experiments R2 



bootstrap_rnn = true



all_pts = {1:30, 1:26}
all_fn = {'2024-10-31-10h_H19__prepTrial200-prevTask', '2024-11-24-11h_A23__prepTrial200_ica-p5_p1-30-firws_prevTask'}
all_mdl = {'H19', 'A23'}
cols = {[102,45,145]/255, [246,139,31]/255}
x_off = [-1,1]
tic

figure;
nexttile; hold on;

for mm = 1:length(all_pts)

    pts = all_pts{mm}
    npt = length(pts)
    fld = sprintf('%s/della-outputs/%s/',ROOT, all_fn{mm})
    run_model = all_mdl{mm}
    
    % RNN fld
    mdl_short = strsplit(run_model, '_');
    mdl_short = mdl_short{1};
    neuroRNN_sweep_fld = sprintf('%s/della-outputs/_RNN/RNN_%s_neuro-sweep-5k',ROOT, mdl_short);
    neuroRNN_untied_fld = sprintf('%s/della-outputs/_RNN/RNN-untied_%s_neuro-sweep-5k',ROOT, mdl_short);


    [test_R2_white,null_mse, test_mse,ssid_R2, null_mse_white, test_mse_white, neuroRNN_untied_loss, neuroRNN_sweep_loss] = deal(nan(npt,nx));
    for pp = 1:npt

        parfor dd = 1:nx

            fdir = dir(sprintf('%s*Pt%d_xdim%d.mat', fld, pts(pp), x_list(dd)));


            if isempty(fdir)
                continue;
            end

            try
                d = load(fullfile(fdir(1).folder, fdir(1).name), 'res', 'dat');
                isempty(d.res);
            catch
                continue

            end

            % get SSM fits
            null_mse(pp,dd) = d.res.null_mse_orig(end);
            null_mse_white(pp,dd) = d.res.null_mse_white(end);

            test_mse(pp,dd) = d.res.test_sse_orig(end) ./ numel(d.dat.y_test_orig);
            test_mse_white(pp,dd) = d.res.test_sse_white(end) ./ numel(d.dat.y_test);

            test_R2_white(pp,dd) = d.res.test_R2_white(end);


            % get RNN fits
            neuroRNN_fn = dir(fullfile(neuroRNN_sweep_fld, sprintf('*_pt%d_xdim%d.mat', pts(pp), x_list(dd))));
            nrnn = load(fullfile(neuroRNN_fn.folder, neuroRNN_fn.name), 'min_test_loss');
            neuroRNN_sweep_loss(pp,dd) = nrnn.min_test_loss;

            if isnan(nrnn.min_test_loss)
                keyboard
            end

            neuroRNN_untied_fn = dir(fullfile(neuroRNN_untied_fld, sprintf('*_pt%d_xdim%d.mat', pts(pp), x_list(dd)) ));
            nrnn = load(fullfile(neuroRNN_untied_fn.folder, neuroRNN_untied_fn.name), 'min_test_loss');
            neuroRNN_untied_loss(pp,dd) = nrnn.min_test_loss;
            if isnan(nrnn.min_test_loss)
                keyboard
            end


        end

    end
    toc




    denom = null_mse_white;
    rnn_untied_R2 = 1 - (neuroRNN_untied_loss./denom);
    rnn_sweep_R2 = 1 - (neuroRNN_sweep_loss./denom);
    pt_R2  = 1 - (test_mse_white./denom);

    % boot_pt = bootstrap_rows(pt_R2, @mean, 10000)

    if bootstrap_rnn

        boot_rnn_untied = bootstrap_rows(rnn_untied_R2, @nanmean, 10000)
        boot_rnn_sweep = bootstrap_rows(rnn_sweep_R2, @nanmean, 10000)

    else
        boot_rnn_untied = prctile(rnn_untied_R2, [2.5, 50, 97.5])
        boot_rnn_sweep = prctile(rnn_sweep_R2, [2.5, 50, 97.5])

    end



    % main plot
    vec_xlim = vec(repmat(x_list, [npt,1]));
    vec_ll = vec(test_R2_white);
    swarmchart(vec_xlim + x_off(mm), vec_ll, 50, cols{mm}, 'o', 'LineWidth', 1, 'MarkerFaceColor', 'w')

    plot(x_list + x_off(mm), nanmean(test_R2_white)', '-', 'color', cols{mm}, 'LineWidth', 3)
    plot(x_list + x_off(mm), nanmean(test_R2_white)', '-o', 'color', cols{mm}, 'LineWidth', 1, 'MarkerFaceColor', cols{mm}, 'MarkerSize', 8)

    yline(0, '-k', 'LineWidth', 1.5)
    title('test R2 (lik)')
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
    xticks(x_list)
    set(gcf, 'Renderer', 'painters')


    errorarea(x_list + x_off(mm), boot_rnn_sweep(2,:), ...
        abs(boot_rnn_sweep(1,:) - boot_rnn_sweep(2,:)),...
        abs(boot_rnn_sweep(3,:) - boot_rnn_sweep(2,:)),...
        '-', 'color', cols{mm}, 'LineWidth', 2,  'MarkerSize', 5)

    % errorarea(x_list, boot_rnn_untied(2,:), ...
    %     abs(boot_rnn_untied(1,:) - boot_rnn_untied(2,:)),...
    %     abs(boot_rnn_untied(3,:) - boot_rnn_untied(2,:)),...
    %     '-og', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'MarkerSize', 5)


end

ylim([-.1, 1.0])
yticks([-.1, 1.0])
set(gcf, 'Renderer', 'painters')








%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYZE SWITCH vs REPEAT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('\n\n -------- %s -------- \n\n',run_model)


norm_mode = 'full' % full recur input cue
norm_method = 'zscore'
x_disp = 112
use_Bu = 1




B0_scale = 1;% 0.1729 match A23 to H19
task_prevTask = 0
norm_basis = 0
use_switch = 0 & any(ismember(dat.pred_name, 'task@prevTask'))
ITI_model = contains(run_model, 'ITI')
A_mode = 'full' % full, real, nnrm
fprintf('\n---------\nswitch? %d \nITI? %d\n---------', use_switch, ITI_model)


cos_sim = @(x,y) normalize(x,1,'norm')'*normalize(y,1,'norm');
nan_vecnorm = @(x) sqrt(nansum(x.^2));
real_mean = @(x) real(nanmean(x));
imag_mean = @(x) imag(nanmean(x));
conj_mean = @(x) conj(nanmean(x));
mxnorm = @(x) x./norm(x,'fro');
mxzscore = @(x) (x-nanmean(x(:)))/nanstd(x(:)); 

log_vecnorm = @(x) log(vecnorm(x));
symt = @(x) .5*(x+x');

metric_Bu = @(x) log(rms(x));
metric_W = @(x) log(vecnorm(x));
metric_C = @(x) log(trace(x));

% metric_C = @(x) real(logdet(x));
% metric_C = @(x) max(abs(eigs(x,1)));


metric_E = @(x) sum(x .* log(max(x,eps)));




n_times = dat.n_times;

% lag analysis
tsel = 1:100;
n_lag = 50;

% states
[...
    switch_Bu, repeat_Bu,...
    ...
    full_distSame_W,full_distDiff_W,...
    full_switch_W, full_repeat_W,...
    full_switch_C, full_repeat_C,...
    full_switch_Cr, full_repeat_Cr,...
    full_switch_Ci, full_repeat_Ci,...
    ctl_AA, ctl_AB, ctl_BA, ctl_BB,...
    ] = deal(nan(npt, dat.n_times));


% 
[dyn_full_switch, dyn_full_repeat,...
    dyn_AA, dyn_AB, dyn_BA, dyn_BB,...
    switch_v, repeat_v,...
    ] = deal(nan(x_disp,n_times,npt));

full_init = deal(nan(npt,1));

C  = cell(0);

g = nan(npt, x_disp);

lag_b= nan(npt,n_lag);

for pp = 1:npt


    % LOAD ========================================
    fdir = dir(sprintf('%s*Pt%d_xdim%d.mat',fld, pts(pp), x_disp));


    if isempty(fdir)
        continue;
    end

    try
        d = load(fullfile(fdir.folder, fdir.name), 'mdl');
        d.mdl;
    catch
        continue
    end

    fprintf('\npt %d', pp)


    
    % standardize realization
    T = diag(diag(d.mdl.Q.mat));

    A = T\d.mdl.A*T;
    B = T\d.mdl.B;
    B0 = T\d.mdl.B0;
    C{pp} = d.mdl.C*T;


    % calcuate Bu ========================================

    u_basis = dat.u_train(1:dat.n_bases,:,1);
    basis_norm = vecnorm(u_basis);

    

    if use_Bu==1

        pt_dat = load(fullfile(fdir.folder, fdir.name), 'dat');


        if norm_basis==1
            u_task = pt_dat.dat.u_train(ismember(pt_dat.dat.pred_name, 'task'),:,find(pt_dat.dat.trial.task(pt_dat.dat.sel_train)==1,1)) ./ basis_norm;
            u_prevTask = pt_dat.dat.u_train(ismember(pt_dat.dat.pred_name, 'prevTask'),:,find(pt_dat.dat.trial.prevTask(pt_dat.dat.sel_train)==1,1)) ./ basis_norm;
        else
            u_task = pt_dat.dat.u_train(ismember(pt_dat.dat.pred_name, 'task'),:,find(pt_dat.dat.trial.task(pt_dat.dat.sel_train)==1,1));
            u_prevTask = pt_dat.dat.u_train(ismember(pt_dat.dat.pred_name, 'prevTask'),:,find(pt_dat.dat.trial.prevTask(pt_dat.dat.sel_train)==1,1));
            u_switch = pt_dat.dat.u_train(ismember(pt_dat.dat.pred_name, 'switch'),:,find(pt_dat.dat.trial.switch(pt_dat.dat.sel_train)==1,1));
        end

        B_task = B(:, ismember(pt_dat.dat.pred_name, 'task'));
        B_prevTask = B(:, ismember(pt_dat.dat.pred_name, 'prevTask'));


        if task_prevTask==1

            Bu_switch = B_task*u_task;
            Bu_repeat = B_prevTask*u_prevTask;

        else

            Bu_switch = B_task*u_task - B_prevTask*u_prevTask;
            Bu_repeat = B_task*u_task + B_prevTask*u_prevTask;

            if use_switch

                B_switch = B(:, ismember(pt_dat.dat.pred_name, 'task@prevTask'));

                Bu_AA = B_task*u_task + B_prevTask*u_prevTask + B_switch*(u_prevTask.*u_task);
                Bu_AB = B_task*u_task - B_prevTask*u_prevTask - B_switch*(u_prevTask.*u_task);
                Bu_BA = -B_task*u_task + B_prevTask*u_prevTask - B_switch*(u_prevTask.*u_task);
                Bu_BB = -B_task*u_task - B_prevTask*u_prevTask + B_switch*(u_prevTask.*u_task);

            end

        end

        fprintf('-[%.4g, %.4g]-',...
            norm(u_task,'fro'),...
            norm(u_prevTask,'fro'))


    else

        if task_prevTask==1

            B_switch = B(:, ismember(dat.pred_name, 'task'));
            Bu_switch = B_switch * u_basis;

            B_repeat = B(:, ismember(dat.pred_name, 'prevTask'));
            Bu_repeat = B_repeat * u_basis;

        else

            B_switch = B(:, ismember(dat.pred_name, 'task')) - B(:, ismember(dat.pred_name, 'prevTask'));
            Bu_switch = B_switch * u_basis;

            B_repeat = B(:, ismember(dat.pred_name, 'task')) + B(:, ismember(dat.pred_name, 'prevTask'));
            Bu_repeat = B_repeat * u_basis;

        end

    end



    % ablate components

    switch norm_mode

        case 'recur'

            A = A*0;
            B0 = B0*0;

        case 'cue'

            A = A*0;
            B0 = B0*0;
            Bu_switch(:,1:25) = 0;
            Bu_repeat(:,1:25) = 0;

        case 'input'

            Bu_switch = normalize(Bu_switch,1,norm_method);
            Bu_repeat = normalize(Bu_repeat,1,norm_method);

    end





    % Input strength ==========================================
    switch_Bu(pp,:) = metric_Bu(Bu_switch);
    repeat_Bu(pp,:) = metric_Bu(Bu_repeat);


    % SIMUALTED timecourse ====================================
    for tt =1:n_times

        if tt ==1


            % INIT
            if use_Bu==1
                B0_prev = B0(:, ismember(pt_dat.dat.pred0_name, 'prevTask'));
                u0_prevTask = pt_dat.dat.u0_train(ismember(pt_dat.dat.pred0_name, 'prevTask'),find(pt_dat.dat.trial.prevTask(pt_dat.dat.sel_train)==1,1));
                B0_prev = B0_scale*B0_prev*u0_prevTask;
            else
                B0_prev = B0(:, ismember(dat.pred0_name, 'prevTask'));
            end

            X_full_switch = -B0_prev;
            X_full_repeat = B0_prev;

            C_full_switch = B0_prev*B0_prev';
            C_full_repeat = B0_prev*B0_prev';

            Cr_full_switch = zeros(x_disp,x_disp);
            Cr_full_repeat = zeros(x_disp,x_disp);

            Ci_full_switch = B0_prev*B0_prev';
            Ci_full_repeat = B0_prev*B0_prev';


            if use_switch
                X_AA = B0_prev;
                X_AB = -B0_prev;
                X_BA = B0_prev;
                X_BB = -B0_prev;

                C_AA = B0_prev*B0_prev';
                C_AB = B0_prev*B0_prev';
                C_BA = B0_prev*B0_prev';
                C_BB = B0_prev*B0_prev';
            end


        else


            % states
            X_full_switch = A*X_full_switch + Bu_switch(:,tt-1);
            X_full_repeat = A*X_full_repeat + Bu_repeat(:,tt-1);

            C_full_switch = A*C_full_switch*A' + Bu_switch(:,tt-1)*Bu_switch(:,tt-1)';
            C_full_repeat = A*C_full_repeat*A' + Bu_repeat(:,tt-1)*Bu_repeat(:,tt-1)';


            Cr_full_switch = A*C_full_switch*A';
            Cr_full_repeat = A*C_full_repeat*A';

            Ci_full_switch = Bu_switch(:,tt-1)*Bu_switch(:,tt-1)';
            Ci_full_repeat = Bu_repeat(:,tt-1)*Bu_repeat(:,tt-1)';


            if use_switch

                X_AA = A*X_AA + Bu_AA(:,tt-1);
                X_AB = A*X_AB + Bu_AB(:,tt-1);
                X_BA = A*X_BA + Bu_BA(:,tt-1);
                X_BB = A*X_BB + Bu_BB(:,tt-1);

                C_AA = A*C_AA*A' + Bu_AA(:,tt-1)*Bu_AA(:,tt-1)';
                C_AB = A*C_AB*A' + Bu_AB(:,tt-1)*Bu_AB(:,tt-1)';
                C_BA = A*C_BA*A' + Bu_BA(:,tt-1)*Bu_BA(:,tt-1)';
                C_BB = A*C_BB*A' + Bu_BB(:,tt-1)*Bu_BB(:,tt-1)';

            end


        end

        % metrics ==========
        full_switch_W(pp,tt) = metric_W(X_full_switch);
        full_repeat_W(pp,tt) = metric_W(X_full_repeat);

        full_switch_C(pp,tt) = metric_C(C_full_switch);
        full_repeat_C(pp,tt) = metric_C(C_full_repeat);

        full_init(pp) = log(rms(B0_prev));


        full_switch_Cr(pp,tt) = metric_C(Cr_full_switch);
        full_repeat_Cr(pp,tt) = metric_C(Cr_full_repeat);

        full_switch_Ci(pp,tt) = metric_C(Ci_full_switch);
        full_repeat_Ci(pp,tt) = metric_C(Ci_full_repeat);

        % state prox
        if ITI_model
            full_distSame_W(pp,tt) = metric_W(X_full_repeat);
            full_distDiff_W(pp,tt) = metric_W(X_full_repeat);
        else
            full_distSame_W(pp,tt) = metric_W(X_full_switch-X_full_repeat);
            full_distDiff_W(pp,tt) = metric_W(X_full_switch+X_full_repeat);
        end


        % dynamics
        dyn_full_switch(:,tt,pp) = X_full_switch;
        dyn_full_repeat(:,tt,pp) = X_full_repeat;

        if use_switch

            dyn_AA(:,tt,pp) = X_AA;
            dyn_AB(:,tt,pp) = X_AB;
            dyn_BA(:,tt,pp) = X_BA;
            dyn_BB(:,tt,pp) = X_BB;

            ctl_AA(pp,tt) = metric_C(C_AA);
            ctl_AB(pp,tt) = metric_C(C_AB);
            ctl_BA(pp,tt) = metric_C(C_BA);
            ctl_BB(pp,tt) = metric_C(C_BB);

        end


    end


    % cross-lag analysis
    if ~ITI_model
        cross_sim = cos_sim(dyn_full_switch(:,:,pp), dyn_full_repeat(:,:,pp));
        for ll = 1:n_lag
            xd = .5*(diag(cross_sim,ll) + diag(cross_sim,-ll));
            lag_b(pp,ll) = regress(xd, dat.dt*center(double([1:(n_times-ll)]')));
        end
    end


end






this_npt = sum(isfinite(switch_Bu(:,1)),1);
threshold = tinv(.95,this_npt)/sqrt(this_npt)

fprintf('N = %d\n', this_npt)


% save norms
dataset = strsplit(run_model, '_')

switch norm_mode

    case 'full'
        if ~ITI_model
            full_C = nanmean(full_switch_C-full_repeat_C, 2);
        else
            full_C = nanmean(full_switch_C,2);
        end

        full_G = nanmean(lag_b,2);

        full_dist = full_switch_W;
        switch dataset{1}
            case 'A23'
                A23_full_dist = full_switch_W;
            case 'H19'
                H19_full_dist = full_switch_W;
        end
                
    case 'recur'
        if ~ITI_model
            A_C = nanmean(full_switch_C-full_repeat_C, 2);
        else
            A_C = nanmean(full_switch_C(:,3:end),2);
        end

        disp('recur normalized!')

        A_G = nanmean(lag_b,2);

        A_dist = full_switch_W;
        switch dataset{1}
            case 'A23'
                A23_A_dist = full_switch_W;
            case 'H19'
                H19_A_dist = full_switch_W;
        end


    case 'input'

        if ~ITI_model
            Bu_C = nanmean(full_switch_C-full_repeat_C, 2);
        else
            Bu_C = nanmean(full_switch_C,2);
        end

        disp('inputs normalized!')

        Bu_G = nanmean(lag_b,2);

        Bu_dist = full_switch_W;
        switch dataset{1}
            case 'A23'
                A23_Bu_dist = full_switch_W;
            case 'H19'
                H19_Bu_dist = full_switch_W;
        end


end

if ITI_model
    full_repeat_W = full_repeat_W*0;
    full_repeat_C = full_repeat_C*0;
    
end




% STATS ====================================================
% MAGNITUDE
[~,full_W_pval, full_W_ci] = ttest(nansum(full_switch_W-full_repeat_W,2));
fprintf('\nMagnitude || d=%.2g, pval=%.2g, CI=[%.2g, %.2g]\n',  ...
    mean(nansum(full_switch_W-full_repeat_W,2))/std(nansum(full_switch_W-full_repeat_W,2)),...
    full_W_pval, full_W_ci(1), full_W_ci(2))

% CONTROL
sum_C = nansum(full_switch_C-full_repeat_C,2);
[~,full_C_pval, full_C_ci] = ttest(sum_C);

d_ci = bootstrap_rows(sum_C, @(x)mean(x)./std(x), 1e4);

fprintf('MEAN CONTROL || d=%.2g, pval=%.2g, bf=%.2g, CI=[%.2g, %.2g], d CI=[%.2g, %.2g]\n',...
    mean(sum_C)/std(sum_C), full_C_pval, log10(bf_ttest(sum_C)), full_C_ci(1), full_C_ci(2), d_ci(1),d_ci(3));

if use_switch

    % CONTROL
    sum_C = nansum((ctl_BA-ctl_AA) - (ctl_AB-ctl_BB),2);
    [~,full_C_pval, full_C_ci] = ttest(sum_C);

    d_ci = bootstrap_rows(sum_C, @(x)mean(x)./std(x), 1e4);

    fprintf('MEAN CONTROL (switch x task) || d=%.2g, pval=%.2g, bf=%.2g, CI=[%.2g, %.2g], d CI=[%.2g, %.2g]\n',...
        mean(sum_C)/std(sum_C), full_C_pval, log10(bf_ttest(sum_C)), full_C_ci(1), full_C_ci(2), d_ci(1),d_ci(3));




end



% lag Gen
sum_C = nansum(lag_b,2);
[~,full_C_pval, full_C_ci] = ttest(sum_C);
d_ci = bootstrap_rows(sum_C, @(x)mean(x)./std(x), 1e4);

fprintf('LAG GEN|| d=%.2g, pval=%.2g, bf=%.2g, CI=[%.2g, %.2g], d CI=[%.2g, %.2g]\n',...
    mean(sum_C)/std(sum_C), full_C_pval, log10(bf_ttest(sum_C)), full_C_ci(1), full_C_ci(2), d_ci(1),d_ci(3));



% INPUTS
[~,Bu_tt_pval, Bu_tt_ci] = ttest(nansum(switch_Bu - repeat_Bu,2));
fprintf('Inputs || pval=%.2g, CI=[%.2g, %.2g]\n', Bu_tt_pval, Bu_tt_ci(1), Bu_tt_ci(2))




% distance analysis
sel_same = double(2:n_times);
sel_diff = double(2:n_times);
r_same = (corr(full_distSame_W(:,sel_same)', (vec(sel_same))));
r_diff = (corr(full_distDiff_W(:,sel_diff)', (vec(sel_diff))));
r_con = (corr(full_distSame_W(:,sel_same)' - full_distDiff_W(:,sel_diff)', (vec(sel_diff))));
r_samediff = diag(corr(full_distSame_W(:,sel_same)', full_distDiff_W(:,sel_diff)'));

% r_same = (full_distSame_W / vec(1:100)');
% r_diff = (full_distDiff_W(:,2:end) / center(vec(2:100))');


[~,dist_t_p]=ttest(atanh([r_same, r_diff, r_same-r_diff, r_con, r_samediff]));
dist_rank_p(1)=signrank(r_same);
dist_rank_p(2)=signrank(r_diff);
dist_rank_p(3)=signrank(r_same-r_diff);
dist_rank_p(4)=signrank(r_con);
dist_rank_p(5)=signrank(r_samediff);

fprintf('dist same trend|| d=%.4g, ttest pval=%.4g, rank pval=%.4g\n', mean(atanh(r_same))/std(atanh(r_same)), dist_t_p(1), dist_rank_p(1))
fprintf('dist diff trend || r=%.4g, ttest pval=%.4g, rank pval=%.4g\n', median(r_diff), dist_t_p(2), dist_rank_p(2))
fprintf('dist comp trend || r=%.4g, ttest pval=%.4g, rank pval=%.4g\n', median(r_con), dist_t_p(4), dist_rank_p(4))
fprintf('dist compare || r=%.4g, ttest pval=%.4g, rank pval=%.4g\n', median(r_same-r_diff), dist_t_p(3), dist_rank_p(3))
fprintf('dist same-diff corr || r=%.4g, ttest pval=%.4g, rank pval=%.4g\n', median(r_samediff), dist_t_p(5), dist_rank_p(5))




% compare initials
switch run_model
    case 'H19_prevTask'
        init_dist_H19 = full_init;
        C_switch_H19 = full_switch_C;
        C_repeat_H19 = full_repeat_C;
        Bu_switch_H19 = switch_Bu;
        Bu_repeat_H19 = repeat_Bu;
    case'A23_prevTask'
        init_dist_A23 = full_init;
        C_switch_A23 = full_switch_C;
        C_repeat_A23 = full_repeat_C;
        Bu_switch_A23 = switch_Bu;
        Bu_repeat_A23 = repeat_Bu;
end

try

    [~,init_pval,~,init_stats] = ttest2(init_dist_H19,init_dist_A23)
    [init_pval,~,init_stats] = ranksum(init_dist_H19,init_dist_A23)

    [~,ctrl_pval,~,ctrl_stats] = ttest2(sum(C_switch_H19(:,2:25) - C_repeat_H19(:,2:25),2),sum(C_switch_A23(:,2:25) - C_repeat_A23(:,2:25),2))
    [ctrl_pval,~,ctrl_stats] = ranksum(sum(C_switch_H19(:,2:25) - C_repeat_H19(:,2:25),2),sum(C_switch_A23(:,2:25) - C_repeat_A23(:,2:25),2))

    figure; hold on;

    % distances
    nexttile; hold on;
    [f,x,bw]=ksdensity(init_dist_H19);
    plot(x,f, 'LineWidth',2, 'color', 'b')

    [f,x]=ksdensity(init_dist_A23, 'BandWidth', bw);
    plot(x,f, 'LineWidth', 2  , 'color', 'r')
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
    set(gcf, 'Renderer', 'Painters')

[f,x]=ksdensity(init_dist_A23/2, 'BandWidth', bw);
    plot(x,f, '--', 'LineWidth', 2  , 'color', 'r')
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
    set(gcf, 'Renderer', 'Painters')


    % Inputs
    nexttile; hold on;
    [f,x,bw]=ksdensity(sum(C_switch_H19(:,2:25) - C_repeat_H19(:,2:25),2));
    plot(x,f, 'LineWidth', 1.5, 'color', 'b')

    [f,x,bw]=ksdensity(sum(C_switch_A23(:,2:25) - C_repeat_A23(:,2:25),2));
    plot(x,f, 'LineWidth', 1.5, 'color', 'r')
    set(gca, 'TickDir', 'out', 'LineWidth', 1)
    title('cue control')


    ctrl_diff_h19 = mean(C_switch_H19- C_repeat_H19,2)
    ctrl_diff_a23 = mean(C_switch_A23- C_repeat_A23,2)

    [~,tot_ctrl_pval,~,tot_ctrl_stats] = ttest2(ctrl_diff_h19, ctrl_diff_a23)
    [tot_ctrl_pval,~,tot_ctrl_stats] = ranksum(ctrl_diff_h19, ctrl_diff_a23)
    [tot_ctrl_bf10,tot_ctrl_pValue] = bf_ttest2(ctrl_diff_h19, ctrl_diff_a23);
    tot_ctrl_bf10 = [tot_ctrl_bf10, 1/tot_ctrl_bf10, log10(tot_ctrl_bf10)]
    tot_ctrl_d = (mean(ctrl_diff_h19) - mean(ctrl_diff_a23)) / sqrt(.5*(var(ctrl_diff_h19)+ var(ctrl_diff_a23)))




% total inputs
nexttile; hold on;
[f,x,bw]=ksdensity(mean(C_switch_H19 - C_repeat_H19,2));
plot(x,f, 'LineWidth', 1.5, 'color', 'b')

[f,x,bw]=ksdensity(mean(C_switch_A23 - C_repeat_A23,2));
plot(x,f, 'LineWidth', 1.5, 'color', 'r')
set(gca, 'TickDir', 'out', 'LineWidth', 1)
title('total control')




    % relationship
    nexttile; hold on;
    plot(init_dist_H19, sum(Bu_switch_H19(:,dat.epoch==2),2), 'ob', 'MarkerSize',10, 'MarkerFaceColor','b');lsline;
    plot(init_dist_A23, sum(Bu_switch_A23(:,dat.epoch==2),2), 'or', 'MarkerSize',10, 'MarkerFaceColor','r');lsline;
    set(gca, 'TickDir', 'out', 'LineWidth', 1)





    % just error bars -------------------
    figure; hold on;


    % init
    nexttile; hold on;
    errorbar(.5, mean(init_dist_H19), sem(init_dist_H19), sem(init_dist_H19), 'ob',...
        'MarkerFaceColor', 'b', 'MarkerSize', 15, 'LineWidth', 2);
    errorbar(1.5, mean(init_dist_A23), sem(init_dist_A23), sem(init_dist_A23), 'or',...
        'MarkerFaceColor', 'r', 'MarkerSize', 15, 'LineWidth', 2);
    xlim([0,2])
    title('norm init')

    % cue control
       mean_Bu_H19 = mean(Bu_switch_H19 - Bu_repeat_H19,2);
     mean_Bu_A23 = mean(Bu_switch_A23 - Bu_repeat_A23,2);

    nexttile; hold on;
    errorbar(.5, mean(mean_Bu_H19), sem(mean_Bu_H19), sem(mean_Bu_H19), 'ob',...
        'MarkerFaceColor', 'b', 'MarkerSize', 15, 'LineWidth', 2);
    errorbar(1.5, mean(mean_Bu_A23), sem(mean_Bu_A23), sem(mean_Bu_A23), 'or',...
        'MarkerFaceColor', 'r', 'MarkerSize', 15, 'LineWidth', 2);
    xlim([0,2])
    title('norm inputs (total)')

    [~,in_total_p] =ttest2(mean_Bu_H19,mean_Bu_A23)


    % cue control inputs
     mean_Bu_H19 = mean(Bu_switch_H19(:,2:25) - Bu_repeat_H19(:,2:25),2);
     mean_Bu_A23 = mean(Bu_switch_A23(:,2:25) - Bu_repeat_A23(:,2:25),2);

    nexttile; hold on;
    errorbar(.5, mean(mean_Bu_H19), sem(mean_Bu_H19), sem(mean_Bu_H19), 'ob',...
        'MarkerFaceColor', 'b', 'MarkerSize', 15, 'LineWidth', 2);
    errorbar(1.5, mean(mean_Bu_A23), sem(mean_Bu_A23), sem(mean_Bu_A23), 'or',...
        'MarkerFaceColor', 'r', 'MarkerSize', 15, 'LineWidth', 2);
    xlim([0,2])
    title('norm inputs (cue)')

        [~,in_cue_p] =ttest2(mean_Bu_H19,mean_Bu_A23)



      % cue control inputs
     mean_Bu_H19 = mean(Bu_switch_H19(:,26:75) - Bu_repeat_H19(:,26:75),2);
     mean_Bu_A23 = mean(Bu_switch_A23(:,26:100) - Bu_repeat_A23(:,26:100),2);

    nexttile; hold on;
    errorbar(.5, mean(mean_Bu_H19), sem(mean_Bu_H19), sem(mean_Bu_H19), 'ob',...
        'MarkerFaceColor', 'b', 'MarkerSize', 15, 'LineWidth', 2);
    errorbar(1.5, mean(mean_Bu_A23), sem(mean_Bu_A23), sem(mean_Bu_A23), 'or',...
        'MarkerFaceColor', 'r', 'MarkerSize', 15, 'LineWidth', 2);
    xlim([0,2])
    title('norm inputs (delay)')


    [~,in_delay_p] =ttest2(mean_Bu_H19,mean_Bu_A23)



    % cue control inputs
    mean_Bu_H19 = mean(Bu_switch_H19(:,76:end) - Bu_repeat_H19(:,76:end),2);
    mean_Bu_A23 = mean(Bu_switch_A23(:,101:end) - Bu_repeat_A23(:,101:end),2);

    nexttile; hold on;
    errorbar(.5, mean(mean_Bu_H19), sem(mean_Bu_H19), sem(mean_Bu_H19), 'ob',...
        'MarkerFaceColor', 'b', 'MarkerSize', 15, 'LineWidth', 2);
    errorbar(1.5, mean(mean_Bu_A23), sem(mean_Bu_A23), sem(mean_Bu_A23), 'or',...
        'MarkerFaceColor', 'r', 'MarkerSize', 15, 'LineWidth', 2);
    xlim([0,2])
    title('norm inputs (trial)')


    [~,in_trial_p] =ttest2(mean_Bu_H19,mean_Bu_A23)



    set(gcf, 'Renderer', 'Painters')



    
catch ME
    
end




%% ====================   PLOT   ====================

do_tfce = 1


if do_tfce
    % cite: https://www.sciencedirect.com/science/article/pii/S1053811912010300
    addpath(genpath(sprintf('%s/src/analysis/MatlabTFCE-master',ROOT)))
    nsim=1e4;
    try
        parpool;
    catch
    end
end

figure;




% % CONTROLLABILITY BOTH ===================================
tsel = 2:n_times;
nexttile; hold on;

errorarea(tsel, nanmean(full_switch_C(:,tsel)), sem_wn(full_switch_C(:,tsel)), sem_wn(full_switch_C(:,tsel)), '-r');
if ~ITI_model
    errorarea(tsel, nanmean(full_repeat_C(:,tsel)), sem_wn(full_repeat_C(:,tsel)), sem_wn(full_repeat_C(:,tsel)), '-g');
end

title('Task Energy')
try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
ylabel('energy')
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
xlim([0, n_times+1])
set(gcf, 'Renderer', 'Painters')
% ylim([5, 8])

if do_tfce
    [pos,neg]=eeg_tfce(full_switch_C(:,tsel), full_repeat_C(:,tsel),nsim);
    
    plot_clusters_seperate(tsel,full_switch_C(:,tsel),full_repeat_C(:,tsel), pos, neg)
end




% CONTROLLABILITY (SWITCH) ===================================
if use_switch
    tsel = 2:n_times;

    nexttile; hold on;
    errorarea(tsel, nanmean(ctl_AA(:,tsel)), sem_wn(ctl_AA(:,tsel)), sem_wn(ctl_AA(:,tsel)), '-r');
    errorarea(tsel, nanmean(ctl_AB(:,tsel)), sem_wn(ctl_AB(:,tsel)), sem_wn(ctl_AB(:,tsel)), '-m');
    errorarea(tsel, nanmean(ctl_BA(:,tsel)), sem_wn(ctl_BA(:,tsel)), sem_wn(ctl_BA(:,tsel)), '-c');
    errorarea(tsel, nanmean(ctl_BB(:,tsel)), sem_wn(ctl_BB(:,tsel)), sem_wn(ctl_BB(:,tsel)), '-b');

    title('Task Flexibility')
    xline(find(dat.epoch == 3, 1)-1); try;xline(find(dat.epoch == 4, 1)-1);catch;end;
    ylabel('switch - repeat')
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
    xlim([0, n_times+1])
    set(gcf, 'Renderer', 'Painters')
    % ylim([5, 8])


    if do_tfce

        [pos,neg]=eeg_tfce(ctl_BA(:,tsel), ctl_AA(:,tsel),nsim);
        plot_clusters_seperate(tsel, ctl_BA(:,tsel), ctl_AA(:,tsel), pos, neg)

        [pos,neg]=eeg_tfce(ctl_AB(:,tsel), ctl_BB(:,tsel),nsim);
        plot_clusters_seperate(tsel, ctl_AB(:,tsel), ctl_BB(:,tsel), pos, neg)

        [pos,neg]=eeg_tfce(ctl_BA(:,tsel)-ctl_AA(:,tsel), ctl_AB(:,tsel)-ctl_BB(:,tsel),nsim);
        plot_clusters_seperate(tsel, .5*(ctl_BA(:,tsel) + ctl_AB(:,tsel)), .5*(ctl_AA(:,tsel) + ctl_BB(:,tsel)), pos, neg)

    end


end







% CONTROLLABILITY CONTRAST===================================
if ~ITI_model

    tsel = 2:n_times;

    nexttile; hold on;
    errorarea(tsel, nanmean(full_switch_C(:,tsel)-full_repeat_C(:,tsel)), sem_wn(full_switch_C(:,tsel)-full_repeat_C(:,tsel)), sem_wn(full_switch_C(:,tsel)-full_repeat_C(:,tsel)), '-k');

    title('Task Energy')
    try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
    ylabel('switch - repeat')
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
    yline(0, '-k')
    xlim([0, n_times+1])
    ylim([-.3,.3])



    if do_tfce
        [pos,neg]=eeg_tfce(full_switch_C(:,tsel), full_repeat_C(:,tsel),nsim);
        plot_clusters_seperate(tsel,full_switch_C(:,tsel)-full_repeat_C(:,tsel),full_switch_C(:,tsel)*0, pos, neg)
    end
end
set(gcf, 'Renderer', 'painters')






% Inputs ==================================================
nexttile; hold on;

errorarea(1:n_times, mean((switch_Bu)),sem_wn((switch_Bu)), sem_wn((switch_Bu)), '-r');
errorarea(1:n_times, mean((repeat_Bu)),sem_wn((repeat_Bu)), sem_wn((repeat_Bu)), '-g');


if do_tfce
    [pos,neg]=eeg_tfce((switch_Bu), (repeat_Bu),nsim);
    plot_clusters_seperate(1:n_times, (switch_Bu),(repeat_Bu),pos, neg)
end


title('Task Inputs')
yline(0, '-k', 'LineWidth', 2)
try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
ylabel('switch - repeat')
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)






% STATE DISTANCES ===================================
tsel = 1:n_times;

nexttile; hold on;
errorarea(tsel, nanmean((full_distSame_W(:,tsel))),sem_wn((full_distSame_W(:,tsel))), sem_wn((full_distSame_W(:,tsel))), '-r');
if ~ITI_model
    errorarea(tsel, nanmean(full_distDiff_W(:,2:end-2)),sem_wn(full_distDiff_W(:,2:end-2)), sem_wn(full_distDiff_W(:,2:end-2)), '-b');
end


if do_tfce
    [pos,neg]=eeg_tfce(full_distSame_W(:,tsel), full_distDiff_W(:,tsel),nsim);
    plot_clusters_seperate(tsel,full_distSame_W(:,tsel),full_distDiff_W(:,tsel),pos, neg)
end



set(gca, 'TickDir', 'out', 'LineWidth', 1.5)

title('State Distance')
try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
ylabel('distance')

xlim([0, n_times])
set(gcf, 'Renderer', 'painters')










% VELOCITY ===================================
tsel = 1:n_times;

[vel_switch, vel_repeat] = deal(nan(npt, n_times));
if use_switch
    [vel_AA, vel_AB, vel_BA, vel_BB] = deal(nan(npt, n_times));
end
for pp = 1:npt

    vel_switch(pp,:) = log(tangvelocity(dyn_full_switch(:,:,pp)'));
    if ~ITI_model
        vel_repeat(pp,:) = log(tangvelocity(dyn_full_repeat(:,:,pp)'));
    end
    if use_switch
        vel_AA(pp,:) = log(tangvelocity(dyn_AA(:,:,pp)'));
        vel_AB(pp,:) = log(tangvelocity(dyn_AB(:,:,pp)'));
        vel_BA(pp,:) = log(tangvelocity(dyn_BA(:,:,pp)'));
        vel_BB(pp,:) = log(tangvelocity(dyn_BB(:,:,pp)'));
    end

end




swre_con = mean(vel_switch(:,tsel)-vel_repeat(:,tsel),2);
[~,vel_switchRepeat_pval]=ttest(swre_con);
fprintf('\nvel sw - re con: d=%.4g, p=%.4g', mean(swre_con)/std(swre_con), vel_switchRepeat_pval)


r_vel_sw = (corr(vel_switch(:,tsel)', (vec(double(tsel)))));
r_vel_re = (corr(vel_repeat(:,tsel)', (vec(double(tsel)))));
[~,vel_time_pval]=ttest([r_vel_sw, r_vel_re]);
fprintf('\nvel sw trend: d=%.4g, p=%.4g', mean(r_vel_sw)/std(r_vel_sw), vel_time_pval(1))
fprintf('\nvel re rend: d=%.4g, p=%.4g', mean(r_vel_re)/std(r_vel_re), vel_time_pval(2))

[vel_sameDist_par, vel_diffDist_par] = deal(nan(npt,1));
for pp = 1:npt
    vel_sameDist_par(pp) = partialcorr(full_distSame_W(pp,tsel)', (vel_switch(pp,tsel) + vel_repeat(pp,tsel))', [vec(double(tsel)), center(vec(double(tsel))).^2]);
    vel_diffDist_par(pp) = partialcorr(full_distDiff_W(pp,tsel)', (vel_switch(pp,tsel) + vel_repeat(pp,tsel))', [vec(double(tsel)), center(vec(double(tsel))).^2]);
end

% vel_sameDist_r = diag(corr(full_distSame_W(:,tsel)', (vel_switch(:,tsel) + vel_repeat(:,tsel))'));
% vel_diffDist_r = diag(corr(full_distDiff_W(:,tsel)', (vel_switch(:,tsel) + vel_repeat(:,tsel))'));

cons = [vel_sameDist_par, vel_diffDist_par];

fprintf('\n vel~sameDist  // vel~diffDist')
[mean(cons)./std(cons)]
[~,vel_dist_pval]=ttest(cons)





nexttile; hold on;
errorarea(tsel, nanmean(vel_switch(:,tsel)), sem_wn(vel_switch(:,tsel)), sem_wn(vel_switch(:,tsel)), '-r');
errorarea(tsel, nanmean(vel_repeat(:,tsel)), sem_wn(vel_repeat(:,tsel)), sem_wn(vel_repeat(:,tsel)), '-g');

title('Task Velocity')
try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
ylabel('velocity')
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
set(gcf, 'Renderer', 'painters')

if do_tfce
    [pos,neg]=eeg_tfce(vel_switch(:,tsel), vel_repeat(:,tsel),nsim);
    plot_clusters_seperate(tsel,vel_switch(:,tsel),vel_repeat(:,tsel), pos, neg)
end

xlim([0, n_times])



if use_switch


    nexttile; hold on;

    errorarea(tsel, nanmean(vel_AA(:,tsel)), sem_wn(vel_AA(:,tsel)), sem_wn(vel_AA(:,tsel)), '-r');
    errorarea(tsel, nanmean(vel_AB(:,tsel)), sem_wn(vel_AB(:,tsel)), sem_wn(vel_AB(:,tsel)), '-m');
    errorarea(tsel, nanmean(vel_BA(:,tsel)), sem_wn(vel_BA(:,tsel)), sem_wn(vel_BA(:,tsel)), '-c');
    errorarea(tsel, nanmean(vel_BB(:,tsel)), sem_wn(vel_BB(:,tsel)), sem_wn(vel_BB(:,tsel)), '-b');


    title('Task Velocity')
    xline(find(dat.epoch == 3, 1)-1); try;xline(find(dat.epoch == 4, 1)-1);catch;end;
    ylabel('switch - repeat')
    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
    set(gcf, 'Renderer', 'painters')

    xlim([0, n_times])


end






%% STATE SIMILARITY

thresh_method = 'contour'
do_tfce = 0;

if ITI_model
    im_range = [.9,1]
else
   im_range= [-.3,.3]
end

% cite: https://www.sciencedirect.com/science/article/pii/S1053811912010300
addpath(genpath(sprintf('%s/src/analysis/MatlabTFCE-master', ROOT)))
nsim=1e4;
try
    parpool;
catch
end


triu_sel = triu(true(n_times,n_times));

[cross_sim,switch_sim, repeat_sim,lag_sim] = deal(nan(n_times, n_times, npt));

tsel=1:n_times;
n_lag = 50;
cross_diag = deal(nan(n_times, n_lag, npt));

for pp = 1:npt

    if ITI_model
        cross_sim(:,:,pp) = cos_sim(dyn_full_switch(:,:,pp), -dyn_full_repeat(:,:,pp));
        switch_sim(:,:,pp) = cos_sim(dyn_full_switch(:,:,pp), -dyn_full_switch(:,:,pp));
        repeat_sim(:,:,pp) = cos_sim(dyn_full_repeat(:,:,pp), -dyn_full_repeat(:,:,pp));

    else
        cross_sim(:,:,pp) = cos_sim(dyn_full_switch(:,:,pp), dyn_full_repeat(:,:,pp));
        switch_sim(:,:,pp) = cos_sim(dyn_full_switch(:,:,pp), dyn_full_switch(:,:,pp));
        repeat_sim(:,:,pp) = cos_sim(dyn_full_repeat(:,:,pp), dyn_full_repeat(:,:,pp));

    end
   

    lag_con = tril(cross_sim(:,:,pp),-1) - triu(cross_sim(:,:,pp),1)';
    lag_con(triu_sel) = nan;
    lag_sim(:,:,pp) = lag_con;


    for dd = 1:n_lag
        xd = .5*(diag(cross_sim(tsel,tsel,pp),dd) + diag(cross_sim(tsel,tsel,pp),-dd));
        cross_diag(1:length(xd),dd,pp) = xd;
    end

end



% cross-sim
figure;


alpha = 2/3;
sim_perm = 1000;
try
    parpool
catch
    
end

R_tfce = [];
R_tfce(:,:,1,:) = cross_sim;
R_tfce(1,1,1,:) = nan;



nexttile;

switch thresh_method

    case 'contour'


        imagesc(nanmean(cross_sim,3), im_range);

        if do_tfce
            [pcorr_pos, pcorr_neg] = matlab_tfce('onesample',2,R_tfce,[],[],sim_perm,2,1,8,.01,[],[]);
            p_map = max(1-pcorr_neg,1-pcorr_pos)>.975;

            hold on;
            contour(max(1-pcorr_neg,1-pcorr_pos)>.975, 1, '-k');
        else
            p_map = cross_sim*0;
        end

      

    case 'alpha'

        imagesc(mean(cross_sim,3), 'AlphaData', alpha + (1-alpha)*p_map, [-.3,.3]);
end


try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
try;yline(find(dat.epoch == 3, 1)-1); yline(find(dat.epoch == 4, 1)-1);catch;end;

if ITI_model
    colormap(batlow);
else
    colormap(vik);
end
axis('square');
title('switch - repeat similarity');


set(gcf, 'Renderer', 'painters')






%% SIM LAG ANALYSIS



addpath(genpath(sprintf('%s/src/analysis/MatlabTFCE-master',ROOT)))
nsim=1e4;
try
    parpool;
catch
end



triu_sel = triu(true(n_times,n_times));

[cross_sim,switch_sim, repeat_sim,lag_sim] = deal(nan(n_times, n_times, npt));

tsel=double(1:n_times);
tlen = length(tsel);
n_lag = 50
cross_diag = deal(nan(n_times, n_lag, npt));
full_cross_diag = deal(nan(n_times,n_times,npt));
for pp = 1:npt
    
    cross_sim(:,:,pp) = cos_sim(dyn_full_switch(:,:,pp), dyn_full_repeat(:,:,pp));
   
    lag_con = tril(cross_sim(:,:,pp),-1) - triu(cross_sim(:,:,pp),1)';
    lag_con(triu_sel) = nan;
    lag_sim(:,:,pp) = lag_con;


    for dd = 1:n_lag
        xd = .5*(diag(cross_sim(tsel,tsel,pp),dd) + diag(cross_sim(tsel,tsel,pp),-dd));
        cross_diag(1:length(xd),dd,pp) = xd;

        % cross_diag(:,dd,pp) = circshift(cross_diag(:,dd,pp), floor(sum(isnan(cross_diag(:,dd,pp)))/2));

    end


     for dd = 1:n_times
        xd = .5*(diag(cross_sim(tsel,tsel,pp),dd) + diag(cross_sim(tsel,tsel,pp),-dd));
        full_cross_diag(1:length(xd),dd,pp) = xd;
    end


end






% plot
figure;


% original similarity
im1=nexttile;
imagesc(mean(cross_sim,3), [-.3,.3]);
colormap(im1, vik)
axis('square')

xline(find(dat.epoch == 3, 1)-1); try;xline(find(dat.epoch == 4, 1)-1);catch;end;
yline(find(dat.epoch == 3, 1)-1); try;yline(find(dat.epoch == 4, 1)-1);catch;end;


[lag_r, lag_b] = deal(nan(npt, n_lag));

for pp = 1:npt

    % lag_r(pp,:)=corr(double([1:n_times]'), cross_diag(:,:,pp), 'rows', 'pairwise');

    for ll = 1:n_lag

        lag_b(pp,ll) = regress((cross_diag(1:(tlen-ll),ll,pp)), dat.dt*center(double([1:(tlen-ll)]')));

    end

end



 nexttile;hold on;
 errorarea(1:n_lag, nanmean(lag_b), nanstd(lag_b)/sqrt(npt), nanstd(lag_b)/sqrt(npt))
yline(0)

[pos,neg]=eeg_tfce(lag_b, lag_b*0,nsim);
plot_clusters_seperate(1:size(lag_b,2),lag_b,lag_b*0,pos, neg)


ylim([-.25, .75])
yticks([-.25,.75])
set(gca, 'TickDir', 'out', 'LineWidth', 1.4)

sig_lag = find(pos>.025,1)-1
if isempty(sig_lag)
    sig_lag = length(pos);
end
sig_lag_dt = sig_lag*dat.dt


im2=nexttile;
if n_lag >50
    b50_col = load('batlow100.mat'); b50_col = b50_col.batlow100;
else
    b50_col = load('batlow50.mat'); b50_col = b50_col.batlow50;
end

imagesc(toeplitz([linspace(1, 256,50), ones(1,n_times-50)]));
colormap(im2, b50_col)
axis('square')

hold on;
plot([1,n_times-sig_lag], [sig_lag,n_times], '-k', 'LineWidth', 1)
plot([n_times,sig_lag], [n_times-sig_lag,1], '-k', 'LineWidth', 1)









nexttile; hold on;
for dd = 1:n_lag

    % errorarea((1+dd):n_times, nanmean(cross_diag(1:(n_times-dd),dd,:),3),...
    %     nanstd(cross_diag(1:(n_times-dd),dd,:),[],3)/sqrt(npt),  nanstd(cross_diag(1:(n_times-dd),dd,:),[],3)/sqrt(npt),...
    %     'LineWidth', 1.5, 'color',  b50_col(dd,:));

    plot(1:(n_times-dd), nanmean(cross_diag(1:(n_times-dd),dd,:),3),...
        '-','LineWidth', 3, 'color',  b50_col(dd,:));

    if dd == sig_lag
        plot(1:(n_times-dd), nanmean(cross_diag(1:(n_times-dd),dd,:),3), '-','LineWidth', 5, 'color',  'k');
    end

    % if dd > sig_lag
    %     plot((1+dd):n_times, nanmean(cross_diag(1:(n_times-dd),dd,:),3),...
    %         '-','LineWidth', 3, 'color',  [b50_col(dd,:), .33]);
    % end


end
xline(find(dat.epoch==3,1)-1, '-k')
xline(find(dat.epoch==4,1)-1, '-k')
yline(0,'-k', 'LineWidth',1.5)
set(gca, 'TickDir', 'out', 'LineWidth', 1.4)
ylim([-.6, .6])
yticks([-.6, .6])





im5=nexttile;


% shift_cross_diag = cross_diag;
% for pp = 1:size(cross_diag,3)
%     for dd = 1:size(cross_diag,2)
%         shift_cross_diag(:,dd,pp) = circshift(shift_cross_diag(:,dd,pp), floor(sum(isnan(shift_cross_diag(:,dd,pp)))/2));
%     end
% end


imagesc(mean(full_cross_diag,3),[-.3,.3])
xline(sig_lag, '-w')
yline(find(dat.epoch==3,1)-1, '-w')
yline(find(dat.epoch==4,1)-1, '-w')

colormap(im5, vik)
axis('square')




%

im6=nexttile;

m_cross = isfinite(mean(full_cross_diag,3)).*[1:n_lag, nan(1,max(tsel)-n_lag)];

imagesc(m_cross)
yline(find(dat.epoch==3,1)-1, '-w')
yline(find(dat.epoch==4,1)-1, '-w')

colormap(im6, batlow)
axis('square')






% cosine < lag

nexttile; hold on;
mean_lag = nan(tt,dd);
for dd = 1:n_lag

    mean_lag = nan(n_times,1);
    for tt = 1:(n_times-dd)

        mean_lag(tt) = nanmean(vec(full_cross_diag(tt, 1:dd,:)));

    end



    plot(1:n_times, mean_lag,...
        '-','LineWidth', 3, 'color',  b50_col(dd,:));


end
xline(find(dat.epoch==3,1)-1, '-k')
xline(find(dat.epoch==4,1)-1, '-k')

yline(0,'-k')
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)

set(gcf, 'Renderer', 'painters')



%% PLOT SVD TEMPORAL DYNAMICS ====================================

time_sel = 1:n_times;
% align_to = 2;
n_sv = 4;

plot_dims = [1,2,3]

if n_times == 125
    az =  -101.9977;
    el =  16.6780;
else
    az = 5.6920;
    el = 6.7144;
end



% normalize subjects
sub_dyn = cat(2,dyn_full_switch,dyn_full_repeat);
for pp = 1:npt
    sub_dyn(:,:,pp) = mxnorm(sub_dyn(:,:,pp));
end


norm_all = reshape(permute(sub_dyn,[2,1,3]), [2*n_times, npt*x_disp]);
norm_switch = norm_all(1:n_times,:);
norm_repeat = norm_all((n_times+1):end,:);


% do SVD
[~,S,V]=svds(norm_all, n_sv);
v_switch = norm_switch*V;
v_repeat = norm_repeat*V;




% get epochs
dot_sel = zeros(size(time_sel));
dot_sel(find(dat.epoch(time_sel)==3,1))=1;
dot_sel(find(dat.epoch(time_sel)==4,1))=1;

plt=figure;
hold on;
plot_3d_set(mean(v_repeat,3), mean(v_switch,3), time_sel, dot_sel, plot_dims)
xlabel(sprintf('dim %d', plot_dims(1)))
ylabel(sprintf('dim %d', plot_dims(2)))
zlabel(sprintf('dim %d', plot_dims(3)))



view(az,el);
axis('equal')
set(gcf, 'Renderer', 'painters')




if use_switch

    time_sel = 1:n_times
    n_tsel = length(time_sel)

    % normalize subjects
    sub_dyn = cat(2, dyn_AA(:,time_sel,:), dyn_AB(:,time_sel,:), dyn_BA(:,time_sel,:), dyn_BB(:,time_sel,:));
    for pp = 1:npt
        sub_dyn(:,:,pp) = mxnorm(sub_dyn(:,:,pp));
    end


    norm_all = reshape(permute(sub_dyn,[2,1,3]), [4*n_tsel, npt*x_disp]);
    norm_AA = norm_all(1:n_tsel,:);
    norm_AB = norm_all((1*n_tsel+1):(2*n_tsel),:);
    norm_BA = norm_all((2*n_tsel+1):(3*n_tsel),:);
    norm_BB = norm_all((3*n_tsel+1):(4*n_tsel),:);


    [~,S,V]=svds(norm_all);
    v_AA = norm_AA*V;
    v_AB = norm_AB*V;
    v_BA = norm_BA*V;
    v_BB = norm_BB*V;



    % get epochs
    dot_sel = zeros(size(time_sel));
    dot_sel(find(dat.epoch(time_sel)==3,1))=1;
    dot_sel(find(dat.epoch(time_sel)==4,1))=1;

    plt=figure;
    hold on;
    grid('on');

    xline(0, '-k', 'LineWidth', 2); yline(0, '-k', 'LineWidth', 1.5);
    axis('equal');
    set(gca, 'TickDir', 'none', 'LineWidth', 1.5)

    plot_3d(v_AA', 'r', {dot_sel, plot_dims});
    plot_3d(v_AB', 'm', {dot_sel, plot_dims});
    plot_3d(v_BB', 'b', {dot_sel, plot_dims});
    plot_3d(v_BA', 'c', {dot_sel, plot_dims});
    


    xlabel(sprintf('dim %d', plot_dims(1)))
    ylabel(sprintf('dim %d', plot_dims(2)))
    zlabel(sprintf('dim %d', plot_dims(3)))



    view(az,el);
    axis('equal')
    set(gcf, 'Renderer', 'painters')



end




%% RT ANALYSIS (LME)


y_dist = @(x,y) diag(cos_sim(x,y));




dv_mx = [];
iv_mx = [];

pt_b = [];

epoch = dat.epoch;
b_mx=[];

ntimes = size(dyn_full_repeat,2)
model_time = cell(1,ntimes);
for tt = 1:ntimes
    model_time{tt} = zeros(0,7);
end

model_sum = zeros(0,7);


for pp = 1:npt


    % LOAD ========================================
    fdir = dir(sprintf('%s*Pt%d_xdim%d.mat',fld, pts(pp), x_disp));


    if isempty(fdir)
        continue;
    end

    try
        d = load(fullfile(fdir.folder, fdir.name), 'mdl', 'dat');
        d.mdl; d.dat;
    catch
        fprintf('\nfailed pt %d\n', pp)
        continue
    end

    fprintf('pt %d\n', pp)


    % epochs and trials
    trial_sel = d.dat.sel_train;
    
    task_trial = d.dat.trial.task(trial_sel);
    switch_trial = d.dat.trial.switch(trial_sel);

    % parameters
    pt = ones(sum(trial_sel),1)*pp;

    rt = d.dat.trial.RT(trial_sel);
    logrt = log(rt);

    prevRT = d.dat.trial.prevRT(trial_sel);
    logprevRT = (log(prevRT));

    task = center(d.dat.trial.task(trial_sel));
    swrep = center(d.dat.trial.switch(trial_sel));

    y = d.dat.y_train;
    y = permute(y, [1,3,2]);




    task_pred = zeros(d.dat.y_dim, d.dat.n_train, d.dat.n_times);
    task_pred(:,switch_trial==1,:) = permute(repmat(C{pp}*dyn_full_switch(:,:,pp), [1,1,sum(switch_trial==1)]), [1,3,2]);
    task_pred(:,switch_trial==-1,:) = permute(repmat(C{pp}*dyn_full_repeat(:,:,pp), [1,1,sum(switch_trial==-1)]), [1,3,2]);
    task_pred = task_pred .* permute(repmat(task_trial, [ 1, d.dat.n_times,d.dat.y_dim]), [3,1,2]);

    y_norm = (log(squeeze(vecnorm(y,1))));
    task_norm = (log(squeeze(vecnorm(task_pred,1))));


    b_est = [];%nan(size(y,3),6); % update with num preds
    ysim_sum =zeros(size(y,2),1);
    parfor tt = 1:size(y,3)

        ysim = (y_dist(task_pred(:,:,tt), y(:,:,tt)));
        % model_time{tt} = [model_time{tt}; [pt,task, swrep, task.*swrep, center(y_norm(:,tt)), center(ysim), logrt]];
        
        ysim_sum = ysim_sum + ysim;
    end

    model_sum = [model_sum; [pt, task, swrep, task.*swrep, center(mean(y_norm,2)), center(ysim_sum./size(y,2)), center(mean(task_norm,2)), logrt]];

end


 
T = array2table(model_sum, 'VariableNames', {'pt','task', 'switch', 'taskSwitch', 'ynorm', 'ysim','tasknorm', 'logrt'});
collintest(T)

lme = fitlme(T, 'logrt ~ 1 + task * switch + ysim + ynorm + tasknorm + (1 + task * switch + ysim + ynorm + tasknorm | pt)',...
    'FitMethod', 'REML', 'DummyVarCoding', 'effects', 'CheckHessian', true);
[ffx_est,ffx_name,ffx_stats]= fixedEffects(lme, 'DFMethod', 'satterthwaite');
lme
ffx_stats

b_sel = ismember(ffx_stats.Name, 'ysim');
beta_t =  ffx_stats.tStat(b_sel)
beta_p = ffx_stats.pValue(b_sel)
beta_d = ffx_stats.tStat(b_sel)/sqrt(ffx_stats.DF(b_sel))
beta_d = ffx_stats.Estimate(b_sel)/(ffx_stats.SE(b_sel)*sqrt(ffx_stats.DF(b_sel)))




% stats
calc_d = @(x) nanmean(x)./nanstd(x);


[~,pval_all]=ttest(sum(b_mx,2))
[pval_all_rank]=signrank(sum(b_mx,2))
d_all = calc_d(sum(b_mx,2))

[~,pval_2]=ttest(sum(b_mx(:,epoch==2),2))
[~,pval_3]=ttest(sum(b_mx(:,epoch==3),2))
[~,pval_23]=ttest(sum(b_mx(:,epoch==2 | epoch==3),2))
[~,pval_4]=ttest(sum(b_mx(:,epoch==4),2))

CI_all = bootstrap_rows(sum(b_mx,2), @(x) mean(x), 1e4)

% do cluster correction
figure;
nexttile; hold on;
yline(0);

CI_d_all = bootstrap_rows(sum(b_mx,2), @(x) calc_d(x), 1e4)


[boot_ci,boot_sd] = bootstrap_rows(b_mx - mean(b_mx,2), calc_d, 10000);
boot_sd = boot_sd * ((size(b_mx,2)-1)/size(b_mx,2));

errorarea(1:n_times, calc_d(b_mx),boot_sd, boot_sd, '-k', 'LineWidth',2);


if do_tfce
    [pos,neg]=eeg_tfce(b_mx, b_mx*0, nsim);
    plot_clusters(b_mx, b_mx*0,pos,neg);
else
    stats = test_cluster_mass(b_mx,threshold,1e4);
    plot_clusters(b_mx, b_mx*0,stats.pos_pval, stats.neg_pval);
end

title('subj-level RT~loading')
xline(find(dat.epoch == 3, 1)-1); try;xline(find(dat.epoch == 4, 1)-1);catch;end;
ylabel('RT~loading')
set(gca, 'TickDir', 'out', 'LineWidth', 1.5)
% ylim([-.04, .04]);
xlim([0, n_times+1])

set(gcf, 'Renderer', 'painters')




%% ====================   PLOT ABLATIONS  ====================

if ITI_model

    figure; hold on;


    % STATE DISTANCES ===================================
    tsel = 1:n_times;

    nexttile; hold on;
    errorarea(tsel, nanmean((full_dist(:,tsel))),sem_wn((full_dist(:,tsel))), sem_wn((full_dist(:,tsel))), '-k');
    errorarea(tsel, nanmean((Bu_dist(:,tsel))),sem_wn((Bu_dist(:,tsel))), sem_wn((Bu_dist(:,tsel))), '-r');


    set(gca, 'TickDir', 'out', 'LineWidth', 1.5)

    title('State Distance')
    try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
    ylabel('distance')

    % ylim([2.9, 6])
    xlim([0, n_times])
    set(gcf, 'Renderer', 'painters')

    try
        % STATE DISTANCES ===================================
        tsel = 1:n_times;

        nexttile; hold on;
        errorarea(tsel, nanmean(H19_full_dist(:,tsel)),sem_wn(H19_full_dist(:,tsel)), sem_wn(H19_full_dist(:,tsel)), '-b');
        errorarea(tsel, nanmean(H19_Bu_dist(:,tsel)),sem_wn(H19_Bu_dist(:,tsel)), sem_wn(H19_Bu_dist(:,tsel)), '--b');

        errorarea(tsel, nanmean(A23_full_dist(:,tsel)),sem_wn(A23_full_dist(:,tsel)), sem_wn(A23_full_dist(:,tsel)), '-r');
        errorarea(tsel, nanmean(A23_Bu_dist(:,tsel)),sem_wn(A23_Bu_dist(:,tsel)), sem_wn(A23_Bu_dist(:,tsel)), '--r');


        set(gca, 'TickDir', 'out', 'LineWidth', 1.5)

        title('State Distance')
        try;xline(find(dat.epoch == 3, 1)-1); xline(find(dat.epoch == 4, 1)-1);catch;end;
        ylabel('distance')

        % ylim([2.9, 6])
        xlim([0, n_times])
        set(gcf, 'Renderer', 'painters')
    catch

    end

end













figure; hold on;


Cs = [full_C, A_C, Bu_C];

c_sem_wn = sem(Cs - mean(Cs,2) + mean(Cs(:)))*(3/2);
[boot_ci,boot_sd] = bootstrap_rows(Cs - mean(Cs,2) + mean(Cs(:)), @mean, 10000);
cis = abs(diff(boot_ci))*(3/2) % morey correction
            

swarmchart(vec(ones(npt,3).*[.95,1.95,2.95]), vec(Cs), 40, 'ok', 'MarkerFaceColor', 'w', 'LineWidth', 1)

errorbar(.95, mean(full_C), c_sem_wn(1), c_sem_wn(1), 'ok',...
    'MarkerFaceColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);

errorbar(1.95, mean(A_C), c_sem_wn(2), c_sem_wn(2), 'ok',...
    'MarkerFaceColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);

errorbar(2.95, mean(Bu_C), c_sem_wn(3), c_sem_wn(3), 'ok',...
    'MarkerFaceColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);


% ylim([-.02,.12])
yline(0, '-k', 'LineWidth',2)
xlim([0.5,3.5])
xticks(1:3)
xticklabels({'full', 'no recur', 'no inputs'})
set(gca, 'TickDir', 'out', 'LineWidth', 2)
% axis('square')
set(gcf, 'Renderer', 'Painters')
title('task energy')







figure; hold on;


Cs = [full_G, A_G, Bu_G];

c_sem_wn = sem(Cs - mean(Cs,2) + mean(Cs(:)))*(3/2);
[boot_ci,boot_sd] = bootstrap_rows(Cs - mean(Cs,2) + mean(Cs(:)), @mean, 10000);
cis = abs(diff(boot_ci))*(3/2) % morey correction
  

swarmchart(vec(ones(npt,3).*[1,2,3]), vec(Cs), 40, 'ok', 'MarkerFaceColor', 'w', 'LineWidth', 1)


% generalization
errorbar(1, mean(full_G), c_sem_wn(1), c_sem_wn(1), 'ok',...
    'MarkerFaceColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);


errorbar(2, mean(A_G), c_sem_wn(2), c_sem_wn(2), 'ok',...
    'MarkerFaceColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);

errorbar(3, mean(Bu_G), c_sem_wn(3), c_sem_wn(3), 'ok',...
    'MarkerFaceColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);

% ylim([-(.6/.12)*.07,.6])



% ylim([-.02,.12])
yline(0, '-k', 'LineWidth',2)
xlim([0.5,3.5])
xticks(1:3)
xticklabels({'full', 'no recur', 'no inputs'})
set(gca, 'TickDir', 'out', 'LineWidth', 2)
% axis('square')
set(gcf, 'Renderer', 'Painters')
title('task gen')








[~,p,~,stats]=ttest([full_C, A_C] - Bu_C);
ci=bootstrap_rows([full_C, A_C] - Bu_C, @mean, 1000);
fprintf('\ngramian: full vs input: d=%.2g, p=%.2g, CI=[%.2g, %.2g]', stats.tstat(1)/sqrt(npt), p(1), ci(1,1), ci(3,1));
fprintf('\ngramian: recur vs input: d=%.2g, p=%.2g, CI=[%.2g, %.2g]\n', stats.tstat(2)/sqrt(npt), p(2), ci(1,2), ci(3,2));



[~,p,~,stats]=ttest([full_G, Bu_G] - A_G);
ci=bootstrap_rows([full_G, Bu_G] - A_G, @mean, 1000);
fprintf('\ntaskgen: full vs input: d=%.2g, p=%.2g, CI=[%.2g, %.2g]', stats.tstat(1)/sqrt(npt), p(1), ci(1,1), ci(3,1));
fprintf('\ntaskgen: recur vs input: d=%.2g, p=%.2g, CI=[%.2g, %.2g]\n', stats.tstat(2)/sqrt(npt), p(2), ci(1,2), ci(3,2));

[~,p,~,stats]=ttest(Cs);
ci=bootstrap_rows(Cs, @mean, 1000);
fprintf('\ntaskgen vs0: d=[%.4g, %.4g, %.4g] p=[%.4g, %.4g, %.4g]\n', stats.tstat./sqrt(npt), p);
disp(ci([1,3],:))







%% PLOT PARAMETERS

x_disp = 128
do_align = 1
plot_all = 1;

f1=figure;
% f2= figure;

% get data
for pp = 1:npt

    fdir = dir(fullfile(fld, sprintf('*Pt%d_xdim%d.mat',pts(pp), x_disp)));
    try
        d = load(fullfile(fdir.folder, fdir.name), 'prm', 'dat', 'mdl');
        prm = d.prm;
        dat = d.dat;
        mdl = d.mdl;
        clear d

    catch
        continue
    end
    break;
end


u_basis = dat.u_train(1:dat.n_bases,:,1);
dat.pred_list = [{'spline'}; dat.pred_list];

clear C
ev = [];
B_vn = nan(size(u_basis,2),length(dat.pred_list),npt);
for pp = 1:npt


    % LOAD ========================================
    fdir = dir(sprintf('%s*Pt%d_xdim%d.mat',fld, pts(pp), x_disp));


    if isempty(fdir)
        continue;
    end

    try
        d = load(fullfile(fdir.folder, fdir.name), 'mdl');
        d.mdl;
    catch
        continue
    end

    fprintf('pt %d\n', pp)



    if do_align==1
     
        T = diag(diag(d.mdl.Q.mat));

        A = T\d.mdl.A*T;
        B = T\d.mdl.B;
        B0 = T\d.mdl.B0;
        C{pp} = d.mdl.C*T;


    else
        A = d.mdl.A;
        B = d.mdl.B;
        B0 = d.mdl.B0;
        C = d.mdl.C;
    end


    % get A eigs
    ev = [ev; eig(A)];


    % figure(f2);
    % nexttile;
    % imagesc(B);
    


% figure(f1)

    e_pt = eig(A);
    nexttile;hold on;
    set(gca, "LineWidth", 1, 'TickDir', 'out', 'FontSize', 10);
    set(gcf, 'Position', [100,100,500,500])
    title('A eigs')
    axis('square')

    ylim([-1.5,1.5])
    xlim([-1.5 1.5])
    xline(0, '-k')
    yline(0, '-k')
    plot(sin(-pi:.001:pi), cos(-pi:.001:pi), '-k', 'LineWidth',1)
    plot(real(e_pt), imag(e_pt), 'ok', 'LineWidth', 1, 'MarkerSize', 8, 'MarkerFaceColor', 'w');


    % add contour
    [f,x]=ksdensity([real(e_pt), imag(e_pt)], 'PlotFcn', 'contour');
    x1 = x(:,1); x2=x(:,2);
    x = linspace(min(x1),max(x1));
    y = linspace(min(x2),max(x2));
    [xq,yq] = meshgrid(x,y);
    orig_state = warning;
    warning('off','all');
    z = griddata(x1,x2,f,xq,yq);
    contour(xq,yq,z, 5,'LineWidth', 1)


    % get B vecnorm
    for bb = 1:dat.n_pred

        b_sel = ismember(dat.pred_name, dat.pred_list{bb});
        B_vn(:,bb,pp) = vecnorm(B(:,b_sel)*u_basis);

    end



end


% plot eigenvalues
ff0=figure;
colormap(ff0,batlow);


nexttile;hold on;
set(gca, "LineWidth", 1, 'TickDir', 'out', 'FontSize', 10);
set(gcf, 'Position', [100,100,500,500])
title('A eigs')
axis('square')

ylim([-1.5,1.5])
xlim([-1.5 1.5])
xline(0, '-k')
yline(0, '-k')
plot(sin(-pi:.001:pi), cos(-pi:.001:pi), '-k', 'LineWidth',1)
plot(real(ev), imag(ev), 'ok', 'LineWidth', 1, 'MarkerSize', 8, 'MarkerFaceColor', 'w');


% add contour
[f,x]=ksdensity([real(ev), imag(ev)], 'PlotFcn', 'contour');
x1 = x(:,1); x2=x(:,2);
x = linspace(min(x1),max(x1));
y = linspace(min(x2),max(x2));
[xq,yq] = meshgrid(x,y);
orig_state = warning;
warning('off','all');
z = griddata(x1,x2,f,xq,yq);
contour(xq,yq,z, 5,'LineWidth', 1)

set(gcf, 'Renderer', 'painters')






% plot B vecnorm 
figure;
nexttile; hold on;
set(gca, "LineWidth", 1, 'TickDir', 'out', 'FontSize', 10);
set(gcf, 'Position', [100,100,500,500])
title('B norm')
for bb = 2:length(dat.pred_list)
    plot(mean(B_vn(:,bb,:),3), '-', 'LineWidth', 1.5)
end
legend(dat.pred_list(2:end))


% plot B_task
figure;
set(gcf, 'Position', [100,100,500,500])
n1=nexttile;
btask = mdl.B(:,ismember(dat.pred_name, 'taskSwitch'));
imagesc(btask, [-max(abs(btask(:))),max(abs(btask(:)))])
colormap(n1,cork)
axis('equal')

n2=nexttile;
imagesc(dat.u_train(ismember(dat.pred_name, 'spline'),:,1), [-1,1])
axis('equal')

colormap(n2,cork)

set(gcf, 'Renderer', 'painters')


