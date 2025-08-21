%% plot parameter recovery

%% setup

clear;clc;
addpath(genpath('../'))

pt = 2;
xdim = 112;

% EEG
% d= load(sprintf('/Users/hr0283/Brown Dropbox/Harrison Ritz/HallM_NeSS/src/validation/paramer_recovery_prevTask_xdim%d_pt%d.mat',xdim,pt));


% GRU
ROOT = ''
d=load(sprtinf('%s/src/validation/recovery_results/paramrec__GRU_450-50-short_7__xdim112_pt7.mat',ROOT))

cork = load('cork.mat'); 
cmap = cork.cork;

%% make plot

plot_Bcol=0
plot_equal=1
log_C=1
average_bases=0

figure;
set(gca, "LineWidth", 1, 'TickDir', 'out', 'FontSize', 10);
set(gcf, 'Position', [100,100,500,500])

vec = @(x) x(:)
cos_sim = @(x,y) normalize(x,'norm')'*normalize(y,'norm');
ka = @(x,y) sum(x'*y, 'all')/(sqrt(sum(x'*x,'all'))*sqrt(sum(y'*y,'all')))
R2 = @(x,y) 1 - (meansqr(x(:)-y(:))/sqrt(var(y(:))*var(x(:)))) 


% A
p1 = d.a1 - eye(size(d.a1));
p2 = d.a2 - eye(size(d.a1));
min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]); 
title('A orig')
axis('square')
xticks([]);yticks([])

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]); 
title('A rec')
axis('square')
xticks([]);yticks([])
colormap(plt2, cmap)
colormap(plt1, cmap)

A_r = corr(p1(:), p2(:))
A_r2 = R2(p1(:), p2(:))
A_beta = p1(:)\p2(:)


% B
u = d.dat.u_train(:,:,2);
u1 = d.dat.u_train(1:20,:,1);
uu = repmat(u1,[9,1]);
p1 = d.b1;
p2 = d.b2;

nbases = double(d.dat.n_bases);
if average_bases

    % p1o = p1;
    % p2o = p2;
    % p1 = [];
    % p2 = [];
    % 
    % for bb = 1:nbases:size(p1o,2)
    %     p1 = [p1, mean(p1o(:,bb:bb+(nbases-1)),2)]; 
    %     p2 = [p2, mean(p2o(:,bb:bb+(nbases-1)),2)];
    % end


    p1 = d.b1*d.dat.u_train(:,:,1);
    p2 = d.b2*d.dat.u_train(:,:,1);

    for ii = 2:size(d.dat.u_train,3)
        p1 = p1 + d.b1*d.dat.u_train(:,:,ii);
        p2 = p2 + d.b2*d.dat.u_train(:,:,ii);
    end

end




min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]); 
title('B orig')
xticks([]);yticks([])
if plot_equal
    axis('equal')
end

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]); 
title('B rec')
xticks([]);yticks([])
colormap(plt2, cmap)
colormap(plt1, cmap)
if plot_equal
    axis('equal')
end

B_r = corr(p1(:), p2(:))
B_beta = p1(:)\p2(:)
B_r2 = R2(p1(:), p2(:))


B_bias = mean(p1(:)-p2(:))
B_bias_pct = ((B_bias^2 / (B_bias^2 + var(p1(:)-p2(:))))*100)


if plot_Bcol
    for ii = 1:20:size(p1,2)

        p1ii = vec(p1(:,ii:(ii+19))*u1);
        p2ii = vec(p2(:,ii:(ii+19))*u1);

        B_bias_ii = mean(p1ii(:)-p2ii(:));
        B_bias_ii = ((B_bias^2 / (B_bias^2 + var(p1ii(:)-p2ii(:))))*100);


        sz_range = min(max(abs([min([p2ii]), max([p2ii])])),max(abs([min([p1ii]), max([p1ii])])))


        nexttile;hold on;
        plot(p1ii, p2ii, 'or')
        xlabel('rec')
        ylabel('orig')
        plot([-sz_range,sz_range], [-sz_range,sz_range], '-k', 'LineWidth', 1.5)
        lsline;
        % title([...
        %     1 - (meansqr(p1ii-p2ii)/var(p2ii,1)), ...
        %     corr(p1ii,p2ii),...
        %     p1ii\p2ii,...
        %     B_bias_ii])
        title(sprintf('r2=%.2g \n r=%.2g \n b=%.2g \n %s=%.2g',...
            1 - (meansqr(p1ii-p2ii)/var(p2ii,1)), ...
            corr(p1ii,p2ii),...
            p1ii\p2ii,...
            'Δ',B_bias_ii))
        axis('equal')

        [min([p1ii;p2ii]), max([p1ii;p2ii])]
    end
end


nexttile; hold on;
plot(log(vecnorm(p2)), diag(corr(p1, p2)), 'ok', 'MarkerFaceColor', 'w'); lsline;
set(gca, 'TickDir', 'out', 'LineWidth', 1)
ylim([0,1])

B_norm_rec =corr(log(vecnorm(p2))', diag(corr(p1, p2)))






% B0
p1 = d.B01;
p2 = d.B02;
min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]); 
title('B0 orig')
xticks([]);yticks([])
if plot_equal
    axis('equal')
end

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]); 
title('B0 rec')
xticks([]);yticks([])
colormap(plt2, cmap)
colormap(plt1, cmap)
if plot_equal
    axis('equal')
end

B0_r = corr(p1(:), p2(:))
B0_r2 = R2(p1(:), p2(:))
B0_beta = p1(:)\p2(:)

B0_bias = mean(p1(:)-p2(:))
B0_bias_pct = (mean(p1(:)-p2(:))^2)/(mean(p1(:)-p2(:))^2 + var(p1(:)-p2(:)))

nexttile; hold on;
plot(log(vecnorm(p2)), diag(corr(p1, p2)), 'ok', 'MarkerFaceColor', 'w'); lsline;
set(gca, 'TickDir', 'out', 'LineWidth', 1)
ylim([0,1])

B0_norm_rec = corr(log(vecnorm(p2))', diag(corr(p1, p2)))






% C
p1 = d.c1;
p2 = d.c2;

if log_C
    p1 = sign(p1).*log(1+abs(p1));
    p2 = sign(p2).*log(1+abs(p2));
end


min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]);
title('C orig')
xticks([]);yticks([])
if plot_equal
    axis('equal')
end

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]);
title('C rec')
xticks([]);yticks([])
if plot_equal
    axis('equal')
end
colormap(plt2, cmap)
colormap(plt1, cmap)

C_r = corr(p1(:), p2(:))
C_r2 = R2(p1(:), p2(:))
C_beta = p1(:)\p2(:)






% Q
p1 = d.q1;
p2 = d.q2.mat;
min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]); 
title('Q orig')
axis('square')
xticks([]);yticks([])

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]); 
title('Q rec')
axis('square')
xticks([]);yticks([])
colormap(plt2, cmap)
colormap(plt1, cmap)


triu_sel = triu(true(size(p1)));
chol_p1 = chol(p1);
chol_p2 = chol(p2);

Q_ka = ka(p1,p2)
Q_r = corr(chol_p1(triu_sel),chol_p2(triu_sel))
Q_r2 = 1 - (meansqr(chol_p1(triu_sel)-chol_p2(triu_sel))/sqrt(var(chol_p1(triu_sel))*var(chol_p2(triu_sel))))
Q_beta = chol_p1(triu_sel)\chol_p2(triu_sel)






% P0
p1 = d.p01;
p2 = d.p02.mat;
min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]); 
title('P0 orig')
axis('square')
xticks([]);yticks([])

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]); 
title('P0 rec')
axis('square')
xticks([]);yticks([])
colormap(plt2, cmap)
colormap(plt1, cmap)


triu_sel = triu(true(size(p1)));
chol_p1 = chol(p1);
chol_p2 = chol(p2);

P0_ka = ka(p1,p2)
P0_r = corr(chol_p1(triu_sel),chol_p2(triu_sel))
P0_r2 = 1 - (meansqr(chol_p1(triu_sel)-chol_p2(triu_sel))/sqrt(var(chol_p1(triu_sel))*var(chol_p2(triu_sel))))
P0_beta = chol_p1(triu_sel)\chol_p2(triu_sel)






% R
p1 = d.fr1;
p2 = d.fr2.mat;
min_val = min([p1(:); p2(:)]);
max_val = max([p1(:); p2(:)]);
absmax_val = max(abs([p1(:); p2(:)]))*.5;


plt2=nexttile;
imagesc(p2,[-absmax_val, absmax_val]); 
title('R orig')
axis('square')
xticks([]);yticks([])

plt1=nexttile;
imagesc(p1,[-absmax_val, absmax_val]); 
title('R rec')
axis('square')
xticks([]);yticks([])
colormap(plt2, cmap)
colormap(plt1, cmap)


triu_sel = triu(true(size(p1)));
chol_p1 = chol(p1);
chol_p2 = chol(p2);

R_ka = ka(p1,p2)
R_r = corr(chol_p1(triu_sel),chol_p2(triu_sel))
R_r2 = 1 - (meansqr(chol_p1(triu_sel)-chol_p2(triu_sel))/sqrt(var(chol_p1(triu_sel))*var(chol_p2(triu_sel))))
R_beta = chol_p1(triu_sel)\chol_p2(triu_sel)



set(gcf, 'Renderer', 'painters')



fprintf(...
    ['\n',...
    'A: r = %.4g, r^2 = %.4g\n',...
    'B: r = %.4g, r^2 = %.4g\n',...
    'C: r = %.4g, r^2 = %.4g\n',...
    'W: r = %.4g, r^2 = %.4g\n',...
    'V: r = %.4g, r^2 = %.4g\n',...
    'B0: r = %.4g, r^2 = %.4g\n',...
    'W0: r = %.4g, r^2 = %.4g\n',...
    ],...
    A_r,A_r2,...
B_r, B_r2,...
C_r, C_r2,...
Q_r, Q_r2,...
R_r, R_r2,...
B0_r, B0_r2,...
P0_r, P0_r2...
)










