function [ci,sd] = bootstrap_mean4_sim(x, y, n_boot)


% center = @(x) x-nanmean(x);

samps = nan(n_boot, 1);

x11_idx = randi(size(x{1},1), [size(x{1},1),n_boot]);
x22_idx = randi(size(x{3},1), [size(x{3},1),n_boot]);

y11_idx = randi(size(y{1},1), [size(y{1},1),n_boot]);
y22_idx = randi(size(y{3},1), [size(y{3},1),n_boot]);

% keyboard

parfor ii = 1:n_boot

    x_ii = [...
        mean(x{1}(x11_idx(:,ii),:)),...
        mean(x{2}(x11_idx(:,ii),:)),...
        mean(x{3}(x22_idx(:,ii),:)),...
        mean(x{4}(x22_idx(:,ii),:))
        ];

    y_ii = [...
        mean(y{1}(y11_idx(:,ii),:)),...
        mean(y{2}(y11_idx(:,ii),:)),...
        mean(y{3}(y22_idx(:,ii),:)),...
        mean(y{4}(y22_idx(:,ii),:))
        ];

    % R2
    samps(ii) =  1 - (meansqr(y_ii - x_ii)/sqrt(var(y_ii,1)*var(x_ii,1)));

    % cos
    % samps(ii) = normalize(x_ii,2,'norm')*normalize(y_ii,2,'norm')';

    % x_ii = ...
    %     mean(x{1}(x11_idx(:,ii),:)) + mean(x{2}(x11_idx(:,ii),:)) -...
    %     mean(x{3}(x22_idx(:,ii),:)) + mean(x{4}(x22_idx(:,ii),:));
    % 
    % y_ii = ...
    %     mean(y{1}(y11_idx(:,ii),:)) + mean(y{2}(y11_idx(:,ii),:)) -...
    %     mean(y{3}(y22_idx(:,ii),:)) + mean(y{4}(y22_idx(:,ii),:));
    % 
    % samps(ii) = normalize(x_ii,2,'norm')*normalize(y_ii,2,'norm')';




end



ci = prctile(samps,[2.5,50,97.5]);
sd = std(samps);

end