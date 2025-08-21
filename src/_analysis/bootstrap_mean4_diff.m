function [ci,sd] = bootstrap_mean4_diff(x, y, z, n_boot)
center = @(x) x-nanmean(x)

samps = nan(n_boot,1);

x11_idx = randi(size(x{1},1), [size(x{1},1),n_boot]);
x22_idx = randi(size(x{3},1), [size(x{3},1),n_boot]);

y11_idx = randi(size(y{1},1), [size(y{1},1),n_boot]);
y22_idx = randi(size(y{3},1), [size(y{3},1),n_boot]);

z11_idx = randi(size(z{1},1), [size(z{1},1),n_boot]);
z22_idx = randi(size(z{3},1), [size(z{3},1),n_boot]);

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

    z_ii = [...
        mean(z{1}(z11_idx(:,ii),:)),...
        mean(z{2}(z11_idx(:,ii),:)),...
        mean(z{3}(z22_idx(:,ii),:)),...
        mean(z{4}(z22_idx(:,ii),:))
        ];

    samps(ii) = ...
        (1 - (meansqr(x_ii-z_ii)/sqrt(var(x_ii,1)*var(z_ii,1)))) - ...
        (1 - (meansqr(y_ii-z_ii)/sqrt(var(y_ii,1)*var(z_ii,1))));

   
    % x_ii = ...
    %     mean(x{1}(x11_idx(:,ii),:)) + mean(x{2}(x11_idx(:,ii),:)) -...
    %     mean(x{3}(x22_idx(:,ii),:)) + mean(x{4}(x22_idx(:,ii),:));
    % 
    % y_ii = ...
    %     mean(y{1}(y11_idx(:,ii),:)) + mean(y{2}(y11_idx(:,ii),:)) -...
    %     mean(y{3}(y22_idx(:,ii),:)) + mean(y{4}(y22_idx(:,ii),:));
    % 
    % z_ii = ...
    %     mean(z{1}(z11_idx(:,ii),:)) + mean(z{2}(z11_idx(:,ii),:)) -...
    %     mean(z{3}(z22_idx(:,ii),:)) + mean(z{4}(z22_idx(:,ii),:));
    % 
    % 
    % m_z = normalize(z_ii,2,'norm')';
    % samps(ii) = ...
    %     (normalize(x_ii,2,'norm')*m_z) -...
    %     (normalize(y_ii,2,'norm')*m_z);






end



ci = prctile(samps,[2.5,50,97.5]);
sd = std(samps);

end