function [ci,sd] = bootstrap_mean_diff(x, y, z, n_boot)

samps = nan(n_boot, 1);

x_idx = randi(size(x,1), [size(x,1),n_boot]);
y_idx = randi(size(y,1), [size(y,1),n_boot]);
z_idx = randi(size(z,1), [size(z,1),n_boot]);
% m_z = normalize(mean(z),2,'norm')';


parfor ii = 1:n_boot


    m_z = normalize(mean(z(z_idx(:,ii),:)),2,'norm')';

    samps(ii) = ...
        ((normalize(mean(x(x_idx(:,ii),:)),2,'norm')*m_z) -...
        normalize(mean(y(y_idx(:,ii),:)),2,'norm'))*m_z;



    % x_ii = mean(x(x_idx(:,ii),:));
    % y_ii = mean(y(y_idx(:,ii),:));
    % z_ii = mean(z(z_idx(:,ii),:));
    % 
    % samps(ii) =  ...
    %     (1 - (var(x_ii-z_ii)/sqrt(var(x_ii)*var(z_ii)))) - ...
    %     (1 - (var(y_ii-z_ii)/sqrt(var(y_ii)*var(z_ii))));

end



ci = prctile(samps,[2.5,50,97.5]);
sd = std(samps);

end