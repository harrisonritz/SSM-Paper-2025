function [ci,sd] = bootstrap_mean_sim(x, y, n_boot)

samps = nan(n_boot, 1);

x_idx = randi(size(x,1), [size(x,1),n_boot]);
y_idx = randi(size(y,1), [size(y,1),n_boot]);

parfor ii = 1:n_boot

    
    samps(ii) = ...
        normalize(mean(x(x_idx(:,ii),:)),2,'norm') * ...
        normalize(mean(y(y_idx(:,ii),:)),2,'norm')';

   
    % samps(ii) =  1 - (meansqr(x_ii-y_ii)/sqrt(var(y_ii,1)*var(x_ii,1)));

end



ci = prctile(samps,[2.5,50,97.5]);
sd = std(samps);

end