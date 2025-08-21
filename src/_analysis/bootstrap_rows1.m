function [ci,sd,pval] = bootstrap_rows1(x, fcn, n_boot)


samps = nan(n_boot, 1);
idx = randi(size(x,1), [size(x,1),n_boot]);

for ii = 1:n_boot
    samps(ii) = fcn(x(idx(:,ii), :));
end

ci = [prctile(samps,2.5); fcn(x); prctile(samps,97.5)];
sd = std(samps);
pval = min(mean(samps>0), 1-mean(samps>0));

end