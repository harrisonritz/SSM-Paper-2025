function [ci,pval] = bootstrap_rows2(x,y, fcn, n_boot)


samps = nan(n_boot, size(x,2));
idx = randi(size(x,1), [size(x,1),n_boot]);

for ii = 1:n_boot
    samps(ii,:) = fcn(x(idx(:,ii), :)) - fcn(y(idx(:,ii), :));
end

ci = [prctile(samps,2.5); prctile(samps,50); prctile(samps,97.5)];

pval = min(mean(samps<0), mean(samps>0))*2;

end