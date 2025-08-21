function [pval] = permutation_rows1(x, fcn, n_perm)


samps = nan(n_perm, 1);
sgn = sign(randn(size(x,1),n_perm));

effect  = fcn(x, ones(size(x,1),1));

for ii = 1:n_perm
    samps(ii) = fcn(x, sgn(:,ii));
end

pval = min(mean(samps>=effect), 1-mean(samps>=effect));
perm95 = prctile(samps, [2.5,97.5])

end