function [pcorr_pos,pcorr_neg]=eeg_tfce(x,y,nsim)


sw_Bu(:,1,1,:) = x';
re_Bu(:,1,1,:) = y';
[pcorr_pos, pcorr_neg] = matlab_tfce('paired',2,sw_Bu,re_Bu,[],nsim,2,1,4,.01,[],[]);

end