function [pcorr_pos,pcorr_neg]=eeg_tfce4(x1,x2,y1,y2,nsim)


mx_x1(:,1,1,:) = x1';
mx_x2(:,1,1,:) = x2';

mx_y1(:,1,1,:) = y1';
mx_y2(:,1,1,:) = y2';

fprintf('NOT DEVELOPED')

% pred = [ones(size(x1,1)), ones(size(x1,1)), ones(size(x1,1)), ones(size(x1,1)), 


[pcorr_pos, pcorr_neg] = matlab_tfce('regression', 2, mx, nsim, 2, 1, 4, .01, [], []);

end