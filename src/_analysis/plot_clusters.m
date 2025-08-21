function plot_clusters(x,y,pcorr_pos, pcorr_neg)

diff_Bu = mean(x - y);

[h,~,~,stat] = ttest(x-y);
h(~any(x-y)) = 0;
h_pos = h & stat.tstat > 0;
h_neg = h & stat.tstat < 0;


hold on;
% plot(diff_Bu, '-k', 'LineWidth',1)

plot(find(h_pos), diff_Bu(:,h_pos), 'o', 'Color', [.5,.5,.5], 'LineWidth',1.5,'MarkerSize', 8)
plot(find(h_neg), diff_Bu(:,h_neg), 'o', 'Color', [.5,.5,.5], 'LineWidth',1.5,'MarkerSize', 8)

% plot(find(pcorr_pos<.05), diff_Bu(:,pcorr_pos<.05), 'oc', 'MarkerFaceColor',[.5,.5,.5], 'LineWidth',1.5,'MarkerSize', 10)
% plot(find(pcorr_neg<.05), diff_Bu(:,pcorr_neg<.05), 'om', 'MarkerFaceColor',[.5,.5,.5], 'LineWidth',1.5,'MarkerSize', 10)

plot(find(pcorr_pos<.025), diff_Bu(:,pcorr_pos<.025), 'ok',  'LineWidth',1.5,'MarkerSize', 10)
plot(find(pcorr_neg<.025), diff_Bu(:,pcorr_neg<.025), 'ok',  'LineWidth',1.5,'MarkerSize', 10)

yline(0);

end