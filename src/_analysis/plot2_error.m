function plt = plot2_error(x, col,linespec)

n = sum(isfinite(x(:,2)));

hold on;
plt=plot(nanmean(x(:,1,:),3), nanmean(x(:,2,:),3), linespec, 'LineWidth', 3, 'Color',col, 'MarkerFaceColor', col, 'MarkerSize',10);
% plot(nanmean(x(:,1,:),3) + nanstd(x(:,1,:),[],3)./sqrt(n), nanmean(x(:,2,:),3), '-', 'LineWidth', .33, 'Color', col);
% plot(nanmean(x(:,1,:),3) - nanstd(x(:,1,:),[],3)./sqrt(n), nanmean(x(:,2,:),3), '-', 'LineWidth', .33, 'Color',col);
% plot(nanmean(x(:,1,:),3), nanmean(x(:,2,:),3) + nanstd(x(:,1,:),[],3)./sqrt(n), '-', 'LineWidth', .33, 'Color', col);
% plot(nanmean(x(:,1,:),3), nanmean(x(:,2,:),3)- nanstd(x(:,1,:),[],3)./sqrt(n), '-', 'LineWidth', .33, 'Color',col);


plot(nanmean(x(:,1,:),3) + nanstd(x(:,1,:),[],3)./sqrt(n), nanmean(x(:,2,:),3) + nanstd(x(:,1,:),[],3)./sqrt(n), '-', 'LineWidth', .33, 'Color', col);
plot(nanmean(x(:,1,:),3) - nanstd(x(:,1,:),[],3)./sqrt(n), nanmean(x(:,2,:),3) - nanstd(x(:,1,:),[],3)./sqrt(n), '-', 'LineWidth', .33, 'Color',col);


end