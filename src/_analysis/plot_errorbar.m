function plt = plot_errorbar(x, col,linespec)

n = sum(isfinite(x(:,1)));
plt=errorbar(nanmean(x), nanstd(x)./sqrt(n), linespec, 'LineWidth', 1.5, 'Color',col, 'MarkerFaceColor', col, 'MarkerSize',10);

end