function plt = plot_cohenD(x, col,linespec)

hold on;
plt=plot(nanmean(x)./nanstd(x), linespec, 'LineWidth', 3, 'Color',col, 'MarkerFaceColor', col, 'MarkerSize',10);

end