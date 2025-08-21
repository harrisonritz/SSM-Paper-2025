function plt = plot_line(x, col, linespec)

plt=plot(nanmean(x,1), linespec, 'LineWidth', 3, 'Color',col, 'MarkerFaceColor', col, 'MarkerSize',10);

end