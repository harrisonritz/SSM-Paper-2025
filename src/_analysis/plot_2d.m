function plot_2d(x, col, varargin)


plot(x(1,:), x(2,:),  '-', 'color', col, 'LineWidth', 2);
plot(x(1,1), x(2,1),  'o', 'color', col,  'LineWidth', 1, 'MarkerSize', 15, 'MarkerFaceColor', 'w');
plot(x(1,end), x(2,end),  'o', 'color', col,  'LineWidth', 1, 'MarkerSize', 15, 'MarkerFaceColor', col);


% plot epochs
if ~isempty(varargin)
    for pp = find(varargin{1})
        plot(x(1,pp), x(2,pp),  'o', 'color', col,  'LineWidth', 1, 'MarkerSize', 10, 'MarkerFaceColor', col);
    end
end



end
