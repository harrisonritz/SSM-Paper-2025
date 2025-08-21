function plot_3d(x, col, varargin)

if ~isempty(varargin) && iscell(varargin{1}) && (length(varargin{1})>1) 
    dims = varargin{1}{2};
else
    dims = [1:3];
end

hold on;
plot3(x(dims(1),:), x(dims(2),:), x(dims(3),:),...
    '-', 'color', col, 'LineWidth', 4);

if strcmp(col,'k')
    return
end

plot3(x(dims(1),1), x(dims(2),1), x(dims(3),1),...
    'ok',   'LineWidth', 1, 'MarkerSize', 25, 'MarkerFaceColor', 'w');

% plot3(x(dims(1),end), x(dims(2),end), x(dims(3),end),...
%     'ok',  'LineWidth', 1, 'MarkerSize', 25, 'MarkerFaceColor', col);


% plot epochs
if ~isempty(varargin)

    varg = varargin{1};

    if iscell(varg)
        finds = find(varg{1});
    else
        finds = find(varg);
    end
  


    for pp = finds
        plot3(x(dims(1),pp), x(dims(2),pp), x(dims(3),pp),...
            'ok',  'LineWidth', 1, 'MarkerSize', 20, 'MarkerFaceColor', col);
    end
end



end
