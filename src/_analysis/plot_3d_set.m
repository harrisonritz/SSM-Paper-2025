function plot_3d_set(dyn_repeat, dyn_switch, time_sel, varargin)


hold on;

grid('on');

xline(0, '-k', 'LineWidth', 2); yline(0, '-k', 'LineWidth', 1.5);
axis('equal');
set(gca, 'TickDir', 'none', 'LineWidth', 1.5)

if ~isempty(varargin)

    plot_3d(dyn_repeat(time_sel,:)', 'r', varargin);
    plot_3d(dyn_switch(time_sel,:)', 'm', varargin);
    plot_3d(-dyn_repeat(time_sel,:)', 'b', varargin);
    plot_3d(-dyn_switch(time_sel,:)', 'c', varargin);
    
else

    plot_3d(dyn_repeat(time_sel,:)', 'r');
    plot_3d(dyn_switch(time_sel,:)', 'm');
    plot_3d(-dyn_repeat(time_sel,:)', 'b');
    plot_3d(-dyn_switch(time_sel,:)', 'c');

end
% 
% xticklabels([])
% yticklabels([])
% zticklabels([])


end