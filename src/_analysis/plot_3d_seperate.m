function plot_3d_seperate(dyn_repeat_a, dyn_repeat_b, dyn_switch_a, dyn_switch_b, time_sel, varargin)


hold on;

grid('on');

xline(0, '-k', 'LineWidth', 1.5); yline(0, '-k', 'LineWidth', 1.5);
axis('equal');
set(gca, 'TickDir', 'none', 'LineWidth', 1.5)

if ~isempty(varargin)

    plot_3d(dyn_repeat_a(time_sel,:)', 'r', varargin{1});
    plot_3d(dyn_switch_a(time_sel,:)', 'm', varargin{1});
    plot_3d(dyn_repeat_b(time_sel,:)', 'b', varargin{1});
    plot_3d(dyn_switch_b(time_sel,:)', 'c', varargin{1});
    
else

    plot_3d(dyn_repeat_a(time_sel,:)', 'r');
    plot_3d(dyn_switch_a(time_sel,:)', 'm');
    plot_3d(dyn_repeat_b(time_sel,:)', 'b');
    plot_3d(dyn_switch_b(time_sel,:)', 'c');

end


end