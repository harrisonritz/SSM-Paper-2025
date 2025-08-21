function plot_2d_set(dyn_repeat, dyn_switch, time_sel, varargin)


dims = [1,2; 2,3;1,3];


for ii = 1:3
    

    nexttile; hold on; hold on;

    grid('on');
    xline(0, '-k', 'LineWidth', 1.5); yline(0, '-k', 'LineWidth', 1.5);
    axis('equal');
    set(gca, 'TickDir', 'none', 'LineWidth', 1.5)
    % xticklabels(""); yticklabels(""); zticklabels("");

    if ~isempty(varargin)

        plot_2d(dyn_repeat(time_sel,dims(ii,:))', 'r', varargin{1});
        plot_2d(dyn_switch(time_sel,dims(ii,:))', 'm', varargin{1});
        plot_2d(-dyn_repeat(time_sel,dims(ii,:))', 'b', varargin{1});
        plot_2d(-dyn_switch(time_sel,dims(ii,:))', 'c', varargin{1});
    else

        plot_2d(dyn_repeat(time_sel,dims(ii,:))', 'r');
        plot_2d(dyn_switch(time_sel,dims(ii,:))', 'm');
        plot_2d(-dyn_repeat(time_sel,dims(ii,:))', 'b');
        plot_2d(-dyn_switch(time_sel,dims(ii,:))', 'c');

    end

    axis('equal');
    xlabel(sprintf('dim %d', dims(ii,1)))
    ylabel(sprintf('dim %d', dims(ii,2)))


end


end