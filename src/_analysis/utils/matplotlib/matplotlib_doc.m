%% Perceptually Uniform Colormaps from MatPlotLib
% The <www.mathworks.com/matlabcentral/fileexchange/62729
% MatPlotLib Perceptually Uniform Colormaps> submission includes the
% default colormap family and default line colororder family from
% MatPlotLib 2 and 3. This document shows examples of their usage.
%% Overview
matplotlib_plot()
%% |VIRIDIS|
% * Sequential
% * Perceptually uniform
% * Default colormap for MatPlotLib 2 and 3
close()
load spine
imagesc(X)
colormap(viridis)
colorbar()
%% |CIVIDIS|
% * Sequential
% * Perceptually uniform
% * Optimized for color-vision deficiency / colorblind
colormap(cividis)
%% |INFERNO|
% * Sequential
% * Perceptually uniform
colormap(inferno)
%% |MAGMA|
% * Sequential
% * Perceptually uniform
colormap(magma)
%% |PLASMA|
% * Sequential
% * Perceptually uniform
colormap(plasma)
%% |TWILIGHT|
% * Cyclical
% * Perceptually uniform
% * Distinct end-colors
colormap(twilight)
%% |TWILIGHT_SHIFTED|
% * Cyclical
% * Perceptually uniform
% * Distinct end-colors
colormap(twilight_shifted)
%% |TAB10|
% * Qualitative
% * Default line ColorOrder for MatPlotLib 2 and 3
% * Named VEGA10 in MatPlotLib 2
close()
V = 1:10;
X = linspace(0,10,1000).';
Y = V.*sin(X+2*V*pi./10);
plot(X,Y, 'linewidth',4)
legend('location','eastoutside')
colororder(tab10(10))
%% |TAB20|
% * Qualitative
% * Named VEGA20 in MatPlotLib 2
V = 1:20;
Y = V.*sin(X+2*V*pi./20);
plot(X,Y, 'linewidth',4)
legend('location','eastoutside')
colororder(tab20(20))
%% |TAB20B|
% * Qualitative
% * Named VEGA20B in MatPlotLib 2
legend('location','eastoutside')
colororder(tab20b(20))
%% |TAB20C|
% * Qualitative
% * Named VEGA20C in MatPlotLib 2
legend('location','eastoutside')
colororder(tab20c(20))