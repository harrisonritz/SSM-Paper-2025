% CI figures




figure; hold on; 

% task geometry ----------------------------------------------------
% Hall-McMaster
% taskgen rnn1: avg sim r= 0.152 CI: [-0.03379 to 0.3636]
% taskgen rnn2: avg sim r= 0.7081 CI: [0.5378 to 0.811]
% taskgen diff: avg sim r= -0.557 CI: [-0.8329 to -0.1809] / pval = 0

% Arnau
% taskgen rnn1: avg sim r= 0.5353 CI: [0.2528 to 0.7483]
% taskgen rnn2: avg sim r= 0.8121 CI: [0.7006 to 0.87]
% taskgen diff: avg sim r= -0.2823 CI: [-0.4637 to -0.07647] / pval = 0

hm = [...
    0.7081, 0.5378, 0.811;...
    0.152,-0.03379,0.3636;...
    ];


arnau = [...
    0.8127,0.7047, 0.8708;...
    0.5371, 0.2561, 0.7479;...
    ];


nexttile; hold on;
errorbar(.5:1.5, hm(:,1), abs(hm(:,2)-hm(:,1)), abs(hm(:,3)-hm(:,1)),...
    "ok", 'MarkerSize', 10, 'LineWidth',1.5, 'MarkerFaceColor', 'k')
xlim([0,2])
yline(0)
ylim([-.1,1])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
title('Task Geometry (Hall-McMaster)')

nexttile; hold on;
errorbar(.5:1.5, arnau(:,1), abs(arnau(:,2)-arnau(:,1)), abs(arnau(:,3)-arnau(:,1)),...
        "ok", 'MarkerSize', 10, 'LineWidth',1.5, 'MarkerFaceColor', 'k')
xlim([0,2])
yline(0)
ylim([-.1,1])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
title('Task Geometry (Arnau)')




% task energy ----------------------------------------------------


% Hall-McMaster
% energy rnn1: avg sim r= -0.08743 CI: [-0.3969 to 0.2226]
% energy rnn2: avg sim r= 0.7968 CI: [0.4769 to 0.9199]
% energy diff: avg sim r= -0.8707 CI: [-1.18 to -0.8707], perm pval = 0 

% Arnau
% energy rnn1: avg sim r= -0.6868 CI: [-0.8599 to -0.212]
% energy rnn2: avg sim r= 0.6057 CI: [0.2022 to 0.8408]
% energy diff: avg sim r= -1.263 CI: [-1.622 to -1.263], perm pval = 0 



hm = [...
    0.7968, 0.4769, 0.9199;...
    -0.08743, -0.3969, 0.2226...
    ];

arnau = [...
    0.6057, 0.2022, 0.8408;...
    -0.6868, -0.8599, -0.212...
    ];

 

nexttile; hold on;
errorbar(.5:1.5, hm(:,1), abs(hm(:,2)-hm(:,1)), abs(hm(:,3)-hm(:,1)),...
    "ok", 'MarkerSize', 10, 'LineWidth',1.5, 'MarkerFaceColor', 'k')
xlim([0,2])
yline(0)
ylim([-1,1])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
title('Task Energy (Hall-McMaster)')


nexttile; hold on;
errorbar(.5:1.5, arnau(:,1), abs(arnau(:,2)-arnau(:,1)), abs(arnau(:,3)-arnau(:,1)),...
        "ok", 'MarkerSize', 10, 'LineWidth',1.5, 'MarkerFaceColor', 'k')
xlim([0,2])
yline(0)
ylim([-1,1])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
title('Task Energy (Arnau)')







% midpoint ----------------------------------------------------


% midpoint rnn1: mse = 0.278 CI: [0.06259 to 0.4603] max:[0.9573]
% midpoint rnn2: mse = 0.5765 CI: [0.3529 to 0.7294] max:[0.9571]
% midpoint diff: mse = -0.2989 CI: [-0.4403 to -0.111] / pval = 0

hm = [...
    .1657, .5015, .7083;...
    -13.25, -10.8, -8.296...
    ];




nexttile; hold on;
yyaxis('left')
errorbar(.5, hm(1,2), abs(hm(1,2)-hm(1,1)), abs(hm(1,2)-hm(1,3)),...
    "ok", 'MarkerSize', 10, 'LineWidth',1.5, 'MarkerFaceColor', 'k')

ylim([-1, 1])
yl1=ylim;

yyaxis('right')
errorbar(1.5, hm(2,2), abs(hm(2,2)-hm(2,1)), abs(hm(2,2)-hm(2,3)),...
    "ok", 'MarkerSize', 10, 'LineWidth',1.5, 'MarkerFaceColor', 'k')

yl2=ylim*1.1;

ratio = yl2(1)/yl1(1);
ylim([yl2(1), yl1(2)*ratio])

xlim([0,2])
yline(0)
% ylim([-1,1])
set(gca, 'TickDir', 'out', 'LineWidth', 1)
title('midpoint centrality')



