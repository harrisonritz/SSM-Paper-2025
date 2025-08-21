%% format Arnau 2023

clear; clc;


ROOT = ''
FIELDTRIP_ROOT=''
load_name = ''
save_name = ''

plot_spectra= false;
do_block_baseline=0

% FILTER DATA
do_filt = true
filt_range = [0, 30]

% RESAMPLE DATA
do_resample = true
new_srate = 125


% epoch(ts <= 0) = 1;                  % ITI = 1
% epoch(ts > 0 & ts <= .200) = 2;      % cue = 2
% epoch(ts > .200 & ts <= .800) = 3;   % ISI = 3
% epoch(ts > .800) = 4;                % trial = 4
epoch_range = [0.0, 1.0]

if do_resample==1
    assert(new_srate > 3*filt_range(2), 'srate must be > 3x LPF')
end


% rename
if do_resample
        save_name = sprintf('%s_srate@%g',save_name, new_srate);
end

if do_filt
    save_name = sprintf('%s_filt@%g-%g',save_name, filt_range(1), filt_range(2));
end

fprintf('\nFILENAME: %s\n\n',save_name)



load_dir = fullfile(ROOT, 'data/Arnau_2023_EEG/autocleaned', load_name);
mkdir(sprintf('%s/data/NeSS-formatted/%s', ROOT, save_name));

addpath(genpath('../utils/'))
addpath(FIELDTRIP_ROOT)


exclude_pts = []
pts = setxor([8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34],exclude_pts)

npt = length(pts)



%%

f=figure;

pt_count = 1;

for pp = 1:npt

    fprintf("===============  pt%d  =============== \n", pt_count)

    fn = sprintf('VP%02d_%s.set', pts(pp), load_name);
    try
        EEG = pop_loadset(fn, load_dir);
    catch ME
        fprintf('\n\n ============================ pt %d / VP%d failed! ============================\n\n\n\n', pt_count, pts(pp))
        continue;
    end


    % filter data
    chan_sel = ~ismember(1:size(EEG.data,1), EEG.chans_rejected);
    % time_sel = (EEG.times >= 0) & (EEG.times <= 800); % =================== EPOCH
    time_sel = find((EEG.times >= -1000) & (EEG.times <= 1500)); % =================== EPOCH
    chanLocs    = EEG.chanlocs;
    srate = EEG.srate;


    % trial_sel =   ...
    %     EEG.trialinfo(:, 2) > 4 &... after run 4
    %     EEG.trialinfo(:, 23) > 1 &... not first position
    %     EEG.trialinfo(:, 12) == 1 &... correct previous trial
    %     EEG.trialinfo(:, 18) == 1 &... correct current trial
    %     EEG.trialinfo(:, 15)  > 200 &... correct RT > 200
    %     EEG.trialinfo(:, 15)  < 2000;... correct RT < 2000

    %    1  pt num
    %    2  block_nr,...
    %    3  trial_nr,...
    %    4  bonustrial,...
    %    5  tilt_task,...
    %    6  cue_ax,...
    %    7  target_red_left,...
    %    8  distractor_red_left,...
    %    9  response_interference,...
    %    10 task_switch,...
    %    11 prev_switch,...
    %    12 prev_accuracy,...
    %    13 correct_response,...
    %    14 response_side,...
    %    15 rt,...
    %    16 prevRT,...
    %    17 rt_thresh_color,...
    %    18 rt_thresh_tilt,...
    %    19 accuracy,...
    %    20 position_color,...
    %    21 position_tilt,...
    %    22 position_target,...
    %    23 position_distractor,...
    %    24 sequence_position,...


    trial_sel =   ...
        EEG.trialinfo(:, 2) > 4 &... after run 4
        EEG.trialinfo(:, 24) ~= 1 &... not first position
        (EEG.trialinfo(:, 15) > 200) & ... RT > 200ms
        isfinite(EEG.trialinfo(:, 15)) & ... made response
        isfinite(EEG.trialinfo(:, 16)); % made response on previous trial

    fprintf('keeping %.4g pct of trials (n=%d) ...', mean(trial_sel)*100, sum(trial_sel))

    % y: data
    y = EEG.data(...
        chan_sel,...
        time_sel,...
        trial_sel...
        );
  
    % FILTER ================================
    if do_filt

        fprintf("\nfilt %g-%gHz\n", filt_range(1), filt_range(2))

        for tl = 1:size(y,3)

            if filt_range(1) == 0
                y(:,:,tl) = ft_preproc_lowpassfilter(y(:,:,tl), srate, filt_range(2));
            else
                y(:,:,tl) = ft_preproc_lowpassfilter(y(:,:,tl), srate, filt_range(2));
                y(:,:,tl) = ft_preproc_highpassfilter(y(:,:,tl), srate, filt_range(1),[],[],[], 'split');
            end

        end

        if plot_spectra

            [spectra,freqs] =spectopo(y, size(y,2), EEG.srate,...
                'plot', 'off', 'verbose', 'off');

            plot(freqs(1:60), mean(spectra(:,1:60)), '-k', 'LineWidth',2)

        end

    end
    % ================================================================

    % RESAMPLE ================================
    if do_resample

        fprintf("resample %gHz\n", new_srate)


        [y_test, tim, ~] = ft_preproc_resample(y(:,:,1), srate, new_srate, 'downsample');
        y_new = nan(size(y,1), size(y_test,2), size(y,3));
        for tl = 1:size(y,3)
            [y_new(:,:,tl), tim, ~] = ft_preproc_resample(y(:,:,tl), srate, new_srate, 'downsample');
        end

        srate = new_srate;
        y = y_new;
        time_sel = time_sel(tim);
    end
    % ================================================================


    % Block Baseline ================================
    if do_block_baseline==1

        fprintf("\nblock baseline %d-%d\n", base_time(1), base_time(2))

        base_sel = (EEG.times > base_time(1)) & (EEG.times<base_time(2));

        seq_pos = EEG.trialinfo(:,24);
        rew = EEG.trialinfo(:,4);
        seq_n = nan(size(EEG.trialinfo,1),1);
        seq_n(1) = 1;
        nn = 1;
        for tr = 2:size(EEG.trialinfo,1)
            if (rew(tr) ~= rew(tr-1)) || (seq_pos(tr) <= seq_pos(tr-1))
                nn=nn+1;
            end
            seq_n(tr) = nn;
        end

        seqsel = seq_n(trial_sel);

        seqs = unique(seqsel);
        for ss = 1:length(seqs)

            % basline from 1st trial in sequence
            baseline =  squeeze(mean(...
                EEG.data(...
                chan_sel,...
                base_sel,seq_n==seqs(ss) & EEG.trialinfo(:,24)==1),...
                2));

            if ~isempty(baseline)
                y(:,:,seqsel==seqs(ss)) = y(:,:,seqsel==seqs(ss)) - baseline;
            end

        end

        
    end
    % ================================================================




    % ts: timepoints, time selectst
    ts = EEG.times(time_sel)/1000;
    dt = 1/srate;

    
    epoch_sel = (ts >= epoch_range(1)) & (ts <= epoch_range(2));
    y = y(:, epoch_sel, :);
    ts = ts(epoch_sel);

    % epoch
    epoch = zeros(size(ts));
    epoch(ts <= 0) = 1;                  % ITI = 1
    epoch(ts > 0 & ts <= .200) = 2;      % cue = 2
    epoch(ts > .200 & ts <= .800) = 3;   % ISI = 3
    epoch(ts > .800) = 4;                % trial = 4


    % task conds
    %    1  pt num
    %    2  block_nr,...
    %    3  trial_nr,...
    %    4  bonustrial,...
    %    5  tilt_task,...
    %    6  cue_ax,...
    %    7  target_red_left,...
    %    8  distractor_red_left,...
    %    9  response_interference,...
    %    10 task_switch,...
    %    11 prev_switch,...
    %    12 prev_accuracy,...
    %    13 correct_response,...
    %    14 response_side,...
    %    15 rt,...
    %    16 prevRT,...
    %    17 rt_thresh_color,...
    %    18 rt_thresh_tilt,...
    %    19 accuracy,...
    %    20 position_color,...
    %    21 position_tilt,...
    %    22 position_target,...
    %    23 position_distractor,...
    %    24 sequence_position,... 



    isSwitch    = EEG.trialinfo(trial_sel, 10); isSwitch(isSwitch==0)=-1;
    
    isTilt      = EEG.trialinfo(trial_sel, 5);  isTilt(isTilt==0)=-1;
    task        = isTilt;
    wasTilt     = -isSwitch .* isTilt;
    block       = EEG.trialinfo(trial_sel, 2);
    resp        = EEG.trialinfo(trial_sel, 14);
    rt          = EEG.trialinfo(trial_sel, 15)/1000;
    prevRT      = EEG.trialinfo(trial_sel, 16)/1000;
    logrt       = log(rt);
    acc         = EEG.trialinfo(trial_sel, 19);
    prevAcc     = EEG.trialinfo(trial_sel, 12);

    isRew       = EEG.trialinfo(trial_sel, 4);

    cue_ax      = EEG.trialinfo(trial_sel, 6); cue_ax(cue_ax==0)=-1;
    cueTilt     = cue_ax .* (task==1);
    cueColor    = cue_ax .* (task==-1);
    
    cueRepeat   = EEG.trialinfo(trial_sel, 25).*(isSwitch==-1); 
    cueRepeat(isSwitch ==-1 & cueRepeat==0) = -1;

    targ_redLeft = EEG.trialinfo(trial_sel, 7);
    dist_redLeft = EEG.trialinfo(trial_sel, 8);

    colorRed = targ_redLeft;
    colorRed(task==1) = dist_redLeft(task==1);
    colorRed(colorRed==0) = -1;
   
    tiltLeft = targ_redLeft;
    tiltLeft(task==-1) = dist_redLeft(task==-1);
    tiltLeft(tiltLeft==0) = -1;


    % make structure
    trial=struct;

    trial.block = block;

    trial.task = isTilt;
    trial.prevTask = wasTilt;
    trial.switch = isSwitch;
    trial.taskSwitch = isTilt.*(isSwitch==1);
    trial.taskRepeat = isTilt.*(isSwitch==-1);

    trial.cueTilt = cueTilt;
    trial.cueColor = cueColor;
    trial.cueRepeat = cueRepeat;

    trial.rew = isRew;

    trial.RT = rt;
    trial.prevRT = prevRT;
    trial.acc = acc ==1;
    trial.prevAcc = prevAcc ==1;
    trial.resp = resp;

    trial.color = colorRed;
    trial.tilt = tiltLeft;



    % save
    fprintf("\nsaving... \n\n")
    save(sprintf('%s/data/NeSS-formatted/%s/%s_%d.mat', ROOT, save_name, save_name, pt_count),...
        'y',...
        'dt',...
        'ts',... 
        'epoch',... 
        'trial',...
        'chanLocs');

    pt_count = pt_count+1;



    % plot
    nexttile; hold on;
    plot(mean(y,3)', 'LineWidth',1)
    title(pts(pp));
    set(gca, 'TickDir', 'out', 'LineWidth', 1);



end

f.Position = [100,100,1900,1000];

saveas(f,sprintf('%s/data/NeSS-formatted/%s/%s.png', ROOT, save_name, save_name))
saveas(f,sprintf('%s/data/NeSS-formatted/%s/%s', ROOT, save_name, save_name), 'epsc')




