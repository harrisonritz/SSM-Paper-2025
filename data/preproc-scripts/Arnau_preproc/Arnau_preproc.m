function Arnau_preproc(which_subj)


% parameters
save_name = 'Arnau2024__srate-125__firws-p01-30__ic-brain50__ts-0-1000'


do_hpf = 1
use_firws = 1

do_minphase = 0
do_lin_causal = 0
assert(do_minphase*do_lin_causal == 0)



ERP_srate = 125
ERP_fcutoff = [.01, 30]

EEG_srate = 125
EEG_fcutoff = [1, 30]


epoch_time = [0, 1]
rm_base = 0



ica_reject = [...
    0  .50;...  NaN NaN
    NaN NaN;... 0.8 1
    NaN NaN;... 0.8 1
    NaN NaN;...
    NaN NaN;...
    NaN NaN;...
    NaN NaN];




%% PATH VARS
if ismac ==1
    PATH_ROOT       = '/Users/hr0283/Brown Dropbox/Harrison Ritz/HallM_NeSS/data'
    PATH_SCRATCH    = PATH_ROOT
else

    PATH_ROOT       = '/home/hr0283/HallM_NeSS/data/'
    PATH_SCRATCH    = '/scratch/gpfs/hr0283/HallM_NeSS/data'
end

PATH_EEGLAB        = fullfile(PATH_ROOT, '/utils/eeglab2024_2/')
PATH_ERPLAB        = fullfile(PATH_ROOT, '/utils/erplab-master/')
PATH_BVIO          = fullfile(PATH_ROOT, '/utils/bva-io/')

PATH_LOGFILES      = fullfile(PATH_SCRATCH, '/Arnau_2023_EEG/logfiles/')
PATH_RAW           = fullfile(PATH_SCRATCH, '/Arnau_2023_EEG/bocotilt_raw/')


% save paths
PATH_ICSET         = fullfile(PATH_SCRATCH, sprintf('/Arnau_2023_EEG/ica_set/%s/', save_name))
mkdir(PATH_ICSET)

PATH_AUTOCLEANED   = fullfile(PATH_SCRATCH, sprintf('/Arnau_2023_EEG/autocleaned/%s/', save_name))
mkdir(PATH_AUTOCLEANED)


%% Subjects
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
    'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
    'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};


try
    disp(which_subj)
    subj_idx = str2double(which_subj)
catch
    subj_idx = 1:length(subject_list)
end





%% Init eeglab
addpath(PATH_EEGLAB);
addpath(PATH_BVIO)
addpath(genpath(PATH_ERPLAB))

eeglab;
channel_location_file = which('dipplot.m');
channel_location_file = channel_location_file(1 : end - length('dipplot.m'));
channel_location_file = [channel_location_file, 'standard_BESA/standard-10-5-cap385.elp'];



%% Iterate subjects
tic_tot = tic;
all_channel_labels = cell(0)
for s = subj_idx

    tic_pt=tic;

    % participant identifiers
    subject = subject_list{s};
    id = str2num(subject(3 : 4));
    fprintf('\n\n\n\n\n-----------SUBJECT %d----------------------\n\n', id);



    % fprintf('\nload set: %d', load_set)
    % fprintf('\nset exists: %d\n', exist(fullfile(PATH_ICSET, [save_name, '_', subject, '_icset.set']), 'file'))

    % if load_set && exist(fullfile(PATH_ICSET, [save_name, '_', subject, '_icset.set']), 'file')
    %
    %     fprintf('\n\n\n-----------LOADING EEG DATA-----------\n\n\n')
    %
    %     ERP = pop_loadset('filename', [save_name, '_', subject, '_icset_erp.set'], 'filepath', PATH_ICSET);
    %     EEG = pop_loadset('filename', [save_name, '_', subject, '_icset.set'], 'filepath', PATH_ICSET);
    %
    % else

    fprintf('\n\n\n-----------FITTING EEG DATA-----------\n\n\n')


    % Load
    EEG = pop_loadbv(PATH_RAW, [subject, '.vhdr'], [], []);

    % Repair subject 13 (first block start marker missing)
    if id == 13
        EEG = pop_editeventvals(EEG, 'insert',...
            {1, [], [], [], [], [], [], [], [], []},...
            'changefield', {1, 'latency', 0.5},...
            'changefield', {1, 'duration', 0.001},...
            'changefield', {1, 'channel', 0},...
            'changefield', {1, 'bvtime', []},...
            'changefield', {1, 'visible', []},...
            'changefield', {1, 'bvmknum', 3733},...
            'changefield', {1, 'type', 'S121'},...
            'changefield', {1, 'code', 'Stimulus'});
    end

    % Subject 30 has been restarted after a couple of trials.
    % Remove all events until second start of block 1 (second occurence of 'S121'),
    % which is event number 36...
    if id == 30
        EEG.event(1 : 35) = [];
    end

    % Fork response button channels
    RESPS = pop_select(EEG, 'channel', [65, 66]);
    EEG = pop_select(EEG, 'nochannel', [65, 66]);

    % Open log file
    fid = fopen([PATH_LOGFILES, subject, '_degreeLog.txt'], 'r');

    % Extract lines as strings
    logcell = {};
    tline = fgetl(fid);
    while ischar(tline)
        logcell{end + 1} = tline;
        tline = fgetl(fid);
    end

    % Delete header
    logcell(1 : 3) = [];

    % Get color and tilt positions in probe display (numbers 1-8)
    positions = [];
    for l = 1 : length(logcell)
        line_values = split(logcell{l}, ' ');
        positions(l, 1) = str2num(line_values{8});
        positions(l, 2) = str2num(line_values{10});
    end

    % Open trial log file
    fid = fopen([PATH_LOGFILES, subject, '_trials.txt'], 'r');

    % Extract lines as strings
    logcell = {};
    tline = fgetl(fid);
    while ischar(tline)
        logcell{end + 1} = tline;
        tline = fgetl(fid);
    end

    % Delete header
    logcell(1 : 3) = [];

    % Get response side, accuracy and rt from log file
    trial_log = [];
    for l = 1 : length(logcell)
        line_values = split(logcell{l}, '|');
        trial_log(l, 1) = str2num(line_values{5});
        trial_log(l, 2) = str2num(line_values{6});
        trial_log(l, 3) = str2num(line_values{7});
    end

    % Get version of task
    if id < 8
        error("Preprocessing invalid for id < 8.");
    elseif id == 8
        EEG.task_version = 1;
    else
        EEG.task_version = mod(id, 8);
        if EEG.task_version == 0
            EEG.task_version = 8;
        end
    end

    % Open log file
    fid = fopen([PATH_LOGFILES, subject, '_degreeLog.txt'], 'r');

    % Extract lines as strings
    logcell = {};
    tline = fgetl(fid);
    while ischar(tline)
        logcell{end + 1} = tline;
        tline = fgetl(fid);
    end

    % Delete header
    logcell(1 : 3) = [];

    % Iterate last 100 trials and extract rt thresholds
    rt_threshs = [];
    for l = 1 : 100
        line_values = split(logcell{length(logcell) - l}, ' ');
        rt_threshs(l, 1) = str2num(line_values{5});
        rt_threshs(l, 2) = str2num(line_values{13});
    end
    rt_thresh_color = mean(rt_threshs(rt_threshs(:, 1) == 2, 2));
    rt_thresh_tilt = mean(rt_threshs(rt_threshs(:, 1) == 1, 2));

    % Event coding
    EEG = event_coding(EEG, RESPS, positions, trial_log, rt_thresh_color, rt_thresh_tilt);

    % Add FCz as empty channel
    EEG.data(end + 1, :) = 0;
    EEG.nbchan = size(EEG.data, 1);
    EEG.chanlocs(end + 1).labels = 'FCz';

    % Add channel locations
    EEG = pop_chanedit(EEG, 'lookup', channel_location_file);

    % Save original channel locations (for later interpolation)
    EEG.chanlocs_original = EEG.chanlocs;

    % Reref to CPz, so that FCz obtains non-interpolated data
    fprintf('\n\n\n---pop_reref--------------\n')
    EEG = pop_reref(EEG, 'CPz');

    % Remove data at boundaries
    EEG = pop_rmdat(EEG, {'boundary'}, [0, 1], 1);

    % Resample data
    fprintf('\n\n\n---pop_resample--------------\n')
    ERP = pop_resample(EEG, ERP_srate); % 500 --> 125
    EEG = pop_resample(EEG, EEG_srate); % 200 --> 200



    % Reject continuous data
    fprintf('\n\n\n---pop_rejcont--------------\n')
    [ERP, selected_regions] = pop_rejcont(ERP, 'freqlimit', [20, 40], 'taper', 'hamming');
    ERP.rejcont_regions = selected_regions;
    [EEG, selected_regions] = pop_rejcont(EEG, 'freqlimit', [20, 40], 'taper', 'hamming');
    EEG.rejcont_regions = selected_regions;

    % Filter
    % fprintf('\n\n\n---pop_basicfilter--------------\n')
    % ERP = pop_basicfilter(ERP, [1 : ERP.nbchan], 'Cutoff', [0.01, 40], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 4, 'RemoveDC', 'on', 'Boundary', 'boundary');
    % EEG = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [   1, 40], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 4, 'RemoveDC', 'on', 'Boundary', 'boundary');


    if use_firws == 1

        fprintf('\n\n\n---pop_firws--------------\n')


        % ERP (OUTPUT) FILTER ===================
        fprintf('\nERP filter\n')


        % high-pass filter
        if do_hpf==1

            [ERP_filt,~,b] = pop_firws(ERP,...
                'fcutoff', ERP_fcutoff(1), ...
                'forder', pop_firwsord('blackman', EEG.srate, 2*ERP_fcutoff(1)), ...
                'ftype', 'highpass', ...
                'wtype', 'blackman',...
                'minphase', do_minphase);

            if do_lin_causal==1
                fprintf('\ncausal (linear)\n')
                ERP.data = fir_filterdcpadded(2*b, 1, ERP.data', 1)';
            else
                ERP = ERP_filt;
            end
            clear ERP_filt

        end


        % Low-pass filter
        forder_lpf = round(pop_firwsord('blackman', EEG.srate, .25*ERP_fcutoff(2)));
        forder_lpf = forder_lpf + mod(forder_lpf,2);

        ERP = pop_firws(ERP,...
            'fcutoff', ERP_fcutoff(2), ...
            'forder', forder_lpf, ...
            'ftype', 'lowpass', ...
            'wtype', 'blackman');

        % =================================






        % EEG (ICA) FILTER ===================
        fprintf('\nEEG filter\n')


        % high-pass filter
        EEG = pop_firws(EEG,...
            'fcutoff', EEG_fcutoff(1), ...
            'forder', pop_firwsord('blackman', EEG.srate, 2*EEG_fcutoff(1)), ...
            'ftype', 'highpass', ...
            'wtype', 'blackman');


        % Low-pass filter
        forder_lpf = round(pop_firwsord('blackman', EEG.srate, .25*EEG_fcutoff(2)));
        forder_lpf = forder_lpf + mod(forder_lpf,2);

        EEG = pop_firws(EEG,...
            'fcutoff', EEG_fcutoff(2), ...
            'forder', forder_lpf, ...
            'ftype', 'lowpass', ...
            'wtype', 'blackman');

        % =================================




    else

        fprintf('\n\n\n---pop_eegfiltnew--------------\n')

        if do_hpf==1
            ERP = pop_eegfiltnew(ERP, 'locutoff', ERP_fcutoff(1), 'hicutoff', []);
        end
        ERP = pop_eegfiltnew(ERP, 'locutoff', [], 'hicutoff', ERP_fcutoff(2));

        if do_hpf==1
            EEG = pop_eegfiltnew(EEG, 'locutoff', EEG_fcutoff(1), 'hicutoff', []);
        end
        EEG = pop_eegfiltnew(EEG, 'locutoff', [], 'hicutoff', EEG_fcutoff(2));

    end


    % ERP = pop_firws(ERP, 'fcutoff', ERP_fcutoff, 'forder', 5*(ERP.srate/ERP_fcutoff(1)), 'ftype', 'bandpass','wtype', 'hamming');
    % EEG = pop_firws(EEG, 'fcutoff', EEG_fcutoff, 'forder', 5*(EEG.srate/EEG_fcutoff(1)), 'ftype', 'bandpass','wtype', 'hamming');



    % Bad channel detection
    fprintf('\n\n\n---pop_rejchan--------------\n')
    [ERP, ERP.chans_rejected] = pop_rejchan(ERP, 'elec', [1 : ERP.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');
    [EEG, EEG.chans_rejected] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');

    % Interpolate channels
    fprintf('\n\n\n---pop_interp--------------\n')
    ERP = pop_interp(ERP, ERP.chanlocs_original, 'spherical');
    EEG = pop_interp(EEG, EEG.chanlocs_original, 'spherical');

    % Reref common average
    ERP = pop_reref(ERP, []);
    EEG = pop_reref(EEG, []);

    % Determine rank of data
    dataRank = sum(eig(cov(double(EEG.data'))) > 1e-6)
    dataRank_ERP  = sum(eig(cov(double(ERP.data'))) > 1e-6)

    % Epoch data
    fprintf('\n\n\n---pop_epoch--------------\n')
    ERP = pop_epoch(ERP, {'trial'}, epoch_time, 'newname', [subject '_epoched'], 'epochinfo', 'yes');
    EEG = pop_epoch(EEG, {'trial'}, epoch_time, 'newname', [subject '_epoched'], 'epochinfo', 'yes');

    if rm_base==1
        fprintf('\n\n\n---pop_rmbase--------------\n')
        ERP = pop_rmbase(ERP, [-200, 0]);
        EEG = pop_rmbase(EEG, [-200, 0]);
    else
        fprintf('\n\n\n---no baseline--------------\n')
    end


    % HR: added previous RT & previous cue
    fprintf('\n\n\n---adding prevRT--------------\n')
    lats = [];
    for e = 1 : length(ERP.event)
        lats(end+1) = mod(ERP.event(e).latency, ERP.pnts);
    end
    lat_mode = mode(lats);

    trialinfo = [];
    RT = 0;
    CUE = 0;
    trial_nr = 0;
    for e = 1 : length(ERP.event)
        if strcmpi(ERP.event(e).type, 'trial') && (mod(ERP.event(e).latency, ERP.pnts) == lat_mode)

            CUE = [CUE; ERP.event(e).cue_ax];
            RT = [RT; ERP.event(e).rt];
            trial_nr = [trial_nr; ERP.event(e).trial_nr];

        end
    end



    % Autoreject trials
    fprintf('\n\n\n---pop_autorej--------------\n')
    [ERP, ERP.rejected_epochs] = pop_autorej(ERP, 'nogui', 'on');
    [EEG, EEG.rejected_epochs] = pop_autorej(EEG, 'nogui', 'on');



    % Find standard latency of event in epoch
    lats = [];
    for e = 1 : length(ERP.event)
        lats(end+1) = mod(ERP.event(e).latency, ERP.pnts);
    end
    lat_mode = mode(lats);

    % Compile a trialinfo matrix
    trialinfo = [];
    counter = 0;
    for e = 1 : length(ERP.event)
        if strcmpi(ERP.event(e).type, 'trial') & (mod(ERP.event(e).latency, ERP.pnts) == lat_mode)

            counter = counter + 1;

            if any((ERP.event(e).trial_nr-1) == trial_nr)
                prevRT = RT((ERP.event(e).trial_nr-1) == trial_nr);
                prevCue = CUE((ERP.event(e).trial_nr-1) == trial_nr);
            else
                prevRT = nan;
                prevCue = nan;
            end


            % Compile table
            trialinfo(counter, :) = [id,...
                ERP.event(e).block_nr,...
                ERP.event(e).trial_nr,...
                ERP.event(e).bonustrial,...
                ERP.event(e).tilt_task,...
                ERP.event(e).cue_ax,...
                ERP.event(e).target_red_left,...
                ERP.event(e).distractor_red_left,...
                ERP.event(e).response_interference,...
                ERP.event(e).task_switch,...
                ERP.event(e).prev_switch,...
                ERP.event(e).prev_accuracy,...
                ERP.event(e).correct_response,...
                ERP.event(e).response_side,...
                ERP.event(e).rt,...
                prevRT,...% previous RT
                ERP.event(e).rt_thresh_color,...
                ERP.event(e).rt_thresh_tilt,...
                ERP.event(e).accuracy,...
                ERP.event(e).position_color,...
                ERP.event(e).position_tilt,...
                ERP.event(e).position_target,...
                ERP.event(e).position_distractor,...
                ERP.event(e).sequence_position,...
                prevCue,...% previous cue_ax
                ];

        end
    end

    % Save trialinfo
    ERP.trialinfo = trialinfo;
    writematrix(trialinfo, [PATH_AUTOCLEANED, subject, '_trialinfo_erp.csv']);

    % Find standard latency of event in epoch
    lats = [];
    for e = 1 : length(EEG.event)
        lats(end+1) = mod(EEG.event(e).latency, EEG.pnts);
    end
    lat_mode = mode(lats);


    % Compile a trialinfo matrix
    trialinfo = [];
    counter = 0;
    for e = 1 : length(EEG.event)
        if strcmpi(EEG.event(e).type, 'trial') & (mod(EEG.event(e).latency, EEG.pnts) == lat_mode)

            counter = counter + 1;

            if any((EEG.event(e).trial_nr-1) == trial_nr)
                prevRT = RT((EEG.event(e).trial_nr-1) == trial_nr);
                prevCue = CUE((EEG.event(e).trial_nr-1) == trial_nr);
            else
                prevRT = nan;
                prevCue = nan;
            end

            % Compile table
            trialinfo(counter, :) = [id,...
                EEG.event(e).block_nr,...
                EEG.event(e).trial_nr,...
                EEG.event(e).bonustrial,...
                EEG.event(e).tilt_task,...
                EEG.event(e).cue_ax,...
                EEG.event(e).target_red_left,...
                EEG.event(e).distractor_red_left,...
                EEG.event(e).response_interference,...
                EEG.event(e).task_switch,...
                EEG.event(e).prev_switch,...
                EEG.event(e).prev_accuracy,...
                EEG.event(e).correct_response,...
                EEG.event(e).response_side,...
                EEG.event(e).rt,...
                prevRT,...% previous RT
                EEG.event(e).rt_thresh_color,...
                EEG.event(e).rt_thresh_tilt,...
                EEG.event(e).accuracy,...
                EEG.event(e).position_color,...
                EEG.event(e).position_tilt,...
                EEG.event(e).position_target,...
                EEG.event(e).position_distractor,...
                EEG.event(e).sequence_position,...
                prevCue,... % previous cue_ax
                ];

        end
    end

    % Save trialinfo
    EEG.trialinfo = trialinfo;
    writematrix(trialinfo, [PATH_AUTOCLEANED, subject, '_trialinfo.csv']);



    % Runica & ICLabel
    fprintf('\n\n\n---pop_runica--------------\n')

    EEG = pop_runica(EEG, 'extended', 1, 'interrupt', 'on', 'PCA', dataRank);
    EEG = iclabel(EEG, 'default');


    % Find nobrainer === HR UPDATED IC LABELLING
    % EEG.nobrainer = find(EEG.etc.ic_classification.ICLabel.classifications(:, 1) < 0.3 | EEG.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);
    EEG = pop_icflag(EEG, ica_reject);

    % Copy ICs to erpset
    ERP = pop_editset(ERP, 'icachansind', 'EEG.icachansind', 'icaweights', 'EEG.icaweights', 'icasphere', 'EEG.icasphere');
    ERP.etc = EEG.etc;
    [ERP.nobrainer, EEG.nobrainer] = deal(find(EEG.reject.gcompreject));
    ERP.reject.gcompreject = EEG.reject.gcompreject;

    try

        disp('transferring parameters')
        ERP.icachansind = EEG.icachansind;
        ERP.icaweights = EEG.icaweights;
        ERP.icasphere = EEG.icasphere;
        ERP.icawinv = EEG.icawinv;

        disp('eeg_checkset')
        ERP = eeg_checkset(ERP, 'ica');

    catch
        fprintf('\n\n~~~~~~ cannot manually set ERP weights ~~~~~~\n\n')
    end


    % Save IC set
    pop_saveset(ERP, 'filename', [save_name, '_', subject, '_icset_erp.set'], 'filepath', PATH_ICSET, 'check', 'on');
    pop_saveset(EEG, 'filename', [save_name, '_', subject, '_icset.set'], 'filepath', PATH_ICSET, 'check', 'on');

    % end


    % PRINT
    fprintf('\n\n\n---pop_subcomp--------------\n')

    disp('ERP component classification')
    disp(ERP.etc.ic_classification.ICLabel.classifications)
    disp('ERP reject gcompreject')
    disp(ERP.reject.gcompreject)


    % ERP matrix sizes
    erp_keep = setdiff_bc(1:size(ERP.icaweights,1), find(ERP.reject.gcompreject == 1));

    disp('ERP reject gcompreject SIZE')
    disp(size(ERP.reject.gcompreject))
    disp('ERP.nobrainer SIZE')
    disp(size(ERP.nobrainer))
    disp('ERP.icaweights SIZE')
    disp(size(ERP.icaweights))
    disp('ERP.icawinv SIZE')
    disp(size(ERP.icawinv))
    disp('ERP.icawinv sel SIZE')
    disp(size(ERP.icawinv(:,erp_keep)))

    erp_dat = eeg_getdatact(ERP, 'component', erp_keep, 'reshape', '2d');
    disp('ERG dat SIZE')
    disp(size(erp_dat))
    clear erp_dat



    % EEG matrix sizes
    eeg_keep = setdiff_bc(1:size(EEG.icaweights,1), find(EEG.reject.gcompreject == 1));

    disp('EEG.icaweights')
    disp(size(EEG.icaweights))
    disp('EEG.icachansind')
    disp(size(EEG.icachansind))
    disp('EEG.icasphere')
    disp(size(EEG.icasphere))
    disp('EEG.icawinv SIZE')
    disp(size(EEG.icawinv))
    disp('EEG.icawinv sel SIZE')
    disp(size(EEG.icawinv(:,eeg_keep)))

    eeg_dat = eeg_getdatact(EEG, 'component', eeg_keep, 'reshape', '2d');
    disp('EEG dat SIZE')
    disp(size(eeg_dat))
    clear eeg_dat





    % REMOVE COMPONENTS

    ERP = pop_subcomp(ERP, [], 0, 0); %remove bad components
    % EEG = pop_subcomp(EEG, [], 0, 0); %remove bad components


    % Save clean data
    fprintf('\n\n\n---pop_saveset--------------\n')
    pop_saveset(ERP, 'filename', sprintf('%s_%s.set', subject, save_name), 'filepath', PATH_AUTOCLEANED, 'check', 'on');


    % Save channel label in order for creating 10-20 montage in mne
    channel_labels = '';
    for ch = 1 : EEG.nbchan
        channel_labels = [channel_labels, ' ', EEG.chanlocs(ch).labels];
    end
    save([PATH_AUTOCLEANED, sprintf('channel_labels_%s.mat', subject)], 'channel_labels');


    fprintf('\n\n\n--- Finished Participant %s after %.3g min (%.3g min total) ---\n', subject, toc(tic_pt)/60, toc(tic_tot)/60)



end




end
