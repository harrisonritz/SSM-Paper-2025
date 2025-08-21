%% extract average data

clear
clc

ROOT = ''
saveName = '';

% FILTER DATA
do_filt = true
filt_range = [0, 30]

% RESAMPLE DATA
do_resample = true
new_srate = 125

% time == 1.2: cue onset
% time == 1.4: ISI onset
% time == 1.8: trial onset
tmin = 1.2
tmax = 2.0


% rename
if do_resample
        saveName = sprintf('%s_srate@%g',saveName, new_srate);
end

if do_filt
    saveName = sprintf('%s_filt@%g-%g',saveName, filt_range(1), filt_range(2));
end

fprintf('\nFILENAME: %s\n\n',saveName)



base_method = "None" % subtract, regress, none

addpath('/Users/hr0283/Documents/MATLAB/spm/external/fieldtrip/preproc')
addpath(genpath(fullfile(ROOT, 'toolbox')))
mkdir(sprintf('%s/data/NeSS-formatted/%s', ROOT, saveName));


ptList = [01 02 03 04 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 30 31 32 33];
npt   = length(ptList);

datapath = fullfile(ROOT, 'data', 'HallMcMaster_2019_EEG', 'eeg-processed');
datapath_behav = fullfile(ROOT, 'data', 'HallMcMaster_2019_EEG', 'behavior', 'experiment_matrix_format');
datapath_triggers = fullfile(ROOT, 'data', 'HallMcMaster_2019_EEG', 'triggers');






% functions
selz = @(x, sel) ((x-nanmean(x(logical(sel)))) ./ nanstd(x(logical(sel)))).*sel;





%% extract average

f=figure;

parfor pp = 1:npt

    tic
    fprintf("===============  pt%d  =============== \n", pp)

    %% load

    %load sub EEG file
    filename = sprintf('%s/TaskRep_%02d_EEG_trials.mat', datapath, ptList(pp));
    if ~exist(filename,'file')
        error(sprintf('%s does not exist!',filename));
    end
    f=load(filename);
    eeg = f.eeg;

    %load sub behavioural file
    filename_behav = sprintf('%s/%01d_behav.mat', datapath_behav, ptList(pp));
    if ~exist(filename_behav,'file')
        error(sprintf('%s does not exist!',filename_behav));
    end
    f=load(filename_behav);
    data_matrix=f.data_matrix;

    %load list of rejected EEG trials
    filename_rej = sprintf('%s/TaskRep_%02d_rejectedTrials.mat',datapath,ptList(pp));
    if ~exist(filename_rej,'file')
        error(sprintf('%s does not exist!',filename_rej));
    end
    f=load(filename_rej);
    rejectedTrialz = f.rejectedTrialz;

    %load trial triggers
    filename_trig = sprintf('%s/ConditionLabels_S%02d',datapath_triggers,pp);
    if ~exist(filename_rej,'file')
        error(sprintf('%s does not exist!',filename_trig));
    end
    f=load(filename_trig);
    conditionmatrix=f.conditionmatrix;


    % filter trials
    bl = data_matrix(:,1);
    unq_block = unique(data_matrix(:,1));

    prevTask = [nan;data_matrix(1:end-1,3)];
    prevRT = [nan;data_matrix(1:end-1,6)];
    prevAcc = [nan;data_matrix(1:end-1,7)];

    GoodTrials = ~rejectedTrialz;
    
    bindx = GoodTrials' & ... % not rejected during EEG preprocessing (GoodTrials')
        data_matrix(:,6) > .200 &... % RT > 200ms
        isfinite(data_matrix(:,6)) & ... % current response (finite RT)
        isfinite(prevRT) & ... % previous response (finite RT)
        [1; abs(diff(data_matrix(:,1)))] == 0; % , no first trials

    eindx   = bindx(logical(GoodTrials')) == 1;  % eeg index
    eegdat      = eeg.data(:,:,eindx);
    ntrials = size(eegdat,3);
    timelist  = eeg.timepoints./1000;

    trigidx     = conditionmatrix(eindx, :);

    behav = [];
    behav(:,:) = data_matrix(bindx,:);

    y = eegdat;
    srate = 1000/median(diff(eeg.timepoints));
   


    % FILTER ================================
    if do_filt

        fprintf("filt %g-%gHz\n", filt_range(1), filt_range(2))

        for tl = 1:size(y,3)

            if filt_range(1) == 0
                y(:,:,tl) = ft_preproc_lowpassfilter(y(:,:,tl), srate, filt_range(2));
            else
               
                y(:,:,tl) = ft_preproc_lowpassfilter(y(:,:,tl), srate, filt_range(2));
                y(:,:,tl) = ft_preproc_highpassfilter(y(:,:,tl), srate, filt_range(1),[],[],[], 'split');



            end

        end


        % if plot_spectra
        % 
        %     figure;hold on;
        %     [spectra,freqs] =spectopo(y, size(y,2), srate,...
        %         'plot', 'off', 'verbose', 'off');
        % 
        %     plot(freqs(1:50), mean(spectra(:,1:50)), '-g', 'LineWidth',2)
        % 
        % 
        % end

    end
    % ================================================================

    % RESAMPLE ================================
    if do_resample


        fprintf("resample %g -> %gHz\n",srate, new_srate)


        [y_test, tim, Fnew] = ft_preproc_resample(y(:,:,1), srate, new_srate, 'downsample');
        y_new = nan(size(y,1), size(y_test,2), size(y,3));
        for tl = 1:size(y,3)
            [y_new(:,:,tl), tim, Fnew] = ft_preproc_resample(y(:,:,tl), srate, new_srate, 'downsample');
        end

        y = y_new;
        srate = Fnew;
        timelist = timelist(tim);

    end
    % ================================================================

    dat   = permute(y,[3 1 2]);



    %% baseline data 200-50ms before task cue presentation
    fsample = srate;
    % the next step allows re-epoching by changing the value in elemtimes:
    elemtimes   = [0.0]; % reward cue onset
    nelems      = size(elemtimes,2);
    if size(elemtimes,1) == 1, 
        elemtimes = repmat(elemtimes,[ntrials 1]); 
    end


    %time window for analysis (relative to elemtimes!!)
    dtime = fix([-1.00 +5.000]*fsample);
    %time window for baselining (relative to elemtimes!!)
    dbase = fix([-0.250  -0.050]*fsample);  % Rew  (RELATIVE TO reward cue onset)
    % dbase = fix([+0.950 +1.150]*fsample); % Cue  (RELATIVE TO reward cue onset)
    dstep   = 1;
    %create new time vector based on selected window
    time    = (dtime(1):dstep:dtime(2))/fsample;
    ntimes  = length(time);

    %channels to analyse
    chanlist = 1:61;
    nchans   = length(chanlist);

    %preallocate new data array
    dataerp_elem = nan(ntrials,nchans,ntimes,nelems);
    %re-epoch data and baseline
    for itrial = 1:ntrials
        for ielem = 1:nelems

            ionset = find(timelist <= elemtimes(itrial,ielem),1,'last');
            itime  = ionset+(dtime(1):dstep:dtime(2));
            itime  = min(itime,length(timelist));
            dataerp_elem(itrial,:,:,ielem) = dat(itrial,chanlist,itime);

        end
    end


    % data in range
    time_window = (time >= tmin) & (time <= tmax); % (time >= 1.0) & (time < 2.0); %(time>= 1.2) & (time < 2.8); % ============================================================================================================== TIME WINDOW
    % time == 1.2: cue onset
    % time == 1.4: ISI onset
    % time == 1.8: trial onset

    dat = dataerp_elem(:,chanlist,time_window);
    ts = time(time_window);
    dt = 1/srate;

    % baseline
    izero = find(timelist <= 0,1,'last');
    ibase = izero + (dbase(1):dstep:dbase(2));
    baseRegress = mean(dataerp_elem(:,chanlist,ibase),3);

    % blocks
    %     uqB = unique(behav(:,1));
    %     [blkOff, blkLin] = deal([]);
    %     for bb = 1:length(uqB)
    %         blkOff = [blkOff, (behav(:,1) == uqB(bb))];
    %         blkLin = [blkLin, selz(cumsum(behav(:,1) == uqB(bb)), behav(:,1) == uqB(bb))];
    %     end



    % baseline
    train_set = (behav(:,1) ~= 5) & (behav(:,1) ~= 6);
    for cc = 1:nchans

        mdl = [baseRegress(:,cc)];

        switch base_method

            case "subtract"

                dat(:,cc,:) = dat(:,cc,:) - mdl;

            case 'regress'
                for tt = 1:size(dat,3)
                    B = mdl(train_set)\dat(train_set,cc,tt);
                    dat(:,cc,tt) = dat(:,cc,tt) - (mdl*B);
                end

        end


    end





    %     % prewhiten!
    %     Sw_reg = rsa.stat.covdiag(dii - mdl*bii, ntrials-1-1,'shrinkage',[]);   %%% regularize Sw_hat through optimal shrinkage
    %
    %     % Postmultiply by the inverse square root of the estimated matrix
    %     [V,L]=eig(Sw_reg);
    %     l=diag(L);
    %     sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';










    % savedata ===========
    %     keyboard
    y = permute(dat, [2,3,1]); % [electrodes, timepoints, trials]


    % epochs
    ts = ts-1.2;
    epoch = nan(size(ts));
    epoch(ts <= 0) = 1;            % ITI = 1
    epoch(ts>0 & ts <= .200) = 2;  % cue = 2
    epoch(ts>.200 & ts <= .600) = 3;  % ISI = 3
    epoch(ts>.600) = 4;             % trial = 4


    % trial info
    isSwitch = 1*(behav(:,9) ==1);
    isSwitch(isSwitch==0) = -1;

    isColor = 1*(behav(:,3) < 3);
    isColor(isColor==0) = -1;

    task = isColor;

    wasColor = 1*(((behav(:,9) ~=1) & (behav(:,3) < 3)) | ((behav(:,9) ==1) & (behav(:,3) > 2)));
    wasColor(wasColor==0) = -1;

    isRew = 1*(behav(:,2) == 2);
    isRew(isRew==0) = -1;

    block = behav(:,1);

    resp = behav(:, 5);

    rt = behav(:, 6);
    prev_rt = [mean(behav(1:end-1, 6)); behav(1:end-1, 6)];
  

    acc = behav(:, 7);

    cueColor = ismember(behav(:,3), 1) - ismember(behav(:,3), 2);
    cueShape = ismember(behav(:,3), 3) - ismember(behav(:,3), 4);
    cueRepeat = ((behav(:,9)==2) - (behav(:,9)==3));

%    Column 4:
% Stimulus
% 
% 1 = yellow square
% 
% 2= blue square
% 
% 3 = yellow circle
% 
% 4 = blue circle

colorYellow = ismember(behav(:,4), [1,3]) - ismember(behav(:,4), [2,4]);
shapeSquare = ismember(behav(:,4), [1,2]) - ismember(behav(:,4), [3,4]);





    % make structure
    trial=struct;

    trial.block = block;

    trial.task = isColor;
    trial.prevTask = wasColor;
    trial.switch = isSwitch;
    trial.taskSwitch = isColor.*(isSwitch==1);
    trial.taskRepeat = isColor.*(isSwitch==-1);


    trial.cueShape = cueShape;
    trial.cueColor = cueColor;
    trial.cueRepeat = cueRepeat;

    trial.RT = rt;
    trial.prevRT = prevRT(bindx);
    trial.acc = acc ==1;
    trial.prevAcc = prevAcc(bindx) ==1;
    trial.resp = resp;

    trial.color = colorYellow;
    trial.shape = shapeSquare;

    trial.rew = isRew;
    chanLocs = eeg.chanlocs;



    % save to cell
    all_y{pp} = y;
    all_dt{pp} = dt;
    all_ts{pp} = ts;
    all_epoch{pp} = epoch;
    all_trial{pp} = trial;
    all_chanLocs{pp} = chanLocs;

end


%% save

% save
for pp = 1:npt

    y = all_y{pp};
    dt = all_dt{pp};
    ts = all_ts{pp};
    epoch = all_epoch{pp};
    trial = all_trial{pp};
    chanLocs = all_chanLocs{pp};


    fprintf("saving %d... \n", pp)
    save(sprintf('%s/data/NeSS-formatted/%s/%s_%d.mat', ROOT, saveName, saveName, pp),...
        'y',...
        'dt',...
        'ts',...
        'epoch',...
        'trial',...
        'chanLocs')


    % plot
    fprintf("plotting %d... \n", pp)
    nexttile; hold on;
    plot(mean(y,3)', 'LineWidth', 1);
    title(sprintf('pt %d',pp));
    set(gca, 'TickDir', 'out', 'LineWidth', 1);
    try
        xline(find(epoch==2,1),'--')
        xline(find(epoch==3,1),'--')
        xline(find(epoch==4,1),'--')
    catch
    end

end




f.Position = [100,100,1900,1000];
saveas(f,sprintf('%s/data/NeSS-formatted/%s/%s.png', ROOT, saveName, saveName))
saveas(f,sprintf('%s/data/NeSS-formatted/%s/%s', ROOT, saveName, saveName), 'epsc')

fprintf("\nDONE\n")




