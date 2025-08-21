function [truecorr, pvals, nullcorrs] = phase_rand_corr_v2(x, y, Nscram, tail, permutation)

% function [truecorr, pvals, nullcorrs] = PHASE_RAND_CORR(x,y,nscram, tail, permutation)
%
% This function calculates the correlation between 
%     [1] a phase-scrambled version of each column of 'x'
% and 
%     [2] the intact vector 'y'
% to produce a distribution of null correlations in which we have controlled
% for the power spectrum (and thus temporal autocorrelation) of the input time-series.
%
% INPUT
% x =        [Nsamp by K] matrix, each of whose K columns will be phase-scrambled,
%             and correlated against the vector input 'y'.
% y =        [Nsamp by 1] input vector [will be left intact]
% Nscram =   [integer] number of phase-scramble correlation value to compute (default: 1000)
% tail =     flag indicating distributional tail to examine for stats:
%            -1 --> left-tail, 0 --> two-tail, +1 --> right-tailed
% permutation = [boolean] true: permute existing phases, false: generate random phases
%
% OUTPUT
% truecorr = [1 by K] vector of Pearson correlation between intact 'x' and intact 'y'
% pvals =    [1 by K] vector of corresponding p-values for the values of truecorr
%                  based on comparison with null distributions 
% nullcorrs = [Nscram by K] matrix of "null" Pearson correlations between
%                   phase-scrambled columns of 'x and intact y
%
% TODO: 
%   implement some tapering to avoid high-freq artifacts in the fft
%   (but the procedure could be potentially probematic)
%   
%
% Author: CJ Honey
% Version: 0.1, April 2010  (phase_scram_corr_Nvs1)
% Version: 0.2, March 2011    -- fixed bug with zero-th phase component;
%                             -- randomizes rather than scrambles phase 
%                             based on feedback from Jochen Weber
%


if nargin < 5; permutation = false; end %by default we generate random phases
if nargin < 4; tail = 0; end  %two-tailed test by default
if nargin < 3; Nscram = 1000; end
if nargin < 2; fprintf('At least two inputs required. Aborting. \n'); return; end

[Nsamp K] = size(x);  %extract number of samples and number of signals

% convert x and y to column vectors if necessary
[Ry Cy] = size(y);
if Cy > 1; fprintf('Column vector input required for input Y. Aborting. \n'); return; end
if Ry ~= Nsamp; fprintf('X and Y must have an equal number of rows. Aborting.\n'); return; end


x = x - repmat(mean(x,1), Nsamp,1);  %remove the mean of each column of X
x = x./sqrt(repmat(dot(x,x), Nsamp, 1)/(Nsamp-1)); %divide by the standard deviation of each column

y = (y-mean(y)); y = y/sqrt(dot(y,y)/(Nsamp-1)); %convert vector Y to unit mean and variance

%transform the vectors-to-be-scrambled to the frequency domain
Fx = fft(x); 

% identify indices of positive and negative frequency components of the fft
% we need to know these so that we can symmetrize phase of neg and pos freq
if mod(Nsamp,2) == 0
    posfreqs = 2:(Nsamp/2);
    negfreqs = Nsamp : -1 : (Nsamp/2)+2;
else
    posfreqs = 2:(Nsamp+1)/2;
    negfreqs = Nsamp : -1 : (Nsamp+1)/2 + 1;
end

x_amp = abs(Fx);  %get the amplitude of the Fourier components

if permutation
    x_phase = atan2(imag(Fx), real(Fx)); %get the phases of the Fourier components [NB: must use 'atan2', not 'atan' to get the sign of the angle right]
end

J = sqrt(-1);  %define the vertical vector in the complex plane

nullcorrs = zeros(Nscram,K);  %will contain the distribution of null correlations for each input
% rand_phase = zeros(Nsamp,K);  %will cotnain the randomized phases for each input channel on each bootstrap
sym_phase = zeros(Nsamp,K);   %will contain symmetrized randomized phases for each bootstrap


for n = 1:Nscram

  if permutation  
        [tmp,rp] = sort(rand(Nsamp,K));
        x_phase=x_phase(rp);
    	sym_phase(posfreqs,:) = x_phase(1:length(posfreqs),:);
        sym_phase(negfreqs,:) = -x_phase(1:length(posfreqs),:);
	else
        new_phase=2*pi*rand(length(posfreqs),K);
    	sym_phase(posfreqs,:) = new_phase;
        sym_phase(negfreqs,:) = -new_phase;
  end
    
    z = x_amp.*exp(J.*sym_phase); %generate (symmetric)-phase-scrambled Fourier components
    null_x = ifft(z); %invert the fft to generate a phase-scrambled version of x
    
    nullcorrs(n,:) = y'*null_x;  %compute rapid correlation via dot product
end

truecorr = y'*x / (Nsamp-1); %produce the true Pearson correlations of the inputs
nullcorrs = nullcorrs / (Nsamp-1);  %normalize to produce a Pearson correlation coefficient

switch tail
    case +1  %right-tailed test 
        pvals = sum(repmat(truecorr,Nscram,1) < nullcorrs)/Nscram;   %what proportion of the boostrapped correlations exceed the observed correlation
    case -1  %left-tailed test
        pvals = sum(repmat(truecorr,Nscram,1) > nullcorrs)/Nscram;   %what proportion of the boostrapped correlations are less than the observed correlation
    case 0   %two tailed test (default)
        pvals = sum( abs(repmat(truecorr,Nscram,1)) < abs(nullcorrs))/Nscram;   
end

end





