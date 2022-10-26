% Computes the perceptual mixing time from data based predictors as
% described in:
%
% Lindau, A.; Kosanke, L.; Weinzierl, S.: 
% "Perceptual evaluation of model- and signal-based predictors of the 
% mixing time in binaural room impulse responses", In:  J. A. Eng. Soc.
%
% tmp50                - average perceptual mixing time
% tmp95                - 95%-point perceptual mixing time
% tmp50_interchannel_mean           - interchannel average of average perceptual mixing time
% tmp95_interchannel_mean           - interchannel average of average of 95%-point perceptual mixing time
% IR                   - room impulse response (multichannel is possible)
% N                    - window length (see Abel 2006)
% fs                   - sampling frequency
% peak_secure_margin   - for onset detection
%
% call:
% [tmp50, tmp95, tmp50_interchannel_mean, tmp95_interchannel_mean] = data_based(IR,N,fs)
%
% dependencies: abel.m
%
% A. Lindau, L. Kosanke, 2011
% alexander.lindau@tu-berlin.de
% audio communication group
% Technical University of Berlin
%-------------------------------------------------------------------------%
function [tmp50, tmp95, tmp50_interchannel_mean, tmp95_interchannel_mean, echo_dens] = data_based(IR,N,fs,peak_secure_margin)

% preallocate
t_abel      = zeros(1,size(IR,2));
echo_dens   = zeros(length(IR),size(IR,2));

for n = 1:size(IR,2)
    [t_abel(n),echo_dens(:,n)] = abel(IR(:,n),N,fs,peak_secure_margin);
end

% tmp from regression equations
tmp50 = 0.8 .* t_abel - 8;
tmp95 = 1.77 .* t_abel -38;


% clip negative values
if sum(tmp50<0)
    idx = find(tmp50<0);
    for g = 1:length(idx)
        tmp50(idx(g)) = 1;
    end
end

if sum(tmp95<0)
    idx = find(tmp95<0);
    for g = 1:length(idx)
        tmp95(idx(g)) = 1;
    end
end

% average perceptual mixing time over all channels
if size(IR,2)>1
    t_abel_interchannel_mean = mean(t_abel);
    tmp50_interchannel_mean = 0.8 .* t_abel_interchannel_mean - 8;
    tmp95_interchannel_mean = 1.77 .* t_abel_interchannel_mean -38;
else
    tmp50_interchannel_mean = [];
    tmp95_interchannel_mean = [];
end










