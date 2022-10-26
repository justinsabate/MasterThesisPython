% This matlab script accompanies the article:
%
% Lindau, A.; Kosanke, L.; Weinzierl, S.: 
% "Perceptual Evaluation of Model- and Signal-Based Predictors of the 
% Mixing Time in Binaural Room Impulse Responses", In:  J. A. Eng. Soc.
%
% It serves for computing estimates of the perceptual mixing time, i.e. 
% the instant a room's sound field is percieved as being diffuse. The 
% perceptual mixing is estimated using model- and/or signal-based 
% predictors. Please note that - assuming a worst-case situation - 
% predictors were developed for the application in rectangular (i.e. 
% shoe-box-shaped) rooms. Their validity might thus be limited to such 
% rooms.
%
% As model-based predictors the enclosure's ratio of volume over surface 
% area V/S, or its volume V are used, respectively. If an impulse response 
% is available, the data-based predictor from Abel & Huang 
% (2006, criterion I, cf. above cited article) is be applied.
%
% In both cases, the perceptual mixing time is predicted as the just 
% audible value for a) the average, and b) - as a more strict criterion - 
% for the 95%-point of the assumed normal distributed population.
%
% In case of data-based estimation, the provided impulse responses should 
% fulfill three conditions. First they should be delivered in a 
% matlab-readable *.wav-format (single- or multichannel). Second, the 
% direct sound in the impulse response should be the strongest signal 
% component as the calculation conducts a peak detection with according 
% assupmtion. Third, per default the peak-detect algorithm assumes a 
% peak-to-noise-ratio of at least 30 dB (value can be changed in the script
% via the variable onset_threshold_dB).
%
% After running the code you will be guided through it by step-by-step 
% instructions.
% 
% Additionally needed files are provided in folder '/tools' and 
% will automatically be added to the matlab path:
%
% 	model_based.m
%	data_based.m
% 	abel.m
% 	cut.m
% 	plot_tmp.m
%
% This code has been tested with Matlab vs. R2009a. Figure's format might be
% corrupted when submitting impulse responses with more than 2 channels.
% 
% Additional references:
% 
% Abel & Huang (2006): "A simple, robust measure of reverberation echo
% density", In: Proc. of the 121st AES Convention, San Francisco
%
%
% A. Lindau, L. Kosanke, 2011
% alexander.lindau@tu-berlin.de
% audio communication group
% Technical University of Berlin
%-------------------------------------------------------------------------%
clear all; close all;  clc

% Add all needed files to matlab path 

% find  path of current script
current_script_path = mfilename('fullpath');

idx = strfind(current_script_path,filesep);
current_script_path = current_script_path(1:max(idx)-1);

% for checking whether folders were already added
p = genpath(current_script_path); % current path and all subfolders
p2 = path;

% if not already done, add with all subfolders to matlab path
if isempty(strfind(p2,p))
    fprintf ('\nAdding all files needed for mixing time prediction to Matlab''s path.\n\n')
    addpath(p);
end




%-------------------------------------------------------------------------%
% processing parameters

N = 1024;                 % window length (see Abel & Huang, 2006)
onset_threshold_dB = -30; % peak criterion(onset_threshold*maximum)
peak_secure_margin = 100; % used to protect peak from being affected, when elimiating time of flight

%-------------------------------------------------------------------------%
% model-based predictors: V/S and V
display('----------------------------------')
display('MODEL-BASED MIXING TIME PREDICTION')
display('----------------------------------')
display('ROOM PROPERTIES:')

h = input('height in [m] (ENTER to skip model-based prediction): ');

if ~(isempty(h)) % ... for skipping...
    
    if  h<0 || ~(isnumeric(h))
        error('Insert positive, numeric values only.')
    end
    l = input('length in [m]: ');
    if  l<0 || ~(isnumeric(l))
        error('Insert positive, numeric values only.')
    end
    b = input('width in [m]: ');
    if  b<0 || ~(isnumeric(b))
        error('Insert positive, numeric values only.')
    end
    
    % compute perceptual mixing time (model_based)
    display('Perceptual mixing times tmp50 and tmp9 (in ms) from model-based predictors:')
    [tmp50_model_based, tmp95_model_based]    = model_based(h,b,l)
    
else
    display('Skipped model-based mixing time prediction.')
    tmp50_model_based = [];
    tmp95_model_based  = [];
end

%-------------------------------------------------------------------------%
% data based predictor: Abel & Huang (2006, criterion I)
display(' ')
display('---------------------------------')
display('DATA-BASED MIXING TIME PREDICTION')
display('---------------------------------')
display('IMPULSE RESPONSE PROPERTIES:')

IR_name    = input('Insert complete path of the impulse response (ENTER to skip): ','s');

% check input
if isempty(IR_name)
    display('Skipped data-based mixing time prediction.')
    tmp50_data_based        = [];
    tmp95_data_based        = [];
    tmp50_interchannel_mean_data_based   = [];
    tmp95_interchannel_mean_data_based   = [];
    echo_dens               = [];
    fs                      = [];
else
    [IR,fs]        = audioread(IR_name);
    
    fprintf('\n')
    channel_number  = input('Channel number (ENTER to use all): ');
        
    % specific channel number chosen?
    if size(channel_number,1) == 1
        % check channel number
        if size(IR,2) < channel_number
            display('Channel does not exist, channel 1 is used')
            channel_number = 1;
        end
        % use the chosen channel of the IR
        IR = IR(:,channel_number);
    end
    
    % ask for stopping time
    fprintf('\n')
    display('To increase calculation speed: Provide a maximum length in [ms]')
    display('of the impulse response to be taken into account (e.g. twice')
    display('the expected mixing time, ENTER to choose default (300 ms),')
    stop_time = input('insert [0] for complete IR): ');
    if stop_time == 0
        fprintf('\nUsing complete impulse response for calculation (might take a while).\n \n')
    else
        if isempty(stop_time)
            fprintf('\n')
            display('Using 300 ms (default) as stopping time!')
            stop_time = 300;
        end
    end
        
    % compare IR-length and provided stop_time
    if stop_time ~= 0
         stop_time_samples = floor(stop_time/1000*fs);
         if length(IR) < stop_time_samples 
             fprintf('\nProvided impulse response is shorter than demanded stopping time.')
             display('Using complete impulse response for calculation (might take a while).')
             stop_time = 0;
         end
    end
        
    % cut IR (use IR only from onset position to stop_time) 
    IR = cut(IR,fs,stop_time,onset_threshold_dB,peak_secure_margin);
    
    % compute perceptual mixing time (data_based)
    fprintf('\n')
    display('Calcluating perceptual mixing times tmp50 and tmp95 (in ms) from data-based predictors ...')
    [tmp50_data_based, tmp95_data_based, tmp50_interchannel_mean_data_based, tmp95_interchannel_mean_data_based,echo_dens] = data_based(IR,N,fs,peak_secure_margin);
    fprintf('\n')
    display('Finished!')
    tmp50_data_based
    tmp95_data_based
    tmp50_interchannel_mean_data_based
    tmp95_interchannel_mean_data_based
end


if ~ ((isempty(h)) && isempty(IR_name)) % ... only if at least sthg. was calculated
    % save results as picture?
    do_print    = input('Save results as *.tif [y/n] (ENTER = No)?','s');
    if do_print == 'Y' | do_print == 'y'
        do_print = 1;
    else
        do_print = 0;
    end 
    plot_tmp(tmp50_model_based, tmp95_model_based, tmp50_data_based, tmp95_data_based, tmp50_interchannel_mean_data_based, tmp95_interchannel_mean_data_based, echo_dens,fs,do_print)
end