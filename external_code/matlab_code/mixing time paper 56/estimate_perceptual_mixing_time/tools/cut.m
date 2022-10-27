% Cuts the impulse response (IR) from onset position to stop_time.
%
% IR_cut                   - IR from onsetposition to stop_time
% IR                       - impulse response (multichannel is possible)
% fs                       - sampling rate
% stop_time                - stopping time in ms, if = 0 apply no shortening
% onset_threshold_dB       - peak criterion (onset_threshold*maximum)
%
%
% call:
% IR_cut = cut(IR,fs,stop_time,onset_threshold_dB,peak_secure_margin)
%
% A. Lindau, L. Kosanke, 2011
% alexander.lindau@tu-berlin.de
% audio communication group
% Technical University of Berlin
%------------------------------------------------------------------------
function IR_cut = cut(IR, fs, stop_time, onset_threshold_dB, peak_secure_margin)

% calculate linear value of onset threshold from dB value
onset_threshold=10^(onset_threshold_dB/20);

% for all IR-channels, find position where onset threshold value is reached
go_on=1; k=0;

for i=1:size(IR,2)
    MAX=max(abs(IR(:,i)));
    % for full lenght of channel, find peak position
    while go_on
        k=k+1;
        
        % speichere beginn der IR je Kanal in "del" ab
        if abs(IR(k,i)) > MAX*(onset_threshold)
            del(i)=k;
            go_on=0;
        end
    end
    go_on=1;k=0;
end

% convert stop_time in [ms] to samples and shorten
if stop_time ~= 0
    
    stop_time = floor(stop_time/1000*fs); % IR length from peak to stop_time in samples
    
    if min(del) <= peak_secure_margin
        peak_secure_margin = 0; % ignore peak_secure_margin
    end
    
    for j=1:size(IR,2)
        IR_cut(:,j) = IR(min(del)-peak_secure_margin:min(del)-peak_secure_margin+stop_time,j);
    end
 
else % ... no shortening
    
    if min(del) <= peak_secure_margin
        peak_secure_margin = 0; % ignore peak_secure_margin
    end
    
    stop_time = length(IR)-(min(del)-peak_secure_margin); % IR length from peak to end in samples
       
    for j=1:size(IR,2)
        IR_cut(:,j) = IR(min(del)-peak_secure_margin:min(del)-peak_secure_margin+stop_time,j);
    end
    
end




