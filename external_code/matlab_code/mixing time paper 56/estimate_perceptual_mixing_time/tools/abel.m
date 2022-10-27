% From: 
%
% Abel & Huang 2006, "A simple, robust measure of reverberation echo
% density", In: Proc. of the 121st AES Convention, San Francisco
%
% Computes the transition time between early reflections and stochastic
% reverberation based on the assumption that sound pressure in a 
% reverberant field is Gaussian distributed.
%
% t_abel             - mixing time nach Abel & Huang (2006, echo density = 1)
% echo_dens          - echo density vector
% IR                 - impulse response (1 channel only!)
% N                  - window length
% fs                 - sampling rate
% peak_secure_margin - for onset detection
%
% call:
% [t_abel,echo_dens] = abel(IR,N,fs,peak_secure_margin)
%
%
% A. Lindau, L. Kosanke, 2011
% alexander.lindau@tu-berlin.de
% audio communication group
% Technical University of Berlin
%-------------------------------------------------------------------------%
function [t_abel,echo_dens] = abel(IR,N,fs,peak_secure_margin)

% preallocate
s           = zeros(1,length(IR));
anz_s       = zeros(1,length(IR));
echo_dens   = zeros(1,length(IR));

if length(IR) < N
    error('IR shorter than analysis window length (1024 samples). Provide at least an IR of some 100 msec.')
end

for n = 1:length(IR)
    % window at the beginning (increasing window length)
    if n <= N/2+1
        
        % standard deviation 
        s(n)        = std(IR(1:n+N/2-1));
        
        % number of tips outside the standard deviation
        anz_s(n)    = sum(abs(IR(1:n+N/2-1))>s(n)); 
        
        % echo density
        echo_dens(n)= anz_s(n)/N;                                   
    end

    % window in the middle (constant window length)
    if n > N/2+1 && n <= length(IR)-N/2+1    
        s(n)        = std(IR(n-N/2:n+N/2-1));
        anz_s(n)    = sum(abs(IR(n-N/2:n+N/2-1))>s(n));
        echo_dens(n)= anz_s(n)/N;                            
    end

    % window at the end (decreasing window length)
    if n > length(IR)-N/2+1   
        s(n)        = std(IR(n-N/2:length(IR)));
        anz_s(n)    = sum(abs(IR(n-N/2:length(IR)))>s(n));
        echo_dens(n)= anz_s(n)/N;
    end
end

% normalize echo density
echo_dens       = echo_dens./erfc(1/sqrt(2));                   

% transition point (Abel & Huang (2006))
% (echo density first time greater than 1)
d           = min(find(echo_dens>1));                       
t_abel      = (d-peak_secure_margin)/fs*1000;

if isempty(t_abel)
    fprintf('\n')
    error('Mixing time not found within given temporal limits. Try again with extended stopping crtiterion.')
end


