%%


% AVILplaybackExample gives an overview how to playback an audiofile in the
% AVIL.

% The example deals with a single- and a 64-channel file.

% contact: aahr@dtu.dk

clear all;
clc;

% Audio player setup
disp('create audio player object')
fs = 32000; %sampling frequency (only 48kHz are supported!)
devName = "Speakers (RME Fireface UCX)"; %sound card device name using ASIO drivers
% devName = "Analog (3+4) (RME Fireface UCX)"
bufferSize = 2^10; %recommended buffer size

setpref('dsp','portaudioHostApi',3);

aDW = audioDeviceWriter('SampleRate', fs, 'Driver', 'DirectSound',...
    'DeviceName', devName,...
    'BufferSize', bufferSize,...
    'ChannelMappingSource', 'Auto');

% read file
disp('create file object')
exampleAudioFile = 'BluesA_Allresampled.wav'; % audio file that should be played incl. path, example: mono file
%exampleAudioFile = 'C:\Local\lscount_horiz.wav'; % audio file that should be played incl. path, example: 64channel file
frameLength = 2^10; %this can be changed depending on the processing (if a lot of processing of the audio file is done, it is recommended to use high numbers. However, that introduces longer latencies.)
aFR = dsp.AudioFileReader(exampleAudioFile, 'SamplesPerFrame', frameLength);

% Channel mapping

channelNum = 2; %vector with same number of channels as in audiofile. Only used if audiofile is is not 64 channels.

% if aFR.info.NumChannels == 2
%     aDW.ChannelMapping = 1:2;
% else
%     aDW.ChannelMapping = channelNum;
% end


% playback
disp('playback started')
global h_az, 
global h_elev

h_az=uicontrol('Style', 'slider',...
        'Min',0,'Max',355,'Value',90,...
        'Position', [400 20 120 20]);

h_elev=uicontrol('Style', 'slider',...
        'Min',-40,'Max',90,'Value',0,...
        'Position', [100 20 120 20]);


pause(2);


final = zeros(0,0);

% run Load_HRTFs.m

%%% Changes from Justin
load('BRIR_L.mat');
load('BRIR_R.mat');
hrtf_l = sl;
hrtf_r = sr;
% create an instance of the natnet client class
natnetclient = natnet;

% connect the client to the server (multicast over local loopback) -
% modify for your network
natnetclient.HostIP = '127.0.0.1';
natnetclient.ClientIP = '127.0.0.1';
natnetclient.ConnectionType = 'Multicast';
natnetclient.connect;



%%
inBuffer = zeros(frameLength+512-1,2);
while ~aFR.isDone
    
    %read chunk of target and maskers
    audioData = step(aFR);
    pause(0.0002);

    source = get(h_az,'Value');

    check_val = NatNetPollingSample(natnetclient.getFrame);% -90 on the left and +90 on the right

    ang = round(check_val/5)*(-5); % between -90 and 90, reversed because we want the opposite HRTF
%     ang = -ang % depends on the setup calibration
    pause(0.0002);
%     ang = round(check_val);
%     
%     ang = source - ang;
% 
%     if ang < 0
%         
%         ang = 360 + ang;
% 
%     end
% 
% 
%     pause(0.0002);

%     v_angle = get(h_elev,"Value");
%     v_angle = 10*round(v_angle/10);
%     disp([v_angle,ang])
% 
%     [index] = angle_index(v_angle, ang);
% 
% 
%     if v_angle >= 0
%         set_r = eval(strcat("hrtf_",num2str(v_angle),"_r"));
%         set_l = eval(strcat("hrtf_",num2str(v_angle),"_l"));
%     else 
%         set_r = eval(strcat("hrtf_m",num2str(abs(v_angle)),"_r"));
%         set_l = eval(strcat("hrtf_m",num2str(abs(v_angle)),"_l"));
%     end
%     h_left = set_l(:,index);
%     h_right = set_r(:,index);
    if ang>90
        index = size(sl,2);
    elseif ang<-90
        index = 1;
    else
        index = (ang+90)/5+1; % between 1 and 37
    end
    h_left = hrtf_l(480:480+512,index);
    h_right = hrtf_r(480:480+512,index);
    [audioData, outBuffer] = overlapAdd(audioData, h_left,h_right, inBuffer);
    
    audioData = audioData * 5;

    % sent audio to soundcard (actual playback)
    numUnderrun = step(aDW,audioData);
    final = [final;audioData];
    inBuffer = outBuffer;
    if numUnderrun > 0
        fprintf('Audio writer queue was underrun by %d samples.\n',numUnderrun);
    end

end

%% cleanup after ourselves
pause(aDW.QueueDuration); % wait until audio is played to the end
release(aFR);            % close the input file
release(aDW);             % close the audio output device


%%


function [y,outBuffer]=overlapAdd(x,h_left,h_right,inBuffer)
L=length(x);
% M=length(h);
N=length(inBuffer); %N >= L+M-1
inBuffer=[inBuffer(:,1)+ifft(fft(h_left,N).*fft(x,N)),inBuffer(:,2)+ifft(fft(h_right,N).*fft(x,N))];
y=inBuffer(1:L,:);
outBuffer=[inBuffer(L+1:N,:); zeros(L,2)];
end



function [index] = angle_index(v_ang, h_ang)
    
    switch v_ang
    
        case 0
            
            res = 5;
            index = index_gen(res, h_ang);

        case 10

            res = 5;
            index = index_gen(res, h_ang);

        case 20

            res = 5;
            index = index_gen(res, h_ang);

        case 30
            
            res = 6;
            index = index_gen(res, h_ang);

        case 40
            
            res = 6.32;
            index = index_gen(res, h_ang);

        case 50

            res = 8;
            index = index_gen(res, h_ang);

        case 60
            
            res = 10;
            index = index_gen(res, h_ang);

        case 70

            res = 15;
            index = index_gen(res, h_ang);

        case 80
            
            res = 30;
            index = index_gen(res, h_ang);

        case 90

            index = 1;
            
        case -10

            res = 5;
            index = index_gen(res, h_ang);            

        case -20

            res = 5;
            index = index_gen(res, h_ang);

        case -30

            res = 6;
            index = index_gen(res, h_ang);

        case -40
    
            res = 6.32;
            index = index_gen(res, h_ang);

    end

end


function [index] = index_gen(res, h_ang)
    h_ang = res*floor(h_ang/res);
    if h_ang == 0
        index = 1;
    else
        index = round(h_ang / res);
    end
end




function [angle_y] = NatNetPollingSample(data)

    qx = data.RigidBodies( 1 ).qx;
    qw = data.RigidBodies( 1 ).qw;
    qy = data.RigidBodies( 1 ).qy;
    qz = data.RigidBodies( 1 ).qz;

    angles = rad2deg(quat2eul([qw, qx, qy, qz]));

    angle_y = -1*angles(2);

end
