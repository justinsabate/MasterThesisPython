[x, fs]=audioread('BluesA_All.wav');
y = resample(x,32000,fs);
audiowrite('BluesA_Allresampled.wav',y,32000)