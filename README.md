order = 3;
[z, p, k] = cheb1ap(order, 6);
[nd, d] = zp2tf(z, p, k);
 
fc = 150;
fh = 200; fl = 100;
wc = 2 * pi * fc;
bw = 2 * pi * (fh - fl);
 
[nhp, dhp] = lp2bp(nd, d, wc, bw);
 
fs = 1000;
[nd, dd] = bilinear(nhp, dhp, fs);
 
f1 = 50; f2 = 150; f3 = 300;
t = 0:1/fs:2;
x = sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t) + sin(2 * pi * f3 * t);
 
subplot(3,1,1);
plot(t, x);
ylabel('Amplitude');
xlabel('Time');
title('Input Signal');
 
L = length(x);
NFFT = 2^nextpow2(L);
y = fft(x, NFFT);
y1 = y(1:NFFT/2);
xaxis = linspace(0, fs/2, NFFT/2);
 
subplot(3,1,2);
plot(xaxis, abs(y1));
ylabel('Magnitude');
xlabel('Frequency');
title('FFT Spectrum of Input Signal');
 
z = filter(nd, dd, x);
z1 = fft(z, NFFT);
z2 = z1(1:NFFT/2);
 
subplot(3,1,3);
plot(xaxis, abs(z2));
ylabel('Magnitude');
xlabel('Frequency');
title('FFT Spectrum of Filtered Signal');

