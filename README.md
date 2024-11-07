AIM: To generate elementary signals
ALGORITHM: 1. Assign sampling frequency and duration of time signal 2. Generate unit impulse, unit step, ramp and sinusoidal signal 3. Plot them continuous time signals and discrete time signals
 
clc;
clear all;
close all;
fs=10;
t=0:1/fs:1;
n=-10:10;
 
% unit impulse
x1=[zeros(1,10),ones(1,1),zeros(1,10)];
subplot(4,2,1);plot(n,x1);
xlabel('Time');ylabel('Amplitude');title('Continuous Unit Impulse');
subplot(4,2,2);stem(n,x1);
xlabel('Index');ylabel('Amplitude');title('Discrete Unit Impulse');
 
% unit step
x2=[zeros(1,10),ones(1,11)];
subplot(4,2,3);plot(n,x2);
xlabel('Time');ylabel('Amplitude');title('Continuous Unit Step');
subplot(4,2,4);stem(n,x2);
xlabel('Index');ylabel('Amplitude');title('Discrete Unit Step');
 
% unit ramp
x3=x2.*n;
subplot(4,2,5);plot(n,x3);
xlabel('Time');ylabel('Amplitude');title('Continuous Ramp');
subplot(4,2,6);stem(n,x3);
xlabel('Index');ylabel('Amplitude');title('Discrete Ramp');
 
% sinusoidal signal
x4=sin(2*pi*t);
subplot(4,2,7);plot(t,x4);
xlabel('Time');ylabel('Amplitude');title('Continuous Sine');
subplot(4,2,8);stem(t,x4);
xlabel('Index');ylabel('Amplitude');title('Discrete Sine');
 
AIM: To get unit response, step response and frequency response for a given transfer function
ALGORITHM: 1. Let y[n]-y[n-1]=x[n]+x[n-1]-x[n-2], find the transfer function 2. Plot impulse response and step response 3. Plot frequency response in s-domain and z-domain
clc;
clear all;
close all;
fs=10;
 
% if y[n]-y[n-1]=x[n]+x[n-1]-x[n-2],
% Y(jw)-e^-jw*Y(jw)=X(jw)+e^-jw*X(jw)-e^-2jw*X(jw)
% Transfer function H(w)=Y(w)/X(w)
% H(w)=(1+(e^-jw)-(e^-2jw))/(1-(e^-jw)),
% numerator coeff=[1 1 -1], denominator coeff=[1 -1]
 
num=[1 1 -1];
den=[1 -1];
subplot(3,1,1);impz(num,den);
subplot(3,1,2);stepz(num,den);
subplot(3,1,3);zplane(num,den);
 
figure(2);freqz(num,den);
figure(3);freqs(num,den);
 
 
EXP 2A: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To perform shifting and scaling operations in a signal and analyse the process in time domain
ALGORITHM: 1. Generate a signal 2. Perform shifting, scaling and reversing operations 3. Plot the signal
 
f = 5;   	% Frequency of the sine wave
fs = 100;	% Sampling frequency
t = 0:1/fs:1; % Time vector from 0 to 1 second with sampling interval of 1/fs
x = sin(2 * pi * f * t); % Sine wave signal
 
subplot(4, 1, 1)
plot(t, x)
xlabel("Time")
ylabel("Amplitude")
title("Original wave")
 
subplot(4, 1, 2)
plot(t / 2, x)
xlabel("Time")
ylabel("Amplitude")
title("Time Scaling")
 
subplot(4, 1, 3)
plot(t + 1, x)
xlabel("Time")
ylabel("Amplitude")
title("Time Shifting")
 
subplot(4, 1, 4)
plot(-t, x)
xlabel("Time")
ylabel("Amplitude")
title("Time Reversal")
 
 
EXP 2B: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To acquire two speech signals by live recording and perform convolution
ALGORITHM: 1. Assign the sampling frequency, channel and bits 2. Record the signals using a microphone 3. Convolute the signals 4. Plot the recorded signal and their convolution
 
rec = audiorecorder(44100, 16, 1);
recordblocking(rec, 5);          	
y1 = getaudiodata(rec);          	
subplot(3, 1, 1); plot(y1);      	
title('Speech Signal 1');
 
rec = audiorecorder(44100, 16, 1);  
recordblocking(rec, 5);          	
y2 = getaudiodata(rec);          	
subplot(3, 1, 2); plot(y2);      	
title('Speech Signal 2');
 
y = conv(y1, y2, 'full');        	
subplot(3, 1, 3); plot(y);       	
title('Convolution');
 
 
EXP 2C: IMAGE AND SOUND SIGNAL PROCESSING
 
AIM: To acquire stored audio data in ‘.wav’ format and perform convolution and correlation
ALGORITHM: 1. Acquire the ‘.wav’ signals using ‘audioread()’ function 2. Plot the acquired signal and perform convolution in time and frequency domain operations 3. Find correlation between the signals 4. Plot the signals
clc;
clear all;
close all;
[sig1, fs1] = audioread('s1.wav');
t = (1:length(sig1)) / fs1;
subplot(3,1,1);
plot(t, sig1);
xlabel('seconds');
ylabel('Relative Signal Strength');
title('song');
[sig2, fs2] = audioread('s2.wav');
x = (1:length(sig1)) * 0;
x1 = (1:length(sig1)) * 0;
ex = (1:length(sig1)) * 0;
ex1 = (1:length(sig1)) * 0;
t1 = (1:length(sig1)) / fs1;
subplot(3,1,2);
plot(t1, sig2);
xlabel('seconds');
ylabel('Relative Signal Strength');
title('speech');
c = conv(sig1, sig2, 'same');
sound(c, fs1);
t = (1:length(sig1)) / fs1;
subplot(3,1,3);
plot(t, c);
xlabel('seconds');
ylabel('Relative Signal Strength');
title('convolved Signal');
 
clc; clear all; close all;
[y, fs] = audioread('s1.wav');
timelag = 1;
delta = round(fs * timelag);
alpha = 0.3;
orig = [y; zeros(delta,2)];
echo = [zeros(delta,2); y] * alpha;
Rx = orig + echo;
t = (0:length(Rx)-1) / fs;
subplot(3,1,1);
plot(t, [orig echo]);
title('Original + Echo');
subplot(3,1,2); plot(t, Rx);
title('Total');
xlabel('Time (s)');
[r, lags] = xcorr(Rx, 'unbiased');
r = r(lags >= 0); lags = lags(lags >= 0);
subplot(3,1,3); plot(lags / fs, r);
title('Correlation');
ylabel('Correlation');
xlabel('Lag (s)');
figure();
[peak_value_loc] = findpeaks(r, lags, 'MinPeakHeight', 0.005);
den = [1 zeros(1,loc(1)-1) alpha];
filtered = filter(1, den, Rx);
subplot(3,1,1); plot(t, Rx);
title('Echo Added');
subplot(3,1,2); plot(t, orig);
title('original');
subplot(3,1,3);
plot(t, filtered);
title('Filtered'); xlabel('Time (s)');
 
clc; clear all; close all;
[sig1, fs] = audioread('s1.wav');
f = (1:length(sig1)) / fs;
X = fftshift(fft(sig1));
Xmag = abs(X);
subplot(3,1,1);
plot(f, Xmag);
xlabel('Frequency');
ylabel('Relative Signal Strength');
title('song');
[sig2, fs] = audioread('s2.wav');
x = (1:length(sig1)) * 0;
x1 = (1:length(sig1)) * 0;
fx = (1:length(sig1)) / fs;
X1 = fftshift(fft(x));
X1mag = abs(X1);
subplot(3,1,2);
plot(fx, X1mag);
xlabel('Frequency');
ylabel('Relative Signal Strength');
title('Speech Signal');
soundsc(sig1, fs);
soundsc(sig2, fs, 'same');
X2 = fftshift(fft(x2));
X2mag = abs(X2);
f2 = (1:length(sig1)) / fs;
subplot(3,1,3);
plot(f2, X2mag);
xlabel('Frequency');
ylabel('Relative Signal Strength');
title('convolved Signal');
 
EXP 2D: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To acquire stored audio data in ‘.wav’ format and perform shifting and scaling operations
 
[y,fs]=audioread(‘.wav’)
Dt=1/fs;
t=0:dt:(length(y)*dt)-dt;
subplot(4,1,1);
plot(t,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘sound signal’);
subplot(4,1,2);
plot(t+5,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘shifted signal’);
subplot(4,1,3);
plot(2*t,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘scaled signal’);
subplot(4,1,4);
plot(-t,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘reversed signal’);
 
EXP 2C: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To read an image and apply Discrete Cosine Transform and reconstruct the image using Inverse DCT.
 
 
Clc;
Clear all;
Close all;
a=imread(‘cameraman.tif’);
[M,N]=size(a);
Subplot(2,3,1);
Imshow(a);
B=dct2(a);
Subplot(2,3,2);
Imshow(abs(b),[]);
Subplot(2,3,3);
e=idct2(b);
subplot(2,3,3);
imshow(e,[]);
 
 
EXP 3A: FOURIER TRANSFORM
AIM: To apply Fourier Transform to a signal and reconstruct the signal using Inverse Fourier Transform
 
%CTFS
t0=0; T=3;           	% Set the initial time and period
w=2*pi/T;            	% Angular frequency
syms t               	% Define symbolic variable t
x=exp(-t);           	% Define the function x(t) = exp(-t)
 
subplot(3,1,1);
fplot(x,[t0 t0+T]);  	% Plot the original function over [t0, t0+T]
xlabel('Amplitude'); ylabel('time'); title('Input');
 
% Calculate CTFS coefficients with k = -10 to 10
for k=-10:10
	c(k+11)=(1/T)*int(x*exp(-1i*k*w*t), t, t0, t0+T);  % Fourier coefficient calculation
end
 
% Generate complex exponentials for k = -10 to 10
for k=-10:10
	cx(k+11)=exp(1i*k*w*t);                        	% e^(j*k*w*t)
end
 
% Reconstruct signal using k = -10 to 10
xx=sum(c.*cx);                                     	% Summation of terms for signal reconstruction
subplot(3,1,2);
fplot(xx, [t0 t0+T]);                                  % Plot reconstructed signal with k = 10
xlabel('Amplitude'); ylabel('time'); title('Reconstructed with k=10');
 
% Calculate CTFS coefficients with k = -50 to 50
for k=-50:50
	c(k+51)=(1/T)*int(x*exp(-1i*k*w*t), t, t0, t0+T);  % Fourier coefficient calculation
end
 
% Generate complex exponentials for k = -50 to 50
for k=-50:50
	cx(k+51)=exp(1i*k*w*t);                        	% e^(j*k*w*t)
end
 
% Reconstruct signal using k = -50 to 50
xx=sum(c.*cx);                                     	% Summation of terms for signal reconstruction
subplot(3,1,3);
fplot(xx, [t0 t0+T]);                                  % Plot reconstructed signal with k = 50
xlabel('Amplitude'); ylabel('time'); title('Reconstructed with k=50');
 
%DTFS
clc;
clear all;
close all;
 
syms w;
x=0.8.^n;           	% Define signal x(n) = (0.8)^n
n=-10:10;           	% Define range for n
subplot(2,1,1); stem(n,x);
xlabel('n'); ylabel('x(n)'); title('Original Signal');
 
% Calculate DTFS coefficients for k = -10 to 10
for k=-10:10
    c(k+11)=(1/length(n))*sum(x.*exp(-j*k*w*n));   % DTFS coefficient calculation
end
 
% Reconstruct signal using DTFS
x1=sum(c.*exp(j*k*w*n));   % Summation of terms for signal reconstruction
subplot(2,1,2); stem(n,x1);
xlabel('n'); ylabel('x1(n)'); title('Reconstructed Signal');
 
%DTFT
clc;
clear all;
close all;
 
syms w;
x=0.8.^n;           	% Define signal x(n) = (0.8)^n
n=-10:10;           	% Define range for n
subplot(3,1,1); stem(n,x);
xlabel('n'); ylabel('x(n)'); title('Original Signal');
 
% Compute DTFT
X=sum(x.*exp(-1i*w*n));
subplot(3,1,2);
ezplot(abs(X), [-10, 10]);
xlabel('frequency'); ylabel('|X(w)|'); title('Magnitude Response');
 
% Inverse DTFT to reconstruct the signal
x1=(1/(2*pi))*int(X.*exp(1i*w*n), w, -pi, pi);
subplot(3,1,3); stem(n,x1);
xlabel('n'); ylabel('x(n)'); title('Reconstructed Signal');
 
%CTFT
t0=0; T=3;
w=2*pi/T;
syms t w;
x=exp(-t^2);           	% Define signal x(t) = exp(-t^2)
 
subplot(3,1,1);
fplot(x, [-t0-T t0+T]);
ylabel('Amplitude'); xlabel('time'); title('Input');
 
% Compute CTFT
X=int(x*exp(-1i*w*t), t, -inf, inf);
subplot(3,1,2);
fplot(X, [-pi pi]);
ylabel('Magnitude'); xlabel('Frequency'); title('Magnitude Response');
 
% Inverse CTFT to reconstruct the signal
x1=(1/(2*pi))*int(X*exp(1i*w*t), w, -inf, inf);
subplot(3,1,3); fplot(x1, [-t0-T t0+T]);
ylabel('Amplitude'); xlabel('time'); title('Reconstructed');
 
EXP 3B: FOURIER TRANSFORM
AIM: To denoise a signal using DFT and IDFT
 
N = 256;
n = 0:N-1;
w = 2*pi/N;
x = 7*cos(3*n*w) + 13*sin(6*n*w);
subplot(321); plot(x); title('original signal');
xlabel('time index, n'); ylabel('x(n)');
Xk = fft(x);
subplot(322); stem(abs(Xk)); title('DFT peaks at k = 3 and 6');
xlabel('frequency index, k'); ylabel('X(k)');
 
xn = x + 10*randn(1,N);
subplot(323); plot(xn); title('original signal+noise');
xlabel('time index, n'); ylabel('xn(n)');
Xnk = fft(xn);
subplot(324); stem(abs(Xnk)); title('DFT peaks at k = 3 and 6');
xlabel('frequency index, k'); ylabel('Xn(k)');
 
iz = find(abs(Xnk)/N*2 < 4);
Xnk(iz) = zeros(size(iz));
subplot(325); stem(abs(Xnk));
title('DFT peaks at k = 3 and 6 are to be retained');
xlabel('frequency index, k'); ylabel('Xn(k)');
xr = ifft(Xnk);
subplot(326); plot(real(xr)); title('recovered signal');
xlabel('time index, n'); ylabel('xr(n)');
 
EXP 3C: FOURIER TRANSFORM
AIM: To apply DFT to a sine waveform with different cycles and a DC component
ALGORITHM: 1. Generate a sine signal with dc component and apply DFT to the signal 2. Add DC component and apply DFT
 
N = 64;
n = 0:N-1;
w = 2*pi/N;
 
x1 = sin(n*w);
subplot(321); stem(x1);
title('input signal, one cycle');
xlabel('time index, n \rightarrow'); ylabel('x1(n) \rightarrow');
X1k = fft(x1);
subplot(322); stem(abs(X1k));
title('DFT of input signal, non-zero at k=1 and k=63');
xlabel('frequency index, k \rightarrow'); ylabel('X1(k) \rightarrow');
 
x2 = sin(2*n*w);
subplot(323); stem(x2);
title('input signal, two cycles');
xlabel('time index, n \rightarrow'); ylabel('x2(n) \rightarrow');
X2k = fft(x2);
subplot(324); stem(abs(X2k));
title('DFT of input signal, non-zero at k=2 and k=62');
xlabel('frequency index, k \rightarrow'); ylabel('X2(k) \rightarrow');
 
x3 = sin(7*n*w);
subplot(325); stem(x3);
title('input signal, seven cycles');
xlabel('time index, n \rightarrow'); ylabel('x3(n) \rightarrow');
X3k = fft(x3);
subplot(326); stem(abs(X3k));
title('DFT of input signal, non-zero at k=7 and k=57');
xlabel('frequency index, k \rightarrow'); ylabel('X3(k) \rightarrow');
 
N = 64;
n = 0:N-1;
w = 2*pi/N;
 
x1 = sin(n*w) + 5;
subplot(221); stem(x1);
title('one cycle added with 5 dc component');
xlabel('time index, n \rightarrow'); ylabel('x1(n) \rightarrow');
X1k = fft(x1);
subplot(222); stem(abs(X1k));
title('DFT of input signal, non-zero at k=1,63 with DC component 320');
xlabel('frequency index, k \rightarrow'); ylabel('X1(k) \rightarrow');
 
x2 = sin(2*n*w) + (4*sin(7*n*w)) + 3;
subplot(223); stem(x2);
title('input signal, sin(2*n*w) + (4*sin(7*n*w)) + 3');
xlabel('time index, n \rightarrow'); ylabel('x2(n) \rightarrow');
X2k = fft(x2);
subplot(224); stem(abs(X2k));
title('DFT of input, non-zero at k=2&62,7&57 with DC component 3x64=192');
xlabel('frequency index, k \rightarrow'); ylabel('X2(k) \rightarrow');
 
 
EXP 4A: FINITE IMPULSE RESPONSE FILTERS
AIM: To design and test a filter using frequency sampling method
ALGORITHM 1. Design a FIR Filter using ‘fir2’ 2. Generate a signal and plot the fft spectrum 3. Apply filter to the signal and plot the fft spectrum
 
 
fs=1000;
t=0:1/fs:1;
f1=100;f2=200;
f3=300;f4=400;
 
x=sin(2*pi*f1*t)+sin(2*pi*f2*t)+sin(2*pi*f3*t)+sin(2*pi*f4*t);
subplot(3,1,1);
plot(t,x);
xlabel('Time');ylabel('Amplitude');title('Generated Signal');
l=nextpow2(length(t));
l1=2^l;
y=fft(x,l1);
y1=y(1:l1/2);
xaxis = linspace(0, fs/2, l1/2);
subplot(3,1,2);
plot(xaxis, abs(y1));
xlabel('Frequency');ylabel('Magnitude');title('Frequency Spectrum');
 
w1=f1/(fs/2);w2=f2/(fs/2);
w3=f3/(fs/2);w4=f4/(fs/2);
 
f2=[0 w2 w2 1];
m2=[1 1 0 0];
b2=fir2(30,f2,m2);a2=1;
zL=filter(b2,a2,x);
zL1=fft(zL,l1);
zL2=zL1(1:l1/2);
subplot(3,1,3);
plot(xaxis,abs(zL2));
xlabel('Frequency');ylabel('Magnitude');title('Frequency Spectrum LPF');
EXP 4B: FINITE IMPULSE RESPONSE FILTERS
AIM: To design and test a filter using windowing method
ALGORITHM 1. Design a FIR Filter using ‘fir1’ 2. Generate a signal and plot the fft spectrum 3. Apply filter to the signal and plot the fft spectrum
 
clc;
 
clear all;
 
close all;
 
N = 25
 
fs = 1000 fc = 100
 
wc=2*fc/fs;
 
t = 0 1/fs:1;
 
f * 1 = 50 f * 2 = 200 f * 3 = 300 f * 4 = 400
 
x=sin(2*pi*f1*t)+sin(2*pi*f2*t)+sin(2*pi*f3*t)+sin(2*pi*f4*t);
 
subplot(3,1,1);
 
plot(t,x);
 
xlabel('Time'); ylabel('Amplitude'); title('Generated Signal');
 
1=nextpow2(length(t));
 
11=2^1;
 
y = fft(x, 11)
 
y * 1 = y((1/11) / 2) ;
 
xaxis = linspace(0, fs/2, 11/2);
 
subplot(3,1,2);
 
plot(xaxis, abs(y1));
 
xlabel('Frequency'); ylabel('Magnitude'); title('Frequency Spectrum');
 
[ba]=fir1(N, wc, 'low', hamming (N+1));
 
z=filter(b,a,x);
 
z1=fft(z,11);
 
z * 2 = z * 1((1/11) / 2)
 
subplot(3,1,3);
 
plot(xaxis, abs (z2));
 
xlabel('Frequency'); ylabel('Magnitude'); title('Frequency Spectrum LPF');
 
EXP 5A: INFINITE IMPULSE RESPONSE FILTERS
AIM: To design butterworth filter using bilinear transform and impulse invariant methods
ALGORITHM: 1. Calculate the required parameters for designing analog butterworth filter 2. Design analog butterworth filter using ‘buttord’ and ‘butter’ 3. Use Bilinear Transform and Impulse Invariant to get digital IIR filter
 
clc;
clear all;
close all;
 
DelP = 0.6;
DelS = 0.1;
wp = 0.2 * pi;
ws = 0.5 * pi;
T = 1;
 
Ap = -20 * log10(DelP);
As = -20 * log10(DelS);
 
OmegaP = (2 / T) * tan(wp / 2);
OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = buttord(OmegaP, OmegaS, Ap, As, 's');
[z, p, k] = butter(n, wn, 'low', 's');
[num_s, den_s] = zp2tf(z, p, k);
 
tf(num_s, den_s);
 
[num_z, den_z] = bilinear(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
clc;
clear all;
close all;
 
DelP = 0.6;
DelS = 0.1;
wp = 0.2 * pi;
ws = 0.6 * pi;
T = 1;
 
Ap = -20 * log10(DelP);
As = 20 * log10(DelS);
 
OmegaP = (2 / T) * tan(wp / 2);
OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = buttord(OmegaP, OmegaS, Ap, As, 's');
[num_s, den_s] = butter(n, wn, 's');
tf(num_s, den_s);
 
[num_z, den_z] = impinvar(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
 
EXP 5B: INFINITE IMPULSE RESPONSE FILTERS
AIM: To design Chebyshev Filter using bilinear transform and impulse invariant methods
 ALGORITHM: 1. Calculate the required parameters for designing analog chebyshev-1 filter 2. Design analog chebyshev filter using ‘cheb1ord’ and ‘cheby1’ 3. Use Bilinear Transform and Impulse Invariant to get digital IIR filter
 
clc; clear all; close all;
 
DelP=0.8; DelS=0.2; wp=0.2*pi; ws=0.32*pi; T = 1;
Ap = -20 * log10(DelP); As = 20 * log10(DelS);
OmegaP = (2 / T) * tan(wp / 2); OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = cheb1ord(OmegaP, OmegaS, Ap, As, 's');
[num_s, den_s] = cheby1(n, Ap, wn, 's');
tf(num_s, den_s);
 
[num_z, den_z] = impinvar(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
clc; clear all; close all;
 
DelP=0.8; DelS=0.2; wp=0.2*pi; ws=0.32*pi; T = 1;
Ap = -20 * log10(DelP); As = 20 * log10(DelS);
OmegaP = (2 / T) * tan(wp / 2); OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = cheb1ord(OmegaP, OmegaS, Ap, As, 's');
[num_s, den_s] = cheby1(n, Ap, wn, 's');
tf(num_s, den_s);
 
[num_z, den_z] = bilinear(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
EXP 6A : PROTOYPE FILTERS
AIM: To design lowpass and bandstop Butterworth filters using prototype filter
ALGORITHM: 1. Get zeroes and poles from given specification 2. Calculate transfer function for prototype filter 3. Prototype conversion 4. Frequency Translation 5. Plot the waveforms
 
order = 3;
[z, p, k] = buttap(order);
[n, d] = zp2tf(z, p, k);
 
fs = 1000;
fc = 150;
wc = 2 * pi * fc;
 
[nhp, dhp] = lp2hp(n, d, wc);
[nd, dd] = bilinear(nhp, dhp, fs);
 
f1 = 50; f2 = 150; f3 = 300;
t = (0:1/fs:1);
x = sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t) + sin(2 * pi * f3 * t);
 
subplot(3, 1, 1);
plot(t, x);
ylabel('Amplitude');
xlabel('Time (s)');
title('Input Signal');
 
N = length(x);
NFFT = 2^nextpow2(N);
y = fft(x, NFFT);
f_axis = linspace(0, fs/2, NFFT/2+1);
 
subplot(3, 1, 2);
plot(f_axis, abs(y(1:NFFT/2+1)));
ylabel('Magnitude');
xlabel('Frequency (Hz)');
title('FFT Spectrum of Input Signal');
 
z = filter(nd, dd, x);
z_fft = fft(z, NFFT);
 
subplot(3, 1, 3);
plot(f_axis, abs(z_fft(1:NFFT/2+1)));
ylabel('Magnitude');
xlabel('Frequency (Hz)');
title('FFT Spectrum of Filtered Signal');
 
 
EXP 6B : PROTOYPE FILTERS
AIM: To design highpass and bandpass Chebyshev filters using prototype filter
ALGORITHM: 1. Get zeroes and poles from given specification 2. Calculate transfer function for prototype filter 3. Prototype conversion 4. Frequency Translation 5. Plot the waveform
 
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
 
 
EXP 7A: DENOISING IMAGE AND SPEECH SIGNALS
AIM: To add noise to a speech signal and and denoise it using filters
 ALGORITHM: 1. Add noise using ‘randn’ to a signal 2. Remove the noise using filters like Gaussian , Moving Average ,etc
 
clc; clear all; close all;
x = cumsum(randn(1, 1000));
plot(x);
g = fspecial('gaussian', [1, 100], 12);
figure;
plot(g);
y = conv(x, g);
figure;
plot(x, 'b');
hold on;
plot(y, 'r', 'linewidth', 2);
legend('noisy', 'noise removal');
title('noise reduction of ld sig');
 
clc;
clear all;
close all;
t = 0:0.11:20;
x = sin(t);
n = randn(1, length(x));
x = x + n;
num = (1/3) * [1 1 1];
den = [1];
y = filter(num, den, x);
figure;
plot(x, 'b');
hold on;
plot(y, 'r', 'linewidth', 3);
legend('Noisy signal', 'Filtered signal');
title('moving average filter for noise reduction');
 
EXP 7B: DENOISING IMAGE AND SPEECH SIGNALS
AIM: To add noise to an image and denoise it using filters and calculate MSE, SNR and PSNR
ALGORITHM: 1. Add noise using ‘imnoise()’ to a signal 2. Remove the noise using filters like Gaussian , Median ,etc 3. Calculate MSE, SNR, and PSNR
image=imread('cameraman.tif');
 
[M, N]=size(image);
 
subplot(2,3,1); imshow(image); title('Original Image')
 
noise1=imnoise(image, "salt & pepper", 0.3);
 
subplot(2,3,2); imshow(noise1), title('Salt & Pepper 30%')
 
noise2=imnoise(image, "gaussian", 0.5);
 
subplot(2,3,3); imshow(noise2), title('Gaussian 50%')
 
[peaksnr1, snr1]=psnr(image, noise1);
 
[peaksnr2, snr2]=psnr(image, noise2);
 
mse1=immse(image, noise1);
 
mse2=immse(image, noise2);
 
filtered1=medfilt2(noise1);
 
subplot(2,3,4); imshow(filtered1), title(' Median Filter')
 
filtered2=filter2(fspecial('average', 3), noise2)/255;
 
subplot(2,3,5); imshow(filtered2), title(' Averaging Filter')
 
 
AIM: To generate elementary signals
ALGORITHM: 1. Assign sampling frequency and duration of time signal 2. Generate unit impulse, unit step, ramp and sinusoidal signal 3. Plot them continuous time signals and discrete time signals
 
clc;
clear all;
close all;
fs=10;
t=0:1/fs:1;
n=-10:10;
 
% unit impulse
x1=[zeros(1,10),ones(1,1),zeros(1,10)];
subplot(4,2,1);plot(n,x1);
xlabel('Time');ylabel('Amplitude');title('Continuous Unit Impulse');
subplot(4,2,2);stem(n,x1);
xlabel('Index');ylabel('Amplitude');title('Discrete Unit Impulse');
 
% unit step
x2=[zeros(1,10),ones(1,11)];
subplot(4,2,3);plot(n,x2);
xlabel('Time');ylabel('Amplitude');title('Continuous Unit Step');
subplot(4,2,4);stem(n,x2);
xlabel('Index');ylabel('Amplitude');title('Discrete Unit Step');
 
% unit ramp
x3=x2.*n;
subplot(4,2,5);plot(n,x3);
xlabel('Time');ylabel('Amplitude');title('Continuous Ramp');
subplot(4,2,6);stem(n,x3);
xlabel('Index');ylabel('Amplitude');title('Discrete Ramp');
 
% sinusoidal signal
x4=sin(2*pi*t);
subplot(4,2,7);plot(t,x4);
xlabel('Time');ylabel('Amplitude');title('Continuous Sine');
subplot(4,2,8);stem(t,x4);
xlabel('Index');ylabel('Amplitude');title('Discrete Sine');
 
AIM: To get unit response, step response and frequency response for a given transfer function
ALGORITHM: 1. Let y[n]-y[n-1]=x[n]+x[n-1]-x[n-2], find the transfer function 2. Plot impulse response and step response 3. Plot frequency response in s-domain and z-domain
clc;
clear all;
close all;
fs=10;
 
% if y[n]-y[n-1]=x[n]+x[n-1]-x[n-2],
% Y(jw)-e^-jw*Y(jw)=X(jw)+e^-jw*X(jw)-e^-2jw*X(jw)
% Transfer function H(w)=Y(w)/X(w)
% H(w)=(1+(e^-jw)-(e^-2jw))/(1-(e^-jw)),
% numerator coeff=[1 1 -1], denominator coeff=[1 -1]
 
num=[1 1 -1];
den=[1 -1];
subplot(3,1,1);impz(num,den);
subplot(3,1,2);stepz(num,den);
subplot(3,1,3);zplane(num,den);
 
figure(2);freqz(num,den);
figure(3);freqs(num,den);
 
 
EXP 2A: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To perform shifting and scaling operations in a signal and analyse the process in time domain
ALGORITHM: 1. Generate a signal 2. Perform shifting, scaling and reversing operations 3. Plot the signal
 
f = 5;   	% Frequency of the sine wave
fs = 100;	% Sampling frequency
t = 0:1/fs:1; % Time vector from 0 to 1 second with sampling interval of 1/fs
x = sin(2 * pi * f * t); % Sine wave signal
 
subplot(4, 1, 1)
plot(t, x)
xlabel("Time")
ylabel("Amplitude")
title("Original wave")
 
subplot(4, 1, 2)
plot(t / 2, x)
xlabel("Time")
ylabel("Amplitude")
title("Time Scaling")
 
subplot(4, 1, 3)
plot(t + 1, x)
xlabel("Time")
ylabel("Amplitude")
title("Time Shifting")
 
subplot(4, 1, 4)
plot(-t, x)
xlabel("Time")
ylabel("Amplitude")
title("Time Reversal")
 
 
EXP 2B: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To acquire two speech signals by live recording and perform convolution
ALGORITHM: 1. Assign the sampling frequency, channel and bits 2. Record the signals using a microphone 3. Convolute the signals 4. Plot the recorded signal and their convolution
 
rec = audiorecorder(44100, 16, 1);
recordblocking(rec, 5);          	
y1 = getaudiodata(rec);          	
subplot(3, 1, 1); plot(y1);      	
title('Speech Signal 1');
 
rec = audiorecorder(44100, 16, 1);  
recordblocking(rec, 5);          	
y2 = getaudiodata(rec);          	
subplot(3, 1, 2); plot(y2);      	
title('Speech Signal 2');
 
y = conv(y1, y2, 'full');        	
subplot(3, 1, 3); plot(y);       	
title('Convolution');
 
 
EXP 2C: IMAGE AND SOUND SIGNAL PROCESSING
 
AIM: To acquire stored audio data in ‘.wav’ format and perform convolution and correlation
ALGORITHM: 1. Acquire the ‘.wav’ signals using ‘audioread()’ function 2. Plot the acquired signal and perform convolution in time and frequency domain operations 3. Find correlation between the signals 4. Plot the signals
clc;
clear all;
close all;
[sig1, fs1] = audioread('s1.wav');
t = (1:length(sig1)) / fs1;
subplot(3,1,1);
plot(t, sig1);
xlabel('seconds');
ylabel('Relative Signal Strength');
title('song');
[sig2, fs2] = audioread('s2.wav');
x = (1:length(sig1)) * 0;
x1 = (1:length(sig1)) * 0;
ex = (1:length(sig1)) * 0;
ex1 = (1:length(sig1)) * 0;
t1 = (1:length(sig1)) / fs1;
subplot(3,1,2);
plot(t1, sig2);
xlabel('seconds');
ylabel('Relative Signal Strength');
title('speech');
c = conv(sig1, sig2, 'same');
sound(c, fs1);
t = (1:length(sig1)) / fs1;
subplot(3,1,3);
plot(t, c);
xlabel('seconds');
ylabel('Relative Signal Strength');
title('convolved Signal');
 
clc; clear all; close all;
[y, fs] = audioread('s1.wav');
timelag = 1;
delta = round(fs * timelag);
alpha = 0.3;
orig = [y; zeros(delta,2)];
echo = [zeros(delta,2); y] * alpha;
Rx = orig + echo;
t = (0:length(Rx)-1) / fs;
subplot(3,1,1);
plot(t, [orig echo]);
title('Original + Echo');
subplot(3,1,2); plot(t, Rx);
title('Total');
xlabel('Time (s)');
[r, lags] = xcorr(Rx, 'unbiased');
r = r(lags >= 0); lags = lags(lags >= 0);
subplot(3,1,3); plot(lags / fs, r);
title('Correlation');
ylabel('Correlation');
xlabel('Lag (s)');
figure();
[peak_value_loc] = findpeaks(r, lags, 'MinPeakHeight', 0.005);
den = [1 zeros(1,loc(1)-1) alpha];
filtered = filter(1, den, Rx);
subplot(3,1,1); plot(t, Rx);
title('Echo Added');
subplot(3,1,2); plot(t, orig);
title('original');
subplot(3,1,3);
plot(t, filtered);
title('Filtered'); xlabel('Time (s)');
 
clc; clear all; close all;
[sig1, fs] = audioread('s1.wav');
f = (1:length(sig1)) / fs;
X = fftshift(fft(sig1));
Xmag = abs(X);
subplot(3,1,1);
plot(f, Xmag);
xlabel('Frequency');
ylabel('Relative Signal Strength');
title('song');
[sig2, fs] = audioread('s2.wav');
x = (1:length(sig1)) * 0;
x1 = (1:length(sig1)) * 0;
fx = (1:length(sig1)) / fs;
X1 = fftshift(fft(x));
X1mag = abs(X1);
subplot(3,1,2);
plot(fx, X1mag);
xlabel('Frequency');
ylabel('Relative Signal Strength');
title('Speech Signal');
soundsc(sig1, fs);
soundsc(sig2, fs, 'same');
X2 = fftshift(fft(x2));
X2mag = abs(X2);
f2 = (1:length(sig1)) / fs;
subplot(3,1,3);
plot(f2, X2mag);
xlabel('Frequency');
ylabel('Relative Signal Strength');
title('convolved Signal');
 
EXP 2D: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To acquire stored audio data in ‘.wav’ format and perform shifting and scaling operations
 
[y,fs]=audioread(‘.wav’)
Dt=1/fs;
t=0:dt:(length(y)*dt)-dt;
subplot(4,1,1);
plot(t,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘sound signal’);
subplot(4,1,2);
plot(t+5,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘shifted signal’);
subplot(4,1,3);
plot(2*t,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘scaled signal’);
subplot(4,1,4);
plot(-t,y);
xlabel(‘seconds’);ylabel(‘amplitude’);title(‘reversed signal’);
 
EXP 2C: IMAGE AND SOUND SIGNAL PROCESSING
AIM: To read an image and apply Discrete Cosine Transform and reconstruct the image using Inverse DCT.
 
 
Clc;
Clear all;
Close all;
a=imread(‘cameraman.tif’);
[M,N]=size(a);
Subplot(2,3,1);
Imshow(a);
B=dct2(a);
Subplot(2,3,2);
Imshow(abs(b),[]);
Subplot(2,3,3);
e=idct2(b);
subplot(2,3,3);
imshow(e,[]);
 
 
EXP 3A: FOURIER TRANSFORM
AIM: To apply Fourier Transform to a signal and reconstruct the signal using Inverse Fourier Transform
 
%CTFS
t0=0; T=3;           	% Set the initial time and period
w=2*pi/T;            	% Angular frequency
syms t               	% Define symbolic variable t
x=exp(-t);           	% Define the function x(t) = exp(-t)
 
subplot(3,1,1);
fplot(x,[t0 t0+T]);  	% Plot the original function over [t0, t0+T]
xlabel('Amplitude'); ylabel('time'); title('Input');
 
% Calculate CTFS coefficients with k = -10 to 10
for k=-10:10
	c(k+11)=(1/T)*int(x*exp(-1i*k*w*t), t, t0, t0+T);  % Fourier coefficient calculation
end
 
% Generate complex exponentials for k = -10 to 10
for k=-10:10
	cx(k+11)=exp(1i*k*w*t);                        	% e^(j*k*w*t)
end
 
% Reconstruct signal using k = -10 to 10
xx=sum(c.*cx);                                     	% Summation of terms for signal reconstruction
subplot(3,1,2);
fplot(xx, [t0 t0+T]);                                  % Plot reconstructed signal with k = 10
xlabel('Amplitude'); ylabel('time'); title('Reconstructed with k=10');
 
% Calculate CTFS coefficients with k = -50 to 50
for k=-50:50
	c(k+51)=(1/T)*int(x*exp(-1i*k*w*t), t, t0, t0+T);  % Fourier coefficient calculation
end
 
% Generate complex exponentials for k = -50 to 50
for k=-50:50
	cx(k+51)=exp(1i*k*w*t);                        	% e^(j*k*w*t)
end
 
% Reconstruct signal using k = -50 to 50
xx=sum(c.*cx);                                     	% Summation of terms for signal reconstruction
subplot(3,1,3);
fplot(xx, [t0 t0+T]);                                  % Plot reconstructed signal with k = 50
xlabel('Amplitude'); ylabel('time'); title('Reconstructed with k=50');
 
%DTFS
clc;
clear all;
close all;
 
syms w;
x=0.8.^n;           	% Define signal x(n) = (0.8)^n
n=-10:10;           	% Define range for n
subplot(2,1,1); stem(n,x);
xlabel('n'); ylabel('x(n)'); title('Original Signal');
 
% Calculate DTFS coefficients for k = -10 to 10
for k=-10:10
    c(k+11)=(1/length(n))*sum(x.*exp(-j*k*w*n));   % DTFS coefficient calculation
end
 
% Reconstruct signal using DTFS
x1=sum(c.*exp(j*k*w*n));   % Summation of terms for signal reconstruction
subplot(2,1,2); stem(n,x1);
xlabel('n'); ylabel('x1(n)'); title('Reconstructed Signal');
 
%DTFT
clc;
clear all;
close all;
 
syms w;
x=0.8.^n;           	% Define signal x(n) = (0.8)^n
n=-10:10;           	% Define range for n
subplot(3,1,1); stem(n,x);
xlabel('n'); ylabel('x(n)'); title('Original Signal');
 
% Compute DTFT
X=sum(x.*exp(-1i*w*n));
subplot(3,1,2);
ezplot(abs(X), [-10, 10]);
xlabel('frequency'); ylabel('|X(w)|'); title('Magnitude Response');
 
% Inverse DTFT to reconstruct the signal
x1=(1/(2*pi))*int(X.*exp(1i*w*n), w, -pi, pi);
subplot(3,1,3); stem(n,x1);
xlabel('n'); ylabel('x(n)'); title('Reconstructed Signal');
 
%CTFT
t0=0; T=3;
w=2*pi/T;
syms t w;
x=exp(-t^2);           	% Define signal x(t) = exp(-t^2)
 
subplot(3,1,1);
fplot(x, [-t0-T t0+T]);
ylabel('Amplitude'); xlabel('time'); title('Input');
 
% Compute CTFT
X=int(x*exp(-1i*w*t), t, -inf, inf);
subplot(3,1,2);
fplot(X, [-pi pi]);
ylabel('Magnitude'); xlabel('Frequency'); title('Magnitude Response');
 
% Inverse CTFT to reconstruct the signal
x1=(1/(2*pi))*int(X*exp(1i*w*t), w, -inf, inf);
subplot(3,1,3); fplot(x1, [-t0-T t0+T]);
ylabel('Amplitude'); xlabel('time'); title('Reconstructed');
 
EXP 3B: FOURIER TRANSFORM
AIM: To denoise a signal using DFT and IDFT
 
N = 256;
n = 0:N-1;
w = 2*pi/N;
x = 7*cos(3*n*w) + 13*sin(6*n*w);
subplot(321); plot(x); title('original signal');
xlabel('time index, n'); ylabel('x(n)');
Xk = fft(x);
subplot(322); stem(abs(Xk)); title('DFT peaks at k = 3 and 6');
xlabel('frequency index, k'); ylabel('X(k)');
 
xn = x + 10*randn(1,N);
subplot(323); plot(xn); title('original signal+noise');
xlabel('time index, n'); ylabel('xn(n)');
Xnk = fft(xn);
subplot(324); stem(abs(Xnk)); title('DFT peaks at k = 3 and 6');
xlabel('frequency index, k'); ylabel('Xn(k)');
 
iz = find(abs(Xnk)/N*2 < 4);
Xnk(iz) = zeros(size(iz));
subplot(325); stem(abs(Xnk));
title('DFT peaks at k = 3 and 6 are to be retained');
xlabel('frequency index, k'); ylabel('Xn(k)');
xr = ifft(Xnk);
subplot(326); plot(real(xr)); title('recovered signal');
xlabel('time index, n'); ylabel('xr(n)');
 
EXP 3C: FOURIER TRANSFORM
AIM: To apply DFT to a sine waveform with different cycles and a DC component
ALGORITHM: 1. Generate a sine signal with dc component and apply DFT to the signal 2. Add DC component and apply DFT
 
N = 64;
n = 0:N-1;
w = 2*pi/N;
 
x1 = sin(n*w);
subplot(321); stem(x1);
title('input signal, one cycle');
xlabel('time index, n \rightarrow'); ylabel('x1(n) \rightarrow');
X1k = fft(x1);
subplot(322); stem(abs(X1k));
title('DFT of input signal, non-zero at k=1 and k=63');
xlabel('frequency index, k \rightarrow'); ylabel('X1(k) \rightarrow');
 
x2 = sin(2*n*w);
subplot(323); stem(x2);
title('input signal, two cycles');
xlabel('time index, n \rightarrow'); ylabel('x2(n) \rightarrow');
X2k = fft(x2);
subplot(324); stem(abs(X2k));
title('DFT of input signal, non-zero at k=2 and k=62');
xlabel('frequency index, k \rightarrow'); ylabel('X2(k) \rightarrow');
 
x3 = sin(7*n*w);
subplot(325); stem(x3);
title('input signal, seven cycles');
xlabel('time index, n \rightarrow'); ylabel('x3(n) \rightarrow');
X3k = fft(x3);
subplot(326); stem(abs(X3k));
title('DFT of input signal, non-zero at k=7 and k=57');
xlabel('frequency index, k \rightarrow'); ylabel('X3(k) \rightarrow');
 
N = 64;
n = 0:N-1;
w = 2*pi/N;
 
x1 = sin(n*w) + 5;
subplot(221); stem(x1);
title('one cycle added with 5 dc component');
xlabel('time index, n \rightarrow'); ylabel('x1(n) \rightarrow');
X1k = fft(x1);
subplot(222); stem(abs(X1k));
title('DFT of input signal, non-zero at k=1,63 with DC component 320');
xlabel('frequency index, k \rightarrow'); ylabel('X1(k) \rightarrow');
 
x2 = sin(2*n*w) + (4*sin(7*n*w)) + 3;
subplot(223); stem(x2);
title('input signal, sin(2*n*w) + (4*sin(7*n*w)) + 3');
xlabel('time index, n \rightarrow'); ylabel('x2(n) \rightarrow');
X2k = fft(x2);
subplot(224); stem(abs(X2k));
title('DFT of input, non-zero at k=2&62,7&57 with DC component 3x64=192');
xlabel('frequency index, k \rightarrow'); ylabel('X2(k) \rightarrow');
 
 
EXP 4A: FINITE IMPULSE RESPONSE FILTERS
AIM: To design and test a filter using frequency sampling method
ALGORITHM 1. Design a FIR Filter using ‘fir2’ 2. Generate a signal and plot the fft spectrum 3. Apply filter to the signal and plot the fft spectrum
 
 
fs=1000;
t=0:1/fs:1;
f1=100;f2=200;
f3=300;f4=400;
 
x=sin(2*pi*f1*t)+sin(2*pi*f2*t)+sin(2*pi*f3*t)+sin(2*pi*f4*t);
subplot(3,1,1);
plot(t,x);
xlabel('Time');ylabel('Amplitude');title('Generated Signal');
l=nextpow2(length(t));
l1=2^l;
y=fft(x,l1);
y1=y(1:l1/2);
xaxis = linspace(0, fs/2, l1/2);
subplot(3,1,2);
plot(xaxis, abs(y1));
xlabel('Frequency');ylabel('Magnitude');title('Frequency Spectrum');
 
w1=f1/(fs/2);w2=f2/(fs/2);
w3=f3/(fs/2);w4=f4/(fs/2);
 
f2=[0 w2 w2 1];
m2=[1 1 0 0];
b2=fir2(30,f2,m2);a2=1;
zL=filter(b2,a2,x);
zL1=fft(zL,l1);
zL2=zL1(1:l1/2);
subplot(3,1,3);
plot(xaxis,abs(zL2));
xlabel('Frequency');ylabel('Magnitude');title('Frequency Spectrum LPF');
EXP 4B: FINITE IMPULSE RESPONSE FILTERS
AIM: To design and test a filter using windowing method
ALGORITHM 1. Design a FIR Filter using ‘fir1’ 2. Generate a signal and plot the fft spectrum 3. Apply filter to the signal and plot the fft spectrum
 
clc;
 
clear all;
 
close all;
 
N = 25
 
fs = 1000 fc = 100
 
wc=2*fc/fs;
 
t = 0 1/fs:1;
 
f * 1 = 50 f * 2 = 200 f * 3 = 300 f * 4 = 400
 
x=sin(2*pi*f1*t)+sin(2*pi*f2*t)+sin(2*pi*f3*t)+sin(2*pi*f4*t);
 
subplot(3,1,1);
 
plot(t,x);
 
xlabel('Time'); ylabel('Amplitude'); title('Generated Signal');
 
1=nextpow2(length(t));
 
11=2^1;
 
y = fft(x, 11)
 
y * 1 = y((1/11) / 2) ;
 
xaxis = linspace(0, fs/2, 11/2);
 
subplot(3,1,2);
 
plot(xaxis, abs(y1));
 
xlabel('Frequency'); ylabel('Magnitude'); title('Frequency Spectrum');
 
[ba]=fir1(N, wc, 'low', hamming (N+1));
 
z=filter(b,a,x);
 
z1=fft(z,11);
 
z * 2 = z * 1((1/11) / 2)
 
subplot(3,1,3);
 
plot(xaxis, abs (z2));
 
xlabel('Frequency'); ylabel('Magnitude'); title('Frequency Spectrum LPF');
 
EXP 5A: INFINITE IMPULSE RESPONSE FILTERS
AIM: To design butterworth filter using bilinear transform and impulse invariant methods
ALGORITHM: 1. Calculate the required parameters for designing analog butterworth filter 2. Design analog butterworth filter using ‘buttord’ and ‘butter’ 3. Use Bilinear Transform and Impulse Invariant to get digital IIR filter
 
clc;
clear all;
close all;
 
DelP = 0.6;
DelS = 0.1;
wp = 0.2 * pi;
ws = 0.5 * pi;
T = 1;
 
Ap = -20 * log10(DelP);
As = -20 * log10(DelS);
 
OmegaP = (2 / T) * tan(wp / 2);
OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = buttord(OmegaP, OmegaS, Ap, As, 's');
[z, p, k] = butter(n, wn, 'low', 's');
[num_s, den_s] = zp2tf(z, p, k);
 
tf(num_s, den_s);
 
[num_z, den_z] = bilinear(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
clc;
clear all;
close all;
 
DelP = 0.6;
DelS = 0.1;
wp = 0.2 * pi;
ws = 0.6 * pi;
T = 1;
 
Ap = -20 * log10(DelP);
As = 20 * log10(DelS);
 
OmegaP = (2 / T) * tan(wp / 2);
OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = buttord(OmegaP, OmegaS, Ap, As, 's');
[num_s, den_s] = butter(n, wn, 's');
tf(num_s, den_s);
 
[num_z, den_z] = impinvar(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
 
EXP 5B: INFINITE IMPULSE RESPONSE FILTERS
AIM: To design Chebyshev Filter using bilinear transform and impulse invariant methods
 ALGORITHM: 1. Calculate the required parameters for designing analog chebyshev-1 filter 2. Design analog chebyshev filter using ‘cheb1ord’ and ‘cheby1’ 3. Use Bilinear Transform and Impulse Invariant to get digital IIR filter
 
clc; clear all; close all;
 
DelP=0.8; DelS=0.2; wp=0.2*pi; ws=0.32*pi; T = 1;
Ap = -20 * log10(DelP); As = 20 * log10(DelS);
OmegaP = (2 / T) * tan(wp / 2); OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = cheb1ord(OmegaP, OmegaS, Ap, As, 's');
[num_s, den_s] = cheby1(n, Ap, wn, 's');
tf(num_s, den_s);
 
[num_z, den_z] = impinvar(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
clc; clear all; close all;
 
DelP=0.8; DelS=0.2; wp=0.2*pi; ws=0.32*pi; T = 1;
Ap = -20 * log10(DelP); As = 20 * log10(DelS);
OmegaP = (2 / T) * tan(wp / 2); OmegaS = (2 / T) * tan(ws / 2);
 
[n, wn] = cheb1ord(OmegaP, OmegaS, Ap, As, 's');
[num_s, den_s] = cheby1(n, Ap, wn, 's');
tf(num_s, den_s);
 
[num_z, den_z] = bilinear(num_s, den_s, 1 / T);
tf(num_z, den_z, T);
freqz(num_z, den_z);
 
EXP 6A : PROTOYPE FILTERS
AIM: To design lowpass and bandstop Butterworth filters using prototype filter
ALGORITHM: 1. Get zeroes and poles from given specification 2. Calculate transfer function for prototype filter 3. Prototype conversion 4. Frequency Translation 5. Plot the waveforms
 
order = 3;
[z, p, k] = buttap(order);
[n, d] = zp2tf(z, p, k);
 
fs = 1000;
fc = 150;
wc = 2 * pi * fc;
 
[nhp, dhp] = lp2hp(n, d, wc);
[nd, dd] = bilinear(nhp, dhp, fs);
 
f1 = 50; f2 = 150; f3 = 300;
t = (0:1/fs:1);
x = sin(2 * pi * f1 * t) + sin(2 * pi * f2 * t) + sin(2 * pi * f3 * t);
 
subplot(3, 1, 1);
plot(t, x);
ylabel('Amplitude');
xlabel('Time (s)');
title('Input Signal');
 
N = length(x);
NFFT = 2^nextpow2(N);
y = fft(x, NFFT);
f_axis = linspace(0, fs/2, NFFT/2+1);
 
subplot(3, 1, 2);
plot(f_axis, abs(y(1:NFFT/2+1)));
ylabel('Magnitude');
xlabel('Frequency (Hz)');
title('FFT Spectrum of Input Signal');
 
z = filter(nd, dd, x);
z_fft = fft(z, NFFT);
 
subplot(3, 1, 3);
plot(f_axis, abs(z_fft(1:NFFT/2+1)));
ylabel('Magnitude');
xlabel('Frequency (Hz)');
title('FFT Spectrum of Filtered Signal');
 
 
EXP 6B : PROTOYPE FILTERS
AIM: To design highpass and bandpass Chebyshev filters using prototype filter
ALGORITHM: 1. Get zeroes and poles from given specification 2. Calculate transfer function for prototype filter 3. Prototype conversion 4. Frequency Translation 5. Plot the waveform
 
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
 
 
EXP 7A: DENOISING IMAGE AND SPEECH SIGNALS
AIM: To add noise to a speech signal and and denoise it using filters
 ALGORITHM: 1. Add noise using ‘randn’ to a signal 2. Remove the noise using filters like Gaussian , Moving Average ,etc
 
clc; clear all; close all;
x = cumsum(randn(1, 1000));
plot(x);
g = fspecial('gaussian', [1, 100], 12);
figure;
plot(g);
y = conv(x, g);
figure;
plot(x, 'b');
hold on;
plot(y, 'r', 'linewidth', 2);
legend('noisy', 'noise removal');
title('noise reduction of ld sig');
 
clc;
clear all;
close all;
t = 0:0.11:20;
x = sin(t);
n = randn(1, length(x));
x = x + n;
num = (1/3) * [1 1 1];
den = [1];
y = filter(num, den, x);
figure;
plot(x, 'b');
hold on;
plot(y, 'r', 'linewidth', 3);
legend('Noisy signal', 'Filtered signal');
title('moving average filter for noise reduction');
 
EXP 7B: DENOISING IMAGE AND SPEECH SIGNALS
AIM: To add noise to an image and denoise it using filters and calculate MSE, SNR and PSNR
ALGORITHM: 1. Add noise using ‘imnoise()’ to a signal 2. Remove the noise using filters like Gaussian , Median ,etc 3. Calculate MSE, SNR, and PSNR
image=imread('cameraman.tif');
 
[M, N]=size(image);
 
subplot(2,3,1); imshow(image); title('Original Image')
 
noise1=imnoise(image, "salt & pepper", 0.3);
 
subplot(2,3,2); imshow(noise1), title('Salt & Pepper 30%')
 
noise2=imnoise(image, "gaussian", 0.5);
 
subplot(2,3,3); imshow(noise2), title('Gaussian 50%')
 
[peaksnr1, snr1]=psnr(image, noise1);
 
[peaksnr2, snr2]=psnr(image, noise2);
 
mse1=immse(image, noise1);
 
mse2=immse(image, noise2);
 
filtered1=medfilt2(noise1);
 
subplot(2,3,4); imshow(filtered1), title(' Median Filter')
 
filtered2=filter2(fspecial('average', 3), noise2)/255;
 
subplot(2,3,5); imshow(filtered2), title(' Averaging Filter')
 
 



