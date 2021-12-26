%% 读取语音信息 
[myAudio, FsA] = audioread('./../media/readAudio.m4a');

%% 播放语音，输出语音的时域波形
N = length(myAudio); %求取抽样点数
t = (0:N-1)/FsA; %每一个采样点的实际时间

myAudio_fft = fft(myAudio);%对信号进行傅里叶变换
myAudio_fft_abs_half = abs(myAudio_fft(1:round(N/2))); %取绝对值与fft的一半

% round四舍五入为最近的小数或整数
f = FsA/N*(0:round(N/2)-1); %显示实际频点的一半 
% ？？？？？？？？？？？？？？？？

% subplots是设置子图的，这句话表示大图含有2行1列共2个子图，正在绘制的是第一个
subplot(211);
plot(t,myAudio,'g');%绘制时域波形
xlabel('Time/(s)');
title('信号的波形');
grid;

subplot(212);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('信号的频谱');
grid;

sound(myAudio, 1 * FsA); % 播放原始声音

%% 对信号卷积 -- 相当于再再另一个系统中发出原始信号

% 读取冲击信号
[impulse,FsI] = audioread('./../media/impluseInClassroom.m4a');

convMyAudio = conv(impulse, myAudio); %对两信号卷积
sound(convMyAudio, FsA);

%% 滤波 -- 使用简易 1维数字滤波器 -- 使用filter

% 构建滤波器参数
a = 1;
windowSize = 10;
b = (1/windowSize)*ones(1,windowSize);
% y = filter(b,a,x);

% 使用滤波器对时域信号进行滤波
fiMyAudio = filter(b, a, myAudio);
fiMyAudio = fiMyAudio'; %转置

%播放滤波后的声音信号
sound(fiMyAudio, FsA);

%对比声音信号
subplot(211);
plot(t, myAudio,'g');%绘制时域波形，放大来看，很明显毛刺被去掉了
xlabel('Time/(s)');
title('原始的波形');
grid;

subplot(212);
plot(t, fiMyAudio);
xlabel('Frequency/(Hz)');
title('滤波后的波形');
grid;

%% 信号分析

myAudio_fft_right = myAudio_fft;

for i = 1:size(myAudio_fft_right)
    if i < 10000 
        myAudio_fft_right(i) = 0;
    end
    if i > 50000 
        myAudio_fft_right(i) = 0;
    end
end

%变换数据，方便绘图
myAudio_fft_toAna_half = abs(myAudio_fft_right(1:round(N/2))); 

%画图对比两信号频域信息
subplot(211);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('原始信号的频谱');
grid;
subplot(212);
plot(f,myAudio_fft_toAna_half);
xlabel('Frequency/(Hz)');
title('分析后信号的频谱');
grid;

% 播放信号分析后音频
sound(abs(ifft(myAudio_fft_right)), FsA);

%% 变声 -- 使频谱统一右移

% myAudio_fft_right = myAudio_fft;
% % myAudio_fft_right = 0;
myAudio_fft_right = zeros(size(myAudio_fft));%4*5*3的数组，数组中每个元素都为0

for i = 1:size(myAudio_fft_right)
    myAudio_fft_right(i + 30000) = myAudio_fft(i);
end

plot(abs(myAudio_fft_right));

back = ifft(myAudio_fft_right);

sound(abs(back), FsA);


%% 滤波 -- 1 -- add noise

% n的序列
Nseq = 0:1:N-1;

% 高频噪声
noise = ( cos(6000/ FsA * pi * Nseq)+ cos(6500/ FsA * pi * Nseq)+ cos(10000/ FsA * pi * N))* 0.5; 

% 原始信号 合成 高频噪声
myAudio_addNoise = myAudio + noise';

sound(myAudio_addNoise, FsA)  % 播放添加了噪音的语音信号

myAudio_addNoise_fft = fft(myAudio_addNoise); %对信号进行傅里叶变换

subplot(211);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('原始信号的频谱');
grid;

subplot(212);
myAudio_addNoise_fft_abs_half = abs(myAudio_addNoise_fft(1:round(N/2)));
plot(f, myAudio_addNoise_fft_abs_half)
xlabel('Frequency/(Hz)');
title('添加噪声后信号的频谱');
grid;

%% 滤波 -- 2 -- design the ButterWorth filter

% 参数给定
fp=800;fs=1300;rs=35;rp=0.5;Fs=44100;

% 预畸变
wp=2*Fs*tan(2*pi*fp/(2*Fs));
ws=2*Fs*tan(2*pi*fs/(2*Fs));

[n,wn]=buttord(wp,ws,rp,rs,'s');
% returns the lowest order
% [n,Wn] = buttord(Wp,Ws,Rp,Rs)
% To design a Butterworth filter, use the output arguments n and Wn as inputs to butter.

[b,a]=butter(n,wn,'s');
% [b,a] = butter(n,Wn) 
% 返回低通模型归一化n阶butterWorth系数

% 去归一化
[num,den]=bilinear(b,a,Fs);
% bi~linear transformation method for analog-to-digital filter conversion
% [numd,dend] = bilinear(num,den,fs) converts the s-domain transfer function specified by numerator num and denominator den to a discrete equivalent.

% 得到系统函数
[h,w]=freqz(num,den,512,Fs);
% Frequency response of digital filter


subplot(3,1,1)
plot(w,abs(h));
xlabel('频率/Hz');ylabel('幅值');
title('巴特沃斯低通滤波器幅度特性'); axis([0,5000,0,1.2]); grid on;

subplot(3,1,2)
plot(w,20*log10(abs(h)));
xlabel('频率/Hz');ylabel('幅值db'); title('巴特沃斯低通滤波器幅度特性db'); axis([0,5000,-90,10]); grid on;

subplot(3,1,3)
plot(w,180/pi*unwrap(angle(h)));
xlabel('频率/Hz');ylabel('相位');
title('巴特沃斯低通滤波器相位特性'); axis([0,5000,-1000,10]) ;grid on; 

%% 滤波 -- 3 -- 滤波操作

x1 = myAudio;
N1=length(x1);
Y1 = fft(x1);

% 进行滤波
myAudio_afterFiltered = filter(num,den,x1);

% 播放滤波后音频
sound(myAudio_afterFiltered,Fs);

%% 滤波 -- 4 -- 滤波后频域

% 求频域，为画图做准备
myAudio_afterFiltered_fft =fft(myAudio_afterFiltered,N1);

% 画图！

subplot(211);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('原始信号的频谱');
grid;

subplot(212);
%取绝对值与fft的一半
myAudio_afterFiltered_fft_half = abs(myAudio_afterFiltered_fft(1:round(N/2))); 
plot(f, myAudio_afterFiltered_fft_half)
xlabel('Frequency/(Hz)');
title('滤波后信号的频谱');
grid;

%% 滤波 -- 5 -- 滤波后时域

% 画图！

subplot(211);
plot(t,myAudio,'g');%绘制时域波形
xlabel('Time/(s)');
title('原始信号的波形');
grid;

subplot(212);
%取绝对值与fft的一半
plot(t,myAudio_afterFiltered,'b');%绘制时域波形
xlabel('Time/(s)');
title('滤波后信号的波形');
grid;






