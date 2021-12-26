%% ��ȡ������Ϣ 
[myAudio, FsA] = audioread('./../media/readAudio.m4a');

%% �������������������ʱ����
N = length(myAudio); %��ȡ��������
t = (0:N-1)/FsA; %ÿһ���������ʵ��ʱ��

myAudio_fft = fft(myAudio);%���źŽ��и���Ҷ�任
myAudio_fft_abs_half = abs(myAudio_fft(1:round(N/2))); %ȡ����ֵ��fft��һ��

% round��������Ϊ�����С��������
f = FsA/N*(0:round(N/2)-1); %��ʾʵ��Ƶ���һ�� 
% ��������������������������������

% subplots��������ͼ�ģ���仰��ʾ��ͼ����2��1�й�2����ͼ�����ڻ��Ƶ��ǵ�һ��
subplot(211);
plot(t,myAudio,'g');%����ʱ����
xlabel('Time/(s)');
title('�źŵĲ���');
grid;

subplot(212);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('�źŵ�Ƶ��');
grid;

sound(myAudio, 1 * FsA); % ����ԭʼ����

%% ���źž�� -- �൱��������һ��ϵͳ�з���ԭʼ�ź�

% ��ȡ����ź�
[impulse,FsI] = audioread('./../media/impluseInClassroom.m4a');

convMyAudio = conv(impulse, myAudio); %�����źž��
sound(convMyAudio, FsA);

%% �˲� -- ʹ�ü��� 1ά�����˲��� -- ʹ��filter

% �����˲�������
a = 1;
windowSize = 10;
b = (1/windowSize)*ones(1,windowSize);
% y = filter(b,a,x);

% ʹ���˲�����ʱ���źŽ����˲�
fiMyAudio = filter(b, a, myAudio);
fiMyAudio = fiMyAudio'; %ת��

%�����˲���������ź�
sound(fiMyAudio, FsA);

%�Ա������ź�
subplot(211);
plot(t, myAudio,'g');%����ʱ���Σ��Ŵ�������������ë�̱�ȥ����
xlabel('Time/(s)');
title('ԭʼ�Ĳ���');
grid;

subplot(212);
plot(t, fiMyAudio);
xlabel('Frequency/(Hz)');
title('�˲���Ĳ���');
grid;

%% �źŷ���

myAudio_fft_right = myAudio_fft;

for i = 1:size(myAudio_fft_right)
    if i < 10000 
        myAudio_fft_right(i) = 0;
    end
    if i > 50000 
        myAudio_fft_right(i) = 0;
    end
end

%�任���ݣ������ͼ
myAudio_fft_toAna_half = abs(myAudio_fft_right(1:round(N/2))); 

%��ͼ�Ա����ź�Ƶ����Ϣ
subplot(211);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('ԭʼ�źŵ�Ƶ��');
grid;
subplot(212);
plot(f,myAudio_fft_toAna_half);
xlabel('Frequency/(Hz)');
title('�������źŵ�Ƶ��');
grid;

% �����źŷ�������Ƶ
sound(abs(ifft(myAudio_fft_right)), FsA);

%% ���� -- ʹƵ��ͳһ����

% myAudio_fft_right = myAudio_fft;
% % myAudio_fft_right = 0;
myAudio_fft_right = zeros(size(myAudio_fft));%4*5*3�����飬������ÿ��Ԫ�ض�Ϊ0

for i = 1:size(myAudio_fft_right)
    myAudio_fft_right(i + 30000) = myAudio_fft(i);
end

plot(abs(myAudio_fft_right));

back = ifft(myAudio_fft_right);

sound(abs(back), FsA);


%% �˲� -- 1 -- add noise

% n������
Nseq = 0:1:N-1;

% ��Ƶ����
noise = ( cos(6000/ FsA * pi * Nseq)+ cos(6500/ FsA * pi * Nseq)+ cos(10000/ FsA * pi * N))* 0.5; 

% ԭʼ�ź� �ϳ� ��Ƶ����
myAudio_addNoise = myAudio + noise';

sound(myAudio_addNoise, FsA)  % ��������������������ź�

myAudio_addNoise_fft = fft(myAudio_addNoise); %���źŽ��и���Ҷ�任

subplot(211);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('ԭʼ�źŵ�Ƶ��');
grid;

subplot(212);
myAudio_addNoise_fft_abs_half = abs(myAudio_addNoise_fft(1:round(N/2)));
plot(f, myAudio_addNoise_fft_abs_half)
xlabel('Frequency/(Hz)');
title('����������źŵ�Ƶ��');
grid;

%% �˲� -- 2 -- design the ButterWorth filter

% ��������
fp=800;fs=1300;rs=35;rp=0.5;Fs=44100;

% Ԥ����
wp=2*Fs*tan(2*pi*fp/(2*Fs));
ws=2*Fs*tan(2*pi*fs/(2*Fs));

[n,wn]=buttord(wp,ws,rp,rs,'s');
% returns the lowest order
% [n,Wn] = buttord(Wp,Ws,Rp,Rs)
% To design a Butterworth filter, use the output arguments n and Wn as inputs to butter.

[b,a]=butter(n,wn,'s');
% [b,a] = butter(n,Wn) 
% ���ص�ͨģ�͹�һ��n��butterWorthϵ��

% ȥ��һ��
[num,den]=bilinear(b,a,Fs);
% bi~linear transformation method for analog-to-digital filter conversion
% [numd,dend] = bilinear(num,den,fs) converts the s-domain transfer function specified by numerator num and denominator den to a discrete equivalent.

% �õ�ϵͳ����
[h,w]=freqz(num,den,512,Fs);
% Frequency response of digital filter


subplot(3,1,1)
plot(w,abs(h));
xlabel('Ƶ��/Hz');ylabel('��ֵ');
title('������˹��ͨ�˲�����������'); axis([0,5000,0,1.2]); grid on;

subplot(3,1,2)
plot(w,20*log10(abs(h)));
xlabel('Ƶ��/Hz');ylabel('��ֵdb'); title('������˹��ͨ�˲�����������db'); axis([0,5000,-90,10]); grid on;

subplot(3,1,3)
plot(w,180/pi*unwrap(angle(h)));
xlabel('Ƶ��/Hz');ylabel('��λ');
title('������˹��ͨ�˲�����λ����'); axis([0,5000,-1000,10]) ;grid on; 

%% �˲� -- 3 -- �˲�����

x1 = myAudio;
N1=length(x1);
Y1 = fft(x1);

% �����˲�
myAudio_afterFiltered = filter(num,den,x1);

% �����˲�����Ƶ
sound(myAudio_afterFiltered,Fs);

%% �˲� -- 4 -- �˲���Ƶ��

% ��Ƶ��Ϊ��ͼ��׼��
myAudio_afterFiltered_fft =fft(myAudio_afterFiltered,N1);

% ��ͼ��

subplot(211);
plot(f,myAudio_fft_abs_half);
xlabel('Frequency/(Hz)');
title('ԭʼ�źŵ�Ƶ��');
grid;

subplot(212);
%ȡ����ֵ��fft��һ��
myAudio_afterFiltered_fft_half = abs(myAudio_afterFiltered_fft(1:round(N/2))); 
plot(f, myAudio_afterFiltered_fft_half)
xlabel('Frequency/(Hz)');
title('�˲����źŵ�Ƶ��');
grid;

%% �˲� -- 5 -- �˲���ʱ��

% ��ͼ��

subplot(211);
plot(t,myAudio,'g');%����ʱ����
xlabel('Time/(s)');
title('ԭʼ�źŵĲ���');
grid;

subplot(212);
%ȡ����ֵ��fft��һ��
plot(t,myAudio_afterFiltered,'b');%����ʱ����
xlabel('Time/(s)');
title('�˲����źŵĲ���');
grid;






