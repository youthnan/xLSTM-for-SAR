clc;
clear;
% close all;
addpath(genpath('D:\zq试验数据\代码\Autofocus_GPU'))
addpath(genpath('D:\zq试验数据\代码\BP_GPU'))
addpath 'D:\zq实验数据\代码\BP_GPU'
addpath 'D:\zq实验数据\代码\Autofocus_GPU'
addpath(genpath('E:\北川无人机数据处理\BP_GPU'))
addpath 'E:\北川无人机数据处理\BP_GPU'
addpath('E:\北川无人机数据处理\北川无人机三位数据处理\3dtool_function');
addpath('E:\716多航过飞行数据处理\BP_GPU_AziWin_v2');

%% 读取回波
% [fileName, filePath] = uigetfile('.dat','Select dat', 'MultiSelect', 'on');
% if ischar(fileName)
%     fileName = {fileName};
% end
load file.mat
%%
%采样深度和PRF需要根据设置更改
SampleDepth = 32768;
PRF = 2000;

%% 读取原始回波文件dat
protocol = GetProtocol(SampleDepth);
records = cell(numel(fileName), 1);
azi_offset=309; % 单位：s 回波起始位置 好的结果：
azi_dur = 2; % 单位：s    回波长度  好的结果：

framelength =0;

for i = 1:size(protocol,1)
    framelength = framelength + double(protocol{i, 3});
end

for i = 1:numel(fileName)
    fileID = fopen([filePath, fileName{i}], 'rb');
    if fileID == -1
        error('文件无法打开');
    end

    startByte = framelength * PRF * azi_offset;
    bytesLength = framelength * PRF * azi_dur;
    endByte = startByte + bytesLength;
    frame_header = [0x7f 0xff 0x1c 0xbc];
    chunk_size = 1024; %framelength; % 1MB

    status = fseek(fileID, startByte, 'bof');
    if status ~= 0
        fclose(flieID);
        error('定位到起始位置失败');
    end

    current_position = startByte;
    buffer = uint8([]);
    frame_positions = [];
    while current_position < endByte
        bytes_to_read = min(chunk_size, endByte - current_position);
        data = fread(fileID, bytes_to_read, 'uint8');

        if isempty(data)
            break;
        end

        data = [buffer; data];
        idx = strfind(data', frame_header);

        if ~isempty(idx)
            absolute_positions = current_position - length(buffer) + idx - 1;
            frame_positions = [frame_positions, absolute_positions];
        end

        if ~isempty(frame_positions)
            break
        end

        buffer = data(end - 3 + 1:end);
        current_position = current_position + bytes_to_read;
    end
    fseek(fileID, frame_positions(1), 'bof');
    dataTemp = fread(fileID, bytesLength, 'uint8');
    recordsTemp = Echo2Protocol(dataTemp, protocol);
    records{i} = recordsTemp;
end
clear recordsTemp
clear dataTemp

% 方位降采样
downsampling_ratio = 1;
PRF = PRF / downsampling_ratio;
for i = 1:numel(fileName)
    records{i} = records{i}(1:downsampling_ratio:end,:);
end

% 四通道回波拼接
y_I = [];
y_Q = [];
IQArray = {};
frCount = length(records{1});

data = zeros(32768,frCount);

for j = 1: frCount
    y_I=[];
    y_Q=[];
    for i = 1 : numel(fileName)
        IQArray{i} = reshape(records{i}(j).IQ,2,[]);%第i个文件的第j条脉冲
        y_I= [y_I;IQArray{i}(1,:)];
        y_Q= [y_Q;IQArray{i}(2,:)];
    end
    y_I=double(reshape(y_I,1,[]));
    y_Q=double(reshape(y_Q,1,[]));
    data(:,j)=1i*y_I+y_Q;
end

clear y_I
clear y_Q

% figure;
% imagesc((abs((data(:,:)))));
% title("二维回波的时域幅度");xlabel("方位向");ylabel("距离向");

%% hilbert变换
% new_data = hilbert(imag(data));
% figure;
% imagesc((dbMax((new_data(:,:)))));
% title("二维回波的时域幅度");xlabel("方位向");ylabel("距离向");

%% 脉压
TiaopinFlag = 1;
N_Signal = SampleDepth;
B = 800e6;%带宽
Tp = 1e-6;%脉宽
Fs=1200e6;%采样率
load refr
RanRefSignal = circshift(refr,-round(32768/2));
refr_fft = ftx(RanRefSignal);
clear refr
clear RanRefSignal

pulseCompress = iftx(ftx(data).*conj(refr_fft));
clear data

% figure;imagesc(dbMax((pulseCompress)))%二维图
% title('二维图');zoom on;

% figure;imagesc(dbMax((fty(pulseCompress))))%二维图
% title('二维图');zoom on;

% pulseCompress_correct = ifty(circshift(fty(pulseCompress),2000,2));  % 校正多普勒质心
% figure;imagesc(dbMax(fty(pulseCompress_correct)))%二维图
% title('二维图');zoom on;


% cd E:\716多航过飞行数据处理\脉压数据\azi_offset_12_9.
% load pulseCompress.mat
% cd E:\716多航过飞行数据处理
%% 读取航迹
[RanNum,AziNum] = size(pulseCompress);
%%%%%%%%%%%%%%%%%%%%%%% 读取航迹
AziNum = length(records{1});
Lat = zeros(1,AziNum);
Lon = zeros(1,AziNum);
Alt = zeros(1,AziNum);
utc = zeros(1,AziNum);
Heading = zeros(1,AziNum);
Pitch = zeros(1,AziNum);
Roll = zeros(1,AziNum);
ms = zeros(1,AziNum);
for i = 1:AziNum
    Lat(i) = records{1}((i)).lat;
    Lon(i) = records{1}((i)).Lon;
    Alt(i) = records{1}((i)).Alt;
    utc(i) = records{1}((i)).UTC;
    Heading(i) = records{1}((i)).Heading;
    Pitch(i) = records{1}((i)).Pitch;
    Roll(i) = records{1}((i)).Roll;
    ms(i)  = records{1}((i)).ms;
end
figure;plot(utc)
title('载荷存储的utc时间')
figure;plot(Alt)
title('载荷存储的高度')
figure;plot(Lon)
title('载荷存储的精度')
figure;plot(Lat)
title('载荷存储的纬度')
%%
%%%%%%%%%%%%%%%%%%%%%%% 修正惯导数据
navi_prf = 100;  % 惯导刷新率
[utc_unique,idx_unique] = unique(utc);
Lat_unique = Lat(idx_unique);
Lon_unique = Lon(idx_unique);
Alt_unique = Alt(idx_unique);
diff_utc_unique = mod(diff(utc_unique),40);  % utc时间
utc_fix = [];
Lat_fix = [];
Lon_fix = [];
Alt_fix = [];
N_ratio_ref = PRF/navi_prf;
for i = 1:length(diff_utc_unique)
    ratio = round(diff_utc_unique(i) * PRF);
    if ratio == N_ratio_ref
        utc_fix = [utc_fix;repmat(utc_unique(i),ratio,1)];
        Lat_fix = [Lat_fix;repmat(Lat_unique(i),ratio,1)];
        Lon_fix = [Lon_fix;repmat(Lon_unique(i),ratio,1)];
        Alt_fix = [Alt_fix;repmat(Alt_unique(i),ratio,1)];
    else
        N = round(ratio/N_ratio_ref);
        interp_utc_unique = (utc_unique(i+1)-utc_unique(i))/N*(0:N-1)+utc_unique(i);
        interp_utc_temp = repmat(interp_utc_unique,N_ratio_ref,1);
        utc_fix = [utc_fix;interp_utc_temp(:)];
        interp_Lat_unique = (Lat_unique(i+1)-Lat_unique(i))/N*(0:N-1)+Lat_unique(i);
        interp_Lat_temp = repmat(interp_Lat_unique,N_ratio_ref,1);
        Lat_fix = [Lat_fix;interp_Lat_temp(:)];
        interp_Lon_unique = (Lon_unique(i+1)-Lon_unique(i))/N*(0:N-1)+Lon_unique(i);
        interp_Lon_temp = repmat(interp_Lon_unique,N_ratio_ref,1);
        Lon_fix = [Lon_fix;interp_Lon_temp(:)];
        interp_Alt_unique = (Alt_unique(i+1)-Alt_unique(i))/N*(0:N-1)+Alt_unique(i);
        interp_Alt_temp = repmat(interp_Alt_unique,N_ratio_ref,1);
        Alt_fix = [Alt_fix;interp_Alt_temp(:)];
    end
end

AziNum = (length(utc_fix)>AziNum) * AziNum + (length(utc_fix)<=AziNum) * length(utc_fix);
utc_fix = utc_fix(1:AziNum);
Lat_fix = Lat_fix(1:AziNum);
Lon_fix = Lon_fix(1:AziNum);
Alt_fix = Alt_fix(1:AziNum);
%%%%%%%%%%%%%%%%%%%%%%% 截取回波数据
pulseCompress_fix =pulseCompress(:,1:AziNum);
Lat0 = Lat_fix(1);
Lon0 = Lon_fix(1);
Alt0 = 3;    % 当地海拔
wgs84 = wgs84Ellipsoid;
%kmlwriteline('ReceiverTrajectory2.kml',Lat_fix,Lon_fix,Alt_fix);
[xEast,yNorth,zUp] = geodetic2enu(Lat_fix,Lon_fix,Alt_fix,Lat0,Lon0,Alt0,wgs84);
R_path = [xEast';yNorth';zUp'];
% figure;plot3(R_path(1,:),R_path(2,:),R_path(3,:))
% title('惯导数据')
[RPos_rotate,~] = RotateTrajectory(R_path');  % 旋转航迹沿着y轴飞行

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 基于滑动平均平滑毛刺
RPos = RPos_rotate;
[~,idx_temp] = unique(utc_fix);
unique_temp = RPos(idx_temp,1);
temp_path = unique_temp;
figure;plot(temp_path)
windowsize = 30;
for i = windowsize/2+1:length(unique_temp)-windowsize/2
    mu = mean(unique_temp(i-windowsize/2:i+windowsize/2));
    sigma = std(unique_temp(i-windowsize/2:i+windowsize/2));
    temp_path(i) = unique_temp(i).*(abs(unique_temp(i)-mu)<1*sigma)+mu.*(abs(unique_temp(i)-mu)>=1*sigma);
end
unique_temp = flipud(unique_temp);
temp_path = flipud(temp_path);
for i = windowsize/2+1:length(unique_temp)-windowsize/2
    mu = mean(unique_temp(i-windowsize/2:i+windowsize/2));
    sigma = std(unique_temp(i-windowsize/2:i+windowsize/2));
    temp_path(i) = unique_temp(i).*(abs(unique_temp(i)-mu)<1*sigma)+mu.*(abs(unique_temp(i)-mu)>=1*sigma);
end
temp_path = flipud(temp_path);
%hold on;plot(temp_path)

%figure;plot(RPos(:,1))
temp = repmat(temp_path',N_ratio_ref,1);
RPos(:,1) = temp(:);
%hold on;plot(RPos(:,1))

unique_temp = RPos(idx_temp,3);
temp_path = unique_temp;
%figure;plot(temp_path)
for i = windowsize/2+1:length(unique_temp)-windowsize/2
    mu = mean(unique_temp(i-windowsize/2:i+windowsize/2));
    sigma = std(unique_temp(i-windowsize/2:i+windowsize/2));
    temp_path(i) = unique_temp(i).*(abs(unique_temp(i)-mu)<1*sigma)+mu.*(abs(unique_temp(i)-mu)>=1*sigma);
end
unique_temp = flipud(unique_temp);
temp_path = flipud(temp_path);
for i = windowsize/2+1:length(unique_temp)-windowsize/2
    mu = mean(unique_temp(i-windowsize/2:i+windowsize/2));
    sigma = std(unique_temp(i-windowsize/2:i+windowsize/2));
    temp_path(i) = unique_temp(i).*(abs(unique_temp(i)-mu)<1*sigma)+mu.*(abs(unique_temp(i)-mu)>=1*sigma);
end
temp_path = flipud(temp_path);
%hold on;plot(temp_path)

%figure;plot(RPos(:,3))
temp = repmat(temp_path',N_ratio_ref,1);
RPos(:,3) = temp(:);
%hold on;plot(RPos(:,3))

%%%%%%%%%%%%%%%%%%%%% 根据窗长调整航迹边界
RPos = RPos(windowsize/2*N_ratio_ref+1:end-windowsize/2*N_ratio_ref,:);
pulseCompress_fix = pulseCompress_fix(:,windowsize/2*N_ratio_ref+1:end-windowsize/2*N_ratio_ref);
AziNum = length(RPos(:,1));

%%%%%%%%%%%%%%%%%%%%% 高斯平滑_拟合航迹
%figure;plot(RPos(:,1))
RPos(:,1) = smoothdata(RPos(:,1),'gaussian',1024);
%hold on;plot(RPos(:,1))
RPos(:,2) = polyval(polyfit(1:AziNum,RPos(:,2),1),1:AziNum);
%figure;plot(RPos(:,3))
RPos(:,3) = smoothdata(RPos(:,3),'gaussian',1024);
%hold on;plot(RPos(:,3))

%figure;plot(RPos(:,1))
RPos(:,1) = polyval(polyfit(1:AziNum,RPos(:,1),30),1:AziNum);
%hold on;plot(RPos(:,1))
% figure;plot(RPos(:,3))
% RPos(:,3) = polyval(polyfit(1:AziNum,RPos(:,3),30),1:AziNum);
% hold on;plot(RPos(:,3))

% figure;plot(RPos(:,1))
RPos(:,1) = smoothdata(RPos(:,1),'gaussian',256);
%hold on;plot(RPos(:,1))
% figure;plot(RPos(:,3))
RPos(:,3) = smoothdata(RPos(:,3),'gaussian',256);
%hold on;plot(RPos(:,3))

TPos = RPos;


% figure;plot3(RPos(:,1),RPos(:,2),RPos(:,3))
% legend('旋转航迹后')
% 
% figure;plot(RPos_rotate(windowsize/2*N_ratio_ref+1:end-windowsize/2*N_ratio_ref,1))
% hold on;plot(RPos(:,1))
% legend('平滑航迹前','平滑航迹后')
% figure;plot(RPos_rotate(windowsize/2*N_ratio_ref+1:end-windowsize/2*N_ratio_ref,3))
% hold on;plot(RPos(:,3))
% legend('平滑航迹前','平滑航迹后')




% RPos(:,1) = 0;
% RPos(:,2) = linspace(46.8,-46.8,AziNum);
% RPos(:,3) = 105.8;
% TPos = RPos;


%% 相关参数
fc = 9.5e9;
c = 3e8;
lambda = c / fc;

%% BP
TRPos = [TPos.';RPos.'];
SapRate = Fs;
Tr = (0:RanNum-1)/SapRate;
fc = 9.5e9;
PointCenter =[916.6 -11.5 0];
BpXNum = 512;
BpYNum = 512;
deltaX_BP = 0.3;
deltaY_BP = 0.3;
X = PointCenter(1) + (-BpXNum/2:BpXNum/2-1)*deltaX_BP;
Y = PointCenter(2) + (-BpYNum/2:BpYNum/2-1)*deltaY_BP;
BP_coaf = 2;
cd E:\北川无人机数据处理\BP_GPU
%imageRe2 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);
imageRe3 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);
imageRe3 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);
cd E:\716多航过飞行数据处理
figure;imagesc(Y,X,dbMax(imageRe3.'));
clim([-50 0])
axis equal
axis tight
title('直接BP')




% %% 保存BP结果图
% idx = strfind(filePath,'\');
% echoname = filePath(idx(end-1)+1:idx(end)-1);
% 
% name = sprintf('EchoName_%s_AziOffset_%d_AziDur_%d_直接BPImage_%d_%d_%d_%d',echoname,azi_offset,azi_dur,BpXNum,BpYNum,deltaX_BP,deltaY_BP);
% save_pngraw_16bit(name,imageRe3,1,[0 1]);
% 

% % 
%% 细节图
PointCenter =[917.65 -20 0];
BpXNum = 256;
BpYNum = 256;
deltaX_BP = 0.05;
deltaY_BP = 0.05;
X = PointCenter(1) + (-BpXNum/2:BpXNum/2-1)*deltaX_BP;
Y = PointCenter(2) + (-BpYNum/2:BpYNum/2-1)*deltaY_BP;
BP_coaf = 2;
cd E:\北川无人机数据处理\BP_GPU
%imageRe2 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);
imageRe9 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);
cd E:\716多航过飞行数据处理
figure;imagesc(Y,X,dbMax(imageRe9.'))
clim([-50 0])
axis equal
axis tight


%save('pulseCompress.mat', 'pulseCompress')
%clear pulseCompress
% hold on;plot(RPo(:,2),RPos(:,1),'r','LineWidth',2)

%% 过程数据记录
TRPos = [TPos.';RPos.'];
folderName=['azi_offset_',num2str(azi_offset)];
folderPath=fullfile('E:/716多航过飞行数据处理/回波数据',folderName);
if ~exist(folderPath,'dir')
    mkdir(folderPath)
end
%save(fullfile(folderPath,'pulseCompress_fix.mat'),'pulseCompress_fix');
save(fullfile(folderPath,'data.mat'),'data');
save(fullfile(folderPath,'TRPos.mat'),'TRPos');
save(fullfile(folderPath,'Tr.mat'),'Tr');
save(fullfile(folderPath,'BpXNum.mat'),'BpXNum');
save(fullfile(folderPath,'BpYNum.mat'),'BpYNum');
save(fullfile(folderPath,'PointCenter.mat'),'PointCenter');
save(fullfile(folderPath,'deltaX_BP.mat'),'deltaX_BP');
save(fullfile(folderPath,'deltaY_BP.mat'),'deltaY_BP');
save(fullfile(folderPath,'azi_dur.mat'),'azi_dur');
save(fullfile(folderPath,'p1'),'p1');
save(fullfile(folderPath,'coe_num'),'coe_num');

%% 聚焦
% [value,idx]=max(abs(imageRe3));
% [~,iidx]=max(value);
% vpa(X(iidx))
% Y(idx(iidx))
AziNum=size(pulseCompress_fix, 2);
c = 3e8;
Rref = Tr*c/2;

%徙动
% figure;imagesc(1:AziNum,Rref,abs((pulseCompress)));clim([0 80000]);%二维图
% title('二维图');zoom on;

p1=[916.75 -20 0];
Pos = TRPos(1:3,:);
R1 = vecnorm(p1'-Pos);
hold on;plot(1:AziNum,R1,'r')
%%

%%%%%%%%%%%%%%%%% 局部
temp = R1;
R_temp1_max = max(temp);
R_temp1_min = min(temp);
[~,idx_1_max] = min(abs(Rref - R_temp1_max));
[~,idx_1_min] = min(abs(Rref - R_temp1_min));
idx_1_max = idx_1_max + 100;
idx_1_min = idx_1_min - 100;
Srancom_RVP_temp1 = pulseCompress_fix(idx_1_min:idx_1_max,:);
figure;imagesc(1:AziNum,Rref(idx_1_min:idx_1_max),dbMax(Srancom_RVP_temp1));caxis([-30 0])
hold on;plot(1:AziNum,temp,'r')

Y1 = R1;
count = 1;
while count <= AziNum
    [~,idx] = min(abs(Rref - Y1(count)));
    [~,idx_temp] = max(abs([pulseCompress_fix(idx-2,count) pulseCompress_fix(idx-1,count) pulseCompress_fix(idx,count)...
        pulseCompress_fix(idx+1,count) pulseCompress_fix(idx+2,count)]));
    
    RCM1(count) = idx-2+idx_temp;
    count = count + 1;
end

Y1 = Rref(RCM1);

RCM1 = zeros(1,AziNum);
count = 1;
while count <= AziNum
    [~,idx] = min(abs(Rref - Y1(count)));
    [~,idx_temp] = max(abs([pulseCompress_fix(idx-1,count) pulseCompress_fix(idx,count) pulseCompress_fix(idx+1,count)]));
    RCM1(count) = idx-2+idx_temp;
    count = count + 1;
end

RCM1_r = Rref(RCM1);
hold on;plot(1:AziNum,RCM1_r,'w')
idx_x = find(RCM1_r~=Rref(1));
idx_y = RCM1_r(idx_x);

coe_num=8;
coe = polyfit(idx_x,idx_y,coe_num);
RCM1_nihe = polyval(coe,1:AziNum);  
hold on;plot(1:AziNum,RCM1_nihe,'k')

newRT = TRPos(1:3,:)';
Ri = vecnorm(newRT'-p1');
Rr = RCM1_nihe;
for i = 1:AziNum
    deltaR = Rr(i) - Ri(i);
    H= [(TRPos(1,i)-p1(1))/Ri(i)];
    delta_xyz = inv(H) * deltaR;
    newRT(i,1) = newRT(i,1) + delta_xyz;
end
newR1 = vecnorm(newRT'-p1');
figure;plot(newR1-Rr)




[~,idx] = min(abs(Rref'-newR1));
RCMdata = pulseCompress_fix(sub2ind(size(pulseCompress_fix),idx,1:AziNum));
phase_p1 = phase(RCMdata);
phase_p1 = phase_p1 - polyval(polyfit(1:AziNum,phase_p1,1),1:AziNum);
phase_ideal = phase(exp(-1j*4*pi*fc*newR1/c));
phase_ideal = phase_ideal - polyval(polyfit(1:AziNum,phase_ideal,1),1:AziNum);
phase_err = unwrap(mod(phase_p1-phase_ideal,2*pi));
phase_err = phase_err - polyval(polyfit(1:AziNum,phase_err,1),1:AziNum);
phase_err = smoothdata(phase_err,'gaussian',128);
figure;plot(phase_err)
%%
newRT2 = newRT;
Ri = vecnorm(newRT2'-p1');
for i = 1:AziNum
    deltaR = -phase_err(i)*c/fc/4/pi;
    H= [(newRT2(i,1)-p1(1))/Ri(i)];
    delta_xyz = inv(H) * deltaR;
    newRT2(i,1) = newRT2(i,1) + delta_xyz;
end
newR2 = vecnorm(newRT2'-p1');

TRPos = [newRT2';newRT2'];
SapRate = Fs;
Tr = (0:RanNum-1)/SapRate;
fc = 9.5e9;
PointCenter = p1;

BpXNum = 256;
BpYNum = 256;
deltaX_BP = 0.3;
deltaY_BP = 0.3;
X = PointCenter(1) + (-BpXNum/2:BpXNum/2-1)*deltaX_BP;
Y = PointCenter(2) + (-BpYNum/2:BpYNum/2-1)*deltaY_BP;
BP_coaf = 2;
imageRe1 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);

figure;imagesc(Y,X,db(abs(imageRe1.')));axis equal;axis tight;title('未加窗BP')


%% 加窗
diff_RPos = diff(RPos); % 相邻点坐标差 (AziNum-1) x 3
step_sizes = sqrt(sum(diff_RPos.^2, 2)); % 每个步长的长度
Lsar = sum(step_sizes)*2;
nwin = 100; 
win = hamming(nwin); 

% CPU BP
echo=pulseCompress_fix';
TRPos_3=TRPos;

CPU_BP_imageRe= BackProjection(echo,TRPos_3,Tr',lambda,fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],8);
figure;imagesc(Y,X,db(abs(CPU_BP_imageRe.')));axis equal;axis tight;title('未加窗BP')

CPU_BPWin_imageRe= BP_AziWin_cpu(echo,TRPos_3,Tr',lambda,fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],8,30,0,ones(30,1));
figure;imagesc(Y,X,db(abs(CPU_BPWin_imageRe.')));axis equal;axis tight;title('加窗BP')


[value,idx]=max(abs(imageRe1));
[~,iidx]=max(value);
vpa(X(iidx))
Y(idx(iidx))

%% 加窗
diff_RPos = diff(RPos); % 相邻点坐标差 (AziNum-1) x 3
step_sizes = sqrt(sum(diff_RPos.^2, 2)); % 每个步长的长度
Lsar = sum(step_sizes)*2;
nwin = 200; 
win = hamming(nwin); 



SLC_stack= BP_AziWinMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,200,-100,win,0);
% SLC_stack= BPWMexPro(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0,200,0);
% SLC_stack= BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);

figure,imagesc(Y,X,db(abs(SLC_stack.')));colormap(jet); axis equal;axis tight
title('加窗BP')



%% 保存BP结果图
idx = strfind(filePath,'\');
echoname = filePath(idx(end-1)+1:idx(end)-1);
name = sprintf('EchoName_%s_AziOffset_%d_AziDur_%d_一个点估轨迹BPImage_%d_%d_%d_%d',echoname,azi_offset,azi_dur,BpXNum,BpYNum,deltaX_BP,deltaY_BP);
save_pngraw_16bit(name,imageRe1,1,[0 1]);

%%
p1 = [916.8 -39.2 0];
R1 = vecnorm(p1'-newRT2');
hold on;plot(1:AziNum,R1,'r')

RCM1_nihe = R1;

%%%%%%%%%%%%%%%%% 局部
p2 = [913.5 39.6 0];
R2 = vecnorm(p2'-newRT2');
temp = R2;
R_temp1_max = max(temp);
R_temp1_min = min(temp);
[~,idx_1_max] = min(abs(Rref - R_temp1_max));
[~,idx_1_min] = min(abs(Rref - R_temp1_min));
idx_1_max = idx_1_max+100;
idx_1_min = idx_1_min-100;
Srancom_RVP_temp1 = pulseCompress(idx_1_min:idx_1_max,:);
figure;imagesc(1:AziNum,Rref(idx_1_min:idx_1_max),dbMax(Srancom_RVP_temp1));caxis([-30 0])
hold on;plot(1:AziNum,temp,'r')

Y2 = R2;
RCM2 = zeros(1,AziNum);
count = 1;
while count <= AziNum
    [~,idx] = min(abs(Rref - Y2(count)));
    [~,idx_temp] = max(abs([pulseCompress(idx-2,count) pulseCompress(idx-1,count) pulseCompress(idx,count)...
        pulseCompress(idx+1,count) pulseCompress(idx+2,count)]));
    RCM2(count) = idx-3+idx_temp;
    count = count + 1;
end

RCM2_r = Rref(RCM2);
hold on;plot(1:AziNum,RCM2_r,'w')
idx_x = find(RCM2_r~=Rref(1));
idx_y = RCM2_r(idx_x);
coe = polyfit(idx_x,idx_y,12);
RCM2_nihe = polyval(coe,1:AziNum);  
hold on;plot(1:AziNum,RCM2_nihe,'r')

%%%%%%%%%%%%%%%%% 局部
p3 = [619.9 -102.9 0];
R3 = vecnorm(p3'-newRT2');
temp = R3;
R_temp1_max = max(temp);
R_temp1_min = min(temp);
[~,idx_1_max] = min(abs(Rref - R_temp1_max));
[~,idx_1_min] = min(abs(Rref - R_temp1_min));
idx_1_max = idx_1_max+100;
idx_1_min = idx_1_min-100;
Srancom_RVP_temp1 = pulseCompress(idx_1_min:idx_1_max,:);
figure;imagesc(1:AziNum,Rref(idx_1_min:idx_1_max),dbMax(Srancom_RVP_temp1));caxis([-30 0])
hold on;plot(1:AziNum,temp,'r')

Y3 = R3;
RCM3 = zeros(1,AziNum);
count = 1;
while count <= AziNum
    [~,idx] = min(abs(Rref - Y3(count)));
    [~,idx_temp] = max(abs([pulseCompress(idx-2,count) pulseCompress(idx-1,count) pulseCompress(idx,count)...
        pulseCompress(idx+1,count) pulseCompress(idx+2,count) pulseCompress(idx+3,count)]));
    RCM3(count) = idx-3+idx_temp;
    count = count + 1;
end

Y3 = Rref(RCM3);

RCM3 = zeros(1,AziNum);
count = 1;
while count <= AziNum
    [~,idx] = min(abs(Rref - Y3(count)));
    [~,idx_temp] = max(abs([pulseCompress(idx-1,count) pulseCompress(idx,count) pulseCompress(idx+1,count) ]));
    RCM3(count) = idx-2+idx_temp;
    count = count + 1;
end

RCM3_r = Rref(RCM3);
hold on;plot(1:AziNum,RCM3_r,'w')
idx_x = find(RCM3_r~=Rref(1));
idx_y = RCM3_r(idx_x);
coe = polyfit(idx_x,idx_y,12);
RCM3_nihe = polyval(coe,1:AziNum);  
hold on;plot(1:AziNum,RCM3_nihe,'r')

%%%%%%%%%%%%%%%%% 包络级校正
Pos_Azi = newRT2;
global Pos_Azi
global RCM1_nihe
global RCM2_nihe
global RCM3_nihe

PointDeviationEstimation;

figure;plot(temp1(:)-RCM1_nihe(:))
hold on;plot(temp2(:)-RCM2_nihe(:))
hold on;plot(temp3(:)-RCM3_nihe(:))

figure;imagesc(1:AziNum,Rref,dbMax((pulseCompress)))%二维图
title('二维图');zoom on;
hold on;plot(1:AziNum,temp1,'r')
hold on;plot(1:AziNum,temp2,'r')
hold on;plot(1:AziNum,temp3,'r')

%%%%%%%%%%%%%%%%% 相位级校正
pulse_num = AziNum;
Lambda = c/fc;

[~,idx1]=min(abs(temp1'-Rref'));
ind = [(1:pulse_num)' idx1(1:pulse_num)'];
p1_phase = phase(pulseCompress(sub2ind(size(pulseCompress),ind(:,2),ind(:,1))));
phase_temp1 = unwrap(mod(p1_phase(:),2*pi))-unwrap(mod(-4*pi*temp1(1:pulse_num)/Lambda,2*pi));

[~,idx2]=min(abs(temp2'-Rref'));
ind = [(1:pulse_num)' idx2(1:pulse_num)'];
p2_phase = phase(pulseCompress(sub2ind(size(pulseCompress),ind(:,2),ind(:,1))));
phase_temp2 = unwrap(mod(p2_phase(:),2*pi))-unwrap(mod(-4*pi*temp2(1:pulse_num)/Lambda,2*pi));

[~,idx3]=min(abs(temp3'-Rref'));
ind = [(1:pulse_num)' idx3(1:pulse_num)'];
p3_phase = phase(pulseCompress(sub2ind(size(pulseCompress),ind(:,2),ind(:,1))));
phase_temp3 = unwrap(mod(p3_phase(:),2*pi))-unwrap(mod(-4*pi*temp3(1:pulse_num)/Lambda,2*pi));

phase_temp1 = phase_temp1' - polyval(polyfit(1:pulse_num,phase_temp1,1),1:pulse_num);
phase_temp2 = phase_temp2' - polyval(polyfit(1:pulse_num,phase_temp2,1),1:pulse_num);
phase_temp3 = phase_temp3' - polyval(polyfit(1:pulse_num,phase_temp3,1),1:pulse_num);

figure;plot(phase_temp1)
hold on;plot(phase_temp2)
hold on;plot(phase_temp3)

new_phase_temp1 = smoothdata(phase_temp1,'gaussian',256);
new_phase_temp2 = smoothdata(phase_temp2,'gaussian',64);
new_phase_temp3 = smoothdata(phase_temp3,'gaussian',64);
figure;plot(new_phase_temp1)
hold on;plot(new_phase_temp2)
hold on;plot(new_phase_temp3)

new_Pos2 = new_Pos;
R_temp1 = vecnorm((new_Pos - point_temp1)')';
R_temp2 = vecnorm((new_Pos - point_temp2)')';
R_temp3 = vecnorm((new_Pos - point_temp3)')';
for i = 1:AziNum
    delta_R = -[new_phase_temp1(i) new_phase_temp2(i) new_phase_temp3(i) ]'*c/fc/4/pi;
    H = [(new_Pos(i,1)-point_temp1(1))/R_temp1(i) (new_Pos(i,1)-point_temp2(1))/R_temp2(i) (new_Pos(i,1)-point_temp3(1))/R_temp3(i);...
        (new_Pos(i,2)-point_temp1(2))/R_temp1(i) (new_Pos(i,2)-point_temp2(2))/R_temp2(i) (new_Pos(i,2)-point_temp3(2))/R_temp3(i);...
        (new_Pos(i,3)-point_temp1(3))/R_temp1(i) (new_Pos(i,3)-point_temp2(3))/R_temp2(i) (new_Pos(i,3)-point_temp3(3))/R_temp3(i) ]';
    delta_xyz = inv(H.'*H)*H.'*delta_R;
    new_Pos2(i,:) = new_Pos2(i,:) + delta_xyz';
end

TRPos = [new_Pos2';new_Pos2'];
SapRate = Fs;
Tr = (0:RanNum-1)/SapRate;
fc = 9.5e9;
PointCenter = [1500 0 0];
BpXNum = 12000;
BpYNum = 12000;
deltaX_BP = .1;
deltaY_BP = .1;
X = PointCenter(1) + (-BpXNum/2:BpXNum/2-1)*deltaX_BP;
Y = PointCenter(2) + (-BpYNum/2:BpYNum/2-1)*deltaY_BP;
BP_coaf = 2;
imageRe2 = BPMex(pulseCompress_fix,TRPos,Tr',fc,[BpXNum,BpYNum],PointCenter,[deltaX_BP,deltaY_BP],BP_coaf,0);

figure;imagesc(Y,X,dbMax(imageRe2.'))
clim([-50 0])
axis equal
axis tight

name = sprintf('BPImage_%d_%d_%d_%d',BpXNum,BpYNum,deltaX_BP,deltaY_BP);
save_pngraw_16bit(name,imageRe1,1,[0 1]);