clear all;
%% Read files
cd('D:\RW\Sonar_Echo\Raw_Data\Outdoor\echo_result_02_23_2021\echo_result_02_23_2021');
npyname = 'filted_echo_Outdoor_13_c.npy';
data = readNPY(npyname);
% cd('D:\RW\Sonar_Echo\Raw_Data\Outdoor\Image\Outdoor_9_a');
% img = readNPY('image_total.npy');
% cd('D:\RW\Sonar_Echo\Raw_Data\Outdoor\prediction_result_2_23\prediction_result_2_23');
% la = readNPY('predict_result_9_a.npy');
[p,m,n] = size(data);
p = p - 1;
fs = 400e3;
time = linspace(0,1e3*n/fs,n);
dist = (time*340/2)/1e3;
%% Pick up the time-window length
echo_left = reshape(data(1:end-1,1,:),p,n);
echo_right = reshape(data(1:end-1,2,:),p,n);
noise_left = echo_left(:,9001:end);
noise_right = echo_right(:,9001:end);
n_l_std = mean(std(noise_left,0,2));
n_r_std = mean(std(noise_right,0,2));
l_thres = n_l_std*10;
r_thres = n_r_std*10;
marg = 1e-3*fs;
sp = 2500;
ep = 8000;
l_env = zeros(p,n);
r_env = zeros(p,n);
l_idx = zeros(p,2);
r_idx = zeros(p,2);
% Get envelope
for i = 1:p
    [l_env(i,:),~] = envelope(abs(echo_left(i,:)),50,'peak');
    [r_env(i,:),~] = envelope(abs(echo_right(i,:)),50,'peak');
end
% Find the index where the envelope crosses the threshold
for i = 1:p
    idx = find(l_env(i,sp:ep)>l_thres);
    if isempty(idx)
        l_idx(i,:) = [NaN,NaN];
    else
        l_idx(i,:) = [idx(1),idx(end)];
    end
    idx = find(r_env(i,sp:ep)>r_thres);
    if isempty(idx)
        r_idx(i,:) = [NaN,NaN];
    else
        r_idx(i,:) = [idx(1),idx(end)];
    end
end
% Get the window length
l_idx = l_idx + sp;
r_idx = r_idx + sp;
l_idx(:,1) = l_idx(:,1) - marg;
l_idx(:,2) = l_idx(:,2) + marg;
r_idx(:,1) = r_idx(:,1) - marg;
r_idx(:,2) = r_idx(:,2) + marg;
l_w = l_idx(:,2) - l_idx(:,1);
r_w = r_idx(:,2) - r_idx(:,1);
l_w = l_w*1e3/fs;
r_w = r_w*1e3/fs;
% Box plot
figure,
boxplot([l_w,r_w],'Labels',{'Left','Right'});
ylim([0,15]);
xlabel('Microphone');
title(npyname);
%%
% Show the echoes/envelopes those are outliers
e = eps(max([l_w,r_w]));
h = flipud(findobj(gcf,'tag','Outliers'));
l_out = h(1);
r_out = h(2);
for i = 1:10:length(l_out.YData)
    l_out_idx = find(abs(l_w - l_out.YData(i))<e)';
    for j = 1:length(l_out_idx)
        if l_out_idx(j)<=p
            figure,
            subplot(1,2,1)
            plot(time,echo_left(l_out_idx(j),:));
            hold on;
            plot(time,l_env(l_out_idx(j),:));
            plot([0,25],[l_thres, l_thres]);
            xlim([0 25])   ;         
            ylim([-0.3,0.3]);
            title(strcat('left ', string(l_out_idx(j)),'th window length:',string(l_w(l_out_idx(j))), 'ms'));
            hold off;
            subplot(1,2,2)
            plot(time,echo_left(l_out_idx(j),:));
            hold on;
            plot(time,l_env(l_out_idx(j),:));
            plot([0,25],[l_thres, l_thres]);
            scatter(l_idx(l_out_idx(j),1)*1e3/fs+1,l_env(l_out_idx(j),l_idx(l_out_idx(j),1)+1e-3*fs),'ro');
            scatter(l_idx(l_out_idx(j),2)*1e3/fs-1,l_env(l_out_idx(j),l_idx(l_out_idx(j),2)-1e-3*fs),'ro');
            xlim([l_idx(l_out_idx(j),1)-2000, l_idx(l_out_idx(j),2)+2000]*1e3/fs);
            ylim([-0.15,0.15]);
            title('zoom-in')
        end
    end
end

for i = 1:10:length(r_out.YData)
    r_out_idx = find(abs(r_w-r_out.YData(i))<e);
    for j = 1:length(r_out_idx)
        if r_out_idx(j)<=p
            figure,
            subplot(1,2,1)
            plot(time,echo_right(r_out_idx(j),:));
            hold on;
            plot(time,r_env(r_out_idx(j),:));
            plot([0,25],[r_thres, r_thres]);
            xlim([0 25])
            ylim([-0.3,0.3]);
            title(strcat('right ', string(r_out_idx(j)),'th window length:',string(r_w(r_out_idx(j))), 'ms'));
            hold off;
            subplot(1,2,2)
            plot(time,echo_right(r_out_idx(j),:));
            hold on;
            plot(time,r_env(r_out_idx(j),:));
            plot([0,25],[r_thres, r_thres]);
            scatter(r_idx(r_out_idx(j),1)*1e3/fs+1,r_env(r_out_idx(j),r_idx(r_out_idx(j),1)+1e-3*fs),'ro');
            scatter(r_idx(r_out_idx(j),2)*1e3/fs-1,r_env(r_out_idx(j),r_idx(r_out_idx(j),2)-1e-3*fs),'ro');
            xlim([r_idx(r_out_idx(j),1)-2000, r_idx(r_out_idx(j),2)+2000]*1e3/fs);
            ylim([-0.15,0.15]);
            title('zoom-in')
        end
    end
end

%% Plot time/distance vs echo, as well as spectrogram
% cd('D:\RW\Sonar_Echo\Raw_Data\Outdoor\Image\Outdoor_9_a');
% for i = 1:p
%    echo1 = reshape(data(i,1,:),1,n);
%    echo2 = reshape(data(i,2,:),1,n);
%    imname = sprintf('%d_camera.png',i);
%    figname = sprintf('%d_demo.jpg',i);
%    im = rgb2gray(imread(imname));
%    resized = im(61:227,55:389);
%    % Plot in time/dist 
% %    figure,
% %    subplot(2,1,1)
% %    plot(time,echo);
% %    ylim([-0.5 0.5]);
% %    xlabel('Time [ms]');
% %    ylabel('Amplitude')
% %    subplot(2,1,2)
% %    plot(dist,echo);
% %    xlabel('Distance [m]');
% %    ylabel('Amplitude')   ;
% %    ylim([-0.5 0.5])
%    % Spectrogram
%    fig = figure( 'Position',  [00,00,1100,1100],'Visible','off');
%    ax(1) = subplot(2,4,[1,4]);
%    imagesc(resized);
%    hold on;
%    
%    ax(2) = subplot(2,4,[5,6]);
%    spectrogram(echo1,128,100,128,fs,'yaxis');
%    title('Left Mic')
%    cmap = colormap;
%    colorbar off
%    ax(3) = subplot(2,4,[7,8]);
%    spectrogram(echo2,128,100,128,fs,'yaxis');
%    title('Right Mic')
%    ylabel('')
%    colormap(ax(1),'gray');
%    colormap(ax(2),cmap);
%    colormap(ax(3),cmap);
%    saveas(fig,figname);
% end
%% plot the 64x64 f/g image and left+right mic spectrogram
% k = 1;
% sum1 = zeros(1,10000);
% sum2 = zeros(1,10000);
% noise1 = zeros(p,1000);
% noise2 = zeros(p,1000);
% for i = 1:1:p-1
%    echo1 = reshape(data(i,1,:),1,n);
%    echo2 = reshape(data(i,2,:),1,n);
%    echo1(1:10) = 0;
%    echo2(1:10) = 0; 
%    sum1 = sum1 + echo1;
%    sum2 = sum2 + echo2;
%    noise1(i,:) = echo1(9001:end);
%    noise2(i,:) = echo2(9001:end);
%    image = reshape(img(i,:,:,:),160,160,3);
%    if la(i) == 0
%        label = ' gap';
%    elseif la(i) == 1
%        label = ' foliage';
%    elseif la(i) == 2
%        label = ' uncertain';
%    end
%    
%    fig = figure('Position',[00,00,1500,800],'visible','off');
%    subplot(2,2,[1,3])
%    imshow(image);
%    title(strcat('prediction:',label));
%    subplot(2,2,2)
%    spectrogram(echo1,128,100,128,fs,'yaxis');
%    caxis([-160, -50 ])
%    title('Left Mic');
%    subplot(2,2,4)
%    spectrogram(echo2,128,100,128,fs,'yaxis');
%    caxis([-160, -50 ])
%    title('Right Mic');
%    figname = sprintf('image_echo_%d',k);
%    saveas(fig,figname,'jpg');
%    k = k + 1;
% end
%% 
