%% This code is to 1) find the time-window for spectrogram
%                  2) generate the spectrogram images as input to CNN
%  Based on the boxplot of the overall time-window length, we chose the
%  maximum window-length as the final length for all 10 datasets. The
%  maximum window length is 16 ms when threshold is 10xnoise_std.
clear all;
close all
%% Dataset path
cd('F:\College\MachineLearning\ShortDistanceProfile\echo_result_02_23_2021');
dataset_name = {'2_a', '4_b'};
%'2_a','4_b','6_a','7_b','9_a','9_d_1','9_d_2','10_a','12_a','13_c'
%%
if ~exist('spectrogram_image_left', 'dir')
   mkdir('spectrogram_image_left')
end 
if ~exist('spectrogram_image_right', 'dir')
   mkdir('spectrogram_image_right')
end 
%%
for i = 1:length(dataset_name)
cd('F:\College\MachineLearning\ShortDistanceProfile\echo_result_02_23_2021');
echoname = ['filted_echo_Outdoor_',dataset_name{i},'.npy'];
labelname = ['predict_result_',dataset_name{i},'.npy'];
data = readNPY(echoname);
le = readNPY(labelname);
%% Echo parameter: sample_size, fs, time & dist.
[p,m,n] = size(data);
p = p - 1;
fs = 400e3;
time = linspace(0,1e3*n/fs,n);
dist = (time*340/2)/1e3;
win_len = 16e-3*fs; % 16 ms
%% Load echo envelope information 
filename = ['time_window_information_',dataset_name{i},'.mat'];
load(filename);
%% Pick up the time-window and generate image
% echo_left = reshape(data(1:end-1,1,:),p,n);
echo_right = reshape(data(1:end-1,2,:),p,n);
marg = 2e-3*fs;
compare = @(a,b) min(a,b);
sample_size = 500;

disp('Right:');
foliage = 0;
gap = 0;
random = randperm(p,p);
cd('spectrogram_image_right');
for a = 1:p
    j = random(a);
    if foliage == sample_size && gap == sample_size
       break;
    end
    % get start point and end point
    if ~isnan(r_idx(j,1))
        start_point = r_idx(j,1)-marg;
        end_point = compare(r_idx(j,1)+win_len+marg, n-250); % 250 here is to avoid empty blank in right side of spectrogram.
    end
    % Normalize
    echo_part = echo_right(j,start_point:end_point);
    echo_part = (echo_part - min(echo_part))/(max(echo_part) - min(echo_part));
    wl_choose = linspace(2, length(echo_part), 5);
    % draw spectrogram 
    if ~isnan(mean(echo_part))
        
        if le(j)== 1
            if foliage == sample_size
                continue;
            end
            % original spectrogram
            fig = figure('visible', 'off');
            axes('Units', 'normalized', 'Position', [0 0 1 1]);
            spectrogram(echo_part,wl_choose(1),0,wl_choose(1),fs,'yaxis');
            %spectrogram(echo_part,320,222,320,fs,'yaxis');
            ylim([20 100]);
            caxis([-120 -50]);
            ax = gca;
            ax.Visible = 'off';
            colorbar('off');
            imgname = sprintf('%d_%s_%d_wl_1.jpg',le(j),dataset_name{i},j);
            saveas(fig,imgname);
            close(fig);
            
            % half overlap spectrogram
            fig = figure('visible', 'off');
            axes('Units', 'normalized', 'Position', [0 0 1 1]);
            spectrogram(echo_part,wl_choose(5),0,wl_choose(5),fs,'yaxis');
            %spectrogram(echo_part,320,222,320,fs,'yaxis');
            ylim([20 100]);
            caxis([-120 -50]);
            ax = gca;
            ax.Visible = 'off';
            colorbar('off');
            imgname = sprintf('%d_%s_%d_wl_5.jpg',le(j),dataset_name{i},j);
            saveas(fig,imgname);
            close(fig);
            
            disp(dataset_name{i});
            disp(j);
            foliage = foliage + 1;
        end
        if le(j)== 0
            if gap == sample_size
                continue;
            end
            % original spectrogram
            fig = figure('visible', 'off');
            axes('Units', 'normalized', 'Position', [0 0 1 1]);
            spectrogram(echo_part,wl_choose(1),0,wl_choose(1),fs,'yaxis');
            %spectrogram(echo_part,320,222,320,fs,'yaxis');
            ylim([20 100]);
            caxis([-120 -50]);
            ax = gca;
            ax.Visible = 'off';
            colorbar('off');
            imgname = sprintf('%d_%s_%d_wl_1.jpg',le(j),dataset_name{i},j);
            saveas(fig,imgname);
            close(fig);
            
            % half overlap spectrogram
            fig = figure('visible', 'off');
            axes('Units', 'normalized', 'Position', [0 0 1 1]);
            spectrogram(echo_part,wl_choose(5),0,wl_choose(5),fs,'yaxis');
            %spectrogram(echo_part,320,222,320,fs,'yaxis');
            ylim([20 100]);
            caxis([-120 -50]);
            ax = gca;
            ax.Visible = 'off';
            colorbar('off');
            imgname = sprintf('%d_%s_%d_wl_5.jpg',le(j),dataset_name{i},j);
            saveas(fig,imgname);
            close(fig);
            
            disp(dataset_name{i});
            disp(j);
            gap = gap + 1;
        end
        disp('foliage:');
        disp(foliage);
        disp( 'gap:');
        disp(gap);
    end
end
end