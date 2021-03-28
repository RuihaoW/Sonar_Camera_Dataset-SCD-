%% This code is to 1) find the time-window for spectrogram
%                  2) generate the spectrogram images as input to CNN
%  Based on the boxplot of the overall time-window length, we chose the
%  maximum window-length as the final length for all 10 datasets. The
%  maximum window length is 16 ms when threshold is 10xnoise_std.
clear all;
close all
%% Dataset path
cd('F:\College\MachineLearning\ShortDistanceProfile\echo_result_02_23_2021');
% /home/lyimeng/ondemand/data/sys/dashboard/batch_connect/sys/bc_vt_matlab_cascades/Profile/echo_result_02_23_2021/
dataset_name = {'2_a','4_b','6_a','7_b','9_a','9_d_1','9_d_2','10_a','12_a','13_c'};
%'2_a','4_b','6_a','7_b','9_a','9_d_1','9_d_2','10_a','12_a','13_c'
index = readNPY('index.npy');
% dataset index
d = 0;
count = 0;
%%
if ~exist('spectrogram_image_left', 'dir')
    mkdir('spectrogram_image_left')
end
if ~exist('spectrogram_image_right', 'dir')
    mkdir('spectrogram_image_right')
end
%%
for j = 1:length(index)
    if (index(j) == -1)
        d = d + 1;
        if d > length(dataset_name)
            break;
        end
        cd('F:\College\MachineLearning\ShortDistanceProfile\echo_result_02_23_2021');
        echoname = ['filted_echo_Outdoor_',dataset_name{d},'.npy'];
        labelname = ['predict_result_',dataset_name{d},'.npy'];
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
        filename = ['time_window_information_',dataset_name{d},'.mat'];
        load(filename);
        %% Pick up the time-window and generate image
        % echo_left = reshape(data(1:end-1,1,:),p,n);
        echo_right = reshape(data(1:end-1,2,:),p,n);
        marg = 2e-3*fs;
        compare = @(a,b) min(a,b);
        count = 0;
        cd('spectrogram_image_right');
        continue;
    end
    
    if ~isnan(r_idx(index(j),1))
        start_point = r_idx(index(j),1)-marg;
        end_point = compare(r_idx(index(j),1)+win_len+marg, n-250); % 250 here is to avoid empty blank in right side of spectrogram.
    end
    % Normalize
    echo_part = echo_right(index(j),start_point:end_point);
    echo_part = (echo_part - min(echo_part))/(max(echo_part) - min(echo_part));
    wl_choose = linspace(2, length(echo_part), 5);
    disp(length(echo_part))
    % draw spectrogram
    if ~isnan(mean(echo_part))
        % original spectrogram
        fig = figure('visible', 'off');
        axes('Units', 'normalized', 'Position', [0 0 1 1]);
        spectrogram(echo_part,256,0,256,fs,'yaxis');
        %spectrogram(echo_part,320,222,320,fs,'yaxis');
        ylim([20 100]);
        caxis([-120 -50]);
        ax = gca;
        ax.Visible = 'off';
        colorbar('off');
        imgname = sprintf('%d_%s_%d_wl_180.jpg',le(index(j)),dataset_name{d},index(j));
        saveas(fig,imgname);
        disp(imgname);
        close(fig);
        count = count + 1;

        disp(dataset_name{d});
        disp(j);
        
        disp('Count:');
        disp(count);
    end
end