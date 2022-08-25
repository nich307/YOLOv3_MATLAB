%% Object Detection Using YOLO v3
clear all; close all; clc

fps = 0;
color = [0,150,0];
imgSize = 608;
useMex = false;

%% Get the Pretrained DAGNetwork
% Download pretrained YOLOv3 (Keras implementation)
if ~exist('yolov3x608.mat','file') && ~exist('yolov3_detect_mex.mexw64','file')
    disp 'pretrained model missing!';
    return
end

% Load Pretrained YOLOv3 Tiny Network (yolov3tiny)
if ~useMex
    disp 'Loading YOLO v3 pretrained model, please wait...';
    load('yolov3x608.mat');
end

%%
% The DAG network contains 256 layers including convolution, ReLU, and 
% batch normalization layers along with the YOLO v3 transform and 
% YOLO v3 output layers. Use the command net.Layers to see all the layers of 
% the network.
%
 yolov3x608.Layers


%% Run the Generated MEX
% Set up the video file reader and read the input video. Create a video
% player to display the video and the output detections.
%
videoFile = 'highway_accidents.avi';
%videoFile = 'highway_lanechange.mp4';
videoFreader = vision.VideoFileReader(videoFile,'VideoOutputDataType','uint8');
%depVideoPlayer = vision.DeployableVideoPlayer('Size','Custom','CustomSize',[640 480]);
depVideoPlayer = vision.DeployableVideoPlayer('Size','Custom','CustomSize',[640 352]);

I = step(videoFreader);
in = im2single(I);

[img_h, img_w, ~] = size(in);

ratio = min(imgSize/img_w, imgSize/img_h);

% Image height and width after resizing image
w = round(img_w * ratio);
h = round(img_h * ratio);


%%
% Read the video input frame-by-frame and detect the vehicles in the video 
% using the detector.
%
cont = ~isDone(videoFreader);
while cont
    in = imresize4Yolo(in, imgSize, w, h);
    tic; % Count FPS
    if useMex
        predictions = yolov3_detect_mex(in);
    else
        predictions = yolov3_detect(in);
    end
    elapsedTime = toc;
    fps = .9*fps + .1*(1/elapsedTime);
    % post-processing and display the results
    out = postProcess(predictions, I, w, h);
    out = insertText(out, [1, 1],  sprintf('FPS %2.2f', fps), 'FontSize', 24);
    step(depVideoPlayer, out);

    I = step(videoFreader);
    in = im2single(I);
    cont = ~isDone(videoFreader) && isOpen(depVideoPlayer); % Exit the loop if the video player figure window is closed
end


%% Supporting Functions
function out = imresize4Yolo(img, imgSz, w, h)

%Resize Image
rimg = imresize(img, [h, w],'Method','bilinear','AntiAliasing',false);

st_h = round((imgSz - h)/2) + 1;
st_w = round((imgSz - w)/2) + 1;

%Creating background
if isfloat(img)
    out = ones(imgSz, imgSz, 3, 'like', img) * 0.5;
else
    out = ones(imgSz, imgSz, 3, 'like', img) * 128;
end

out(st_h:st_h+h-1, st_w:st_w+w-1, :) = rimg;
end
