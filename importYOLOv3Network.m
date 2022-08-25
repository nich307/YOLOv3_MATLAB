function yolov3x608 = importYOLOv3Network()
%% Import Darknet Network YOLOv3
%  you can download weights from https://pjreddie.com/media/files/yolov3.weights
%    and cfg file from https://github.com/pjreddie/darknet/tree/master/cfg
%    or you can refer to the live script of darknet add-on
%clear all;close all;clc  % clear all before import to ensure layer name be correct
[lgraph, anchors, classes] = importDarknetNetwork('yolov3.weights', 'yolov3.cfg');

%% Replace Upsampling2d (custom) Layer to TransposedConv2d for codegen
layer = transposedConv2dLayer(2, 256, 'Stride', 2, 'Name', 'up_sampling2d_1');
layer.Weights = zeros(2,2,256,256);
for i = 1:256
    idx = (i-1)*2*2*256 + (i-1)*2*2+1;
    layer.Weights(idx:idx+3) = 1;
end
layer.Bias = zeros(1,1,256);
lgraph = replaceLayer(lgraph, 'upsampling_layer_1', layer);

layer = transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'up_sampling2d_2');
layer.Weights = zeros(2,2,128,128);
for i = 1:128
    idx = (i-1)*2*2*128 + (i-1)*2*2+1;
    layer.Weights(idx:idx+3) = 1;
end
layer.Bias = zeros(1,1,128);
lgraph = replaceLayer(lgraph, 'upsampling_layer_2', layer);

%% Add Zero padding for Scale3 output
lname = 'conv2d_59';

layer = transposedConv2dLayer(4, 255, 'Stride', 4, 'Name', 'zero_padding2d_l1');
layer.Weights = zeros(4,4,255,255);
for i = 1:255
    idx = (i-1)*4*4*255 + (i-1)*4*4+1;
    layer.Weights(idx:idx+15) = 1;
end
layer.Bias = zeros(1,1,255);
lgraph2 = addLayers(lgraph, layer);
lgraph2 = connectLayers(lgraph2, lname, 'zero_padding2d_l1');
%analyzeNetwork(lgraph2)

%% Add Zero padding for Scale2 output
lname = 'conv2d_67';

layer = transposedConv2dLayer(2, 255, 'Stride', 2, 'Name', 'zero_padding2d_l2');
layer.Weights = zeros(2,2,255,255);
for i = 1:255
    idx = (i-1)*2*2*255 + (i-1)*2*2+1;
    layer.Weights(idx:idx+3) = 1;
end
layer.Bias = zeros(1,1,255);
lgraph2 = addLayers(lgraph2, layer);
lgraph2 = connectLayers(lgraph2, lname, 'zero_padding2d_l2');
%analyzeNetwork(lgraph3)

%% Add Depth concatenation Layer
lname = 'conv2d_75';

layer = depthConcatenationLayer(3, 'Name','concatenate_all');
lgraph2 = addLayers(lgraph2, layer);
lgraph2 = connectLayers(lgraph2,'zero_padding2d_l1', 'concatenate_all/in1');
lgraph2 = connectLayers(lgraph2,'zero_padding2d_l2', 'concatenate_all/in2');
lgraph2 = connectLayers(lgraph2, lname, 'concatenate_all/in3');

%% Add Regression Layer
lgraph2 = lgraph2;
layer = regressionLayer('Name','output')
lgraph2 = addLayers(lgraph2, layer);
lgraph2 = connectLayers(lgraph2,'concatenate_all', 'output');

%% Assemble deep learning network 
yolov3x608 = assembleNetwork(lgraph2);
%analyzeNetwork(lgraph)
% save the output (yolov3x608) to mat file : yolov3x608.mat for code gen or inference usage.
end