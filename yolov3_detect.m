function predictions = yolov3_detect(in)

%   Copyright 2018-2019 The MathWorks, Inc.

% A persistent object yolov2Obj is used to load the YOLOv2ObjectDetector object.
% At the first call to this function, the persistent object is constructed and
% setup. When the function is called subsequent times, the same object is reused 
% to call detection on inputs, thus avoiding reconstructing and reloading the
% network object.
persistent yolov3x608;

if isempty(yolov3x608)
    yolov3x608 = coder.loadDeepLearningNetwork('yolov3x608.mat');
end

% pass in input
predictions = predict(yolov3x608, in);

