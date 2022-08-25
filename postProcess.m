function I = postProcess(predictions, img, w, h)
   %% 
    classes = { 'Person', 'Bicycle', 'Car', 'Motorbike', 'Aeroplane', 'Bus',... 
        'Train', 'Truck', 'Boat', 'Traffic light', 'Fire hydrant', 'Stop sign',...
        'Parking meter', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep',...
        'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack', 'Umbrella',...
        'Handbag', 'Tie', 'Suitcase', 'Frisbee', 'Skis', 'Snowboard',...
        'Sports ball', 'Kite', 'Baseball bat', 'Baseball glove', 'Skateboard',...
        'Surfboard', 'Tennis racket', 'Bottle', 'Wine glass', 'Cup', 'Fork',...
        'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple', 'Sandwich', 'Orange',...
        'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut', 'Cake', 'Chair',...
        'Sofa', 'Pottedplant', 'Bed', 'Diningtable', 'Toilet', 'Tvmonitor',...
        'Laptop', 'Mouse', 'Remote', 'Keyboard', 'Cell phone', 'Microwave',...
        'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book', 'Clock', 'Vase',...
        'Scissors', 'Teddy bear', 'Hair drier', 'Toothbrush' };

    
    label_str = '';
    scores = zeros([200 1],'single');
    labels = zeros([200 1],'single');
    
    %% Separate predictions
    sz = size(predictions);
    scale3 = predictions(1:sz(1), 1:sz(2), 511:end);
    scale2 = predictions(1:2:sz(1), 1:2:sz(2), 256:510);
    scale1 = predictions(1:4:sz(1), 1:4:sz(2), 1:255);

    %% Set parameters for post-processing
    imSz = size(img);
    % Image size
    inputW = 608;
    inputH = 608;
    % number of classes (COCO dataset). 
    numClasses = 80;
    % X,Y,W,H,Score (5 elements).
    numElements = 5;
    % Ancers per Scale
    numAnchors = 3;
    % AnchorBoxes(3 different scales)
    % anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    Anchors1 = [116,90;156,198;373,326];
    Anchors2 = [30,61;62,45;59,119];
    Anchors3 = [10,13;16,30;33,23];
    % threshold for predictions
    thresh = 0.5;
    nms = 0.5;

    %% Transform Predictions Array (ex. 13x13x255 > (13x13x3) x 85)
    % Transform Predictions Array( Box Co-ordinates, Objectness Score and Class Scores )
    fmap1 = transformPredictions(scale1, numAnchors, numClasses, numElements);
    fmap2 = transformPredictions(scale2, numAnchors, numClasses, numElements);
    fmap3 = transformPredictions(scale3, numAnchors, numClasses, numElements);

    %% Apply sigmoid to constraint its possible offset range
    fmap1 = applySigmoid(fmap1);
    fmap2 = applySigmoid(fmap2);
    fmap3 = applySigmoid(fmap3);

    %% Calculate Bounding Box
    sz1 = size(scale1);
    sz2 = size(scale2);
    sz3 = size(scale3);
    % ?
    fmap1 = calculateDetections(sz1,fmap1,Anchors1,inputW,inputH);
    fmap2 = calculateDetections(sz2,fmap2,Anchors2,inputW,inputH);
    fmap3 = calculateDetections(sz3,fmap3,Anchors3,inputW,inputH);

    %% Marge Detections
    detections = [fmap1;fmap2;fmap3];

    %% Filtering with the threshold
    objectnessTmp = detections(:,5);
    detections = detections(objectnessTmp>thresh,:);
    objectness = detections(:,5);
    bboxes = [];

    %?
    if ~isempty(detections)
        classProbs = detections(:,6:end);
        tmpSz = size(classProbs,2);
        tmpObj = repmat(objectness,1,tmpSz);
        classProbs = classProbs.*tmpObj;

        idx = classProbs > thresh;
        classProbs(~idx) = single(0);

        [idxa,idxb,probs] = find(classProbs);
        if size(classProbs,1)==1
            detections = [detections(idxa',1:4),probs',idxb'];
        else
            detections = [detections(idxa,1:4),probs,idxb];
        end
        if ~isempty(detections)
            % BBox 
            % Extract tx, ty, tw and th for Bounding Box
            bboxes = [detections(:,1),detections(:,2),detections(:,3),detections(:,4)];

            % Bounding Box ?
            % Scale the size of BBox to align with imput image size
            bboxes = scaleBboxes(bboxes,imSz(1:2),inputW, inputH, w, h);

            bboxes = convertToXYWH(bboxes);
            scores = detections(:,5);
            labels = detections(:,6:end);

            % 
            % Clip the bounding box when it is positioned outside the image
            bboxes = vision.internal.detector.clipBBox(bboxes, imSz(1:2));

            % 
            idx = all(bboxes>=1,2);
            bboxes = bboxes(idx,:);
            scores = scores(idx,:);
            labels = labels(idx,:);

            if ~isempty(bboxes)
                % Nonmaximal suppression to eliminate overlapping bounding boxes
                [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxes, scores, labels ,...
                    'RatioType', 'Union', 'OverlapThreshold', nms);
            end
        end
    end

    %% 
    if ~isempty(bboxes)
        for ii=1:size(scores,1)
            class = char(classes{labels(ii,1)});
            label_str = char([class ' ' sprintf('%0.0f',scores(ii)*100) '%']);
            img = insertObjectAnnotation(img, 'rectangle', bboxes(ii,:), label_str, ...
                'TextBoxOpacity', 0.9, 'FontSize', 10);
        end
    end
    I = img;
end

%% Supporting Functions
function tPred = transformPredictions(fmap,numAnchors,numClasses,numElements)
    sz = size(fmap);
    %13x13x255
    tmpArray = permute(fmap,[2,1,3]);
    %169x85x3
    tmpArray = reshape(tmpArray,sz(1)*sz(2),numClasses+numElements,numAnchors);
    %85x169x3
    tmpArray = permute(tmpArray,[2,1,3]);
    tmpSz = size(tmpArray);
    %85x507
    tmpArray = reshape(tmpArray,tmpSz(1),tmpSz(2)*tmpSz(3));
    %507x85
    tPred = permute(tmpArray,[2,1,3]);

end

function sPred = applySigmoid(tPred)

    sigmoid = @(x) 1./(1+exp(-x));
    xy = sigmoid(tPred(:,1:2));
    wh = exp(tPred(:,3:4));
    scores = sigmoid(tPred(:,5:end));

    sPred = [xy, wh, scores];

end

function fmap = calculateDetections(sz,fmap,anchors,inputW,inputH)

    base = [0:sz(2)-1]';
    numAnchors = size(anchors,1);

    colIndexes = repmat(repmat(base,sz(1),1),numAnchors,1);
    rowIndexes = repmat(repelem(base,sz(1),1),numAnchors,1);
    anchors = repelem(anchors,sz(1)*sz(2),1);

    x = (colIndexes + fmap(:,1))./sz(2);
    y = (rowIndexes + fmap(:,2))./sz(1);
    w = fmap(:,3).*anchors(:,1)./inputW;
    h = fmap(:,4).*anchors(:,2)./inputH;
    r = fmap(:,5:end);

    fmap = [x,y,w,h,r];
end

% 
function dets = convertToXYWH(dets)
    dets(:,1) = dets(:,1)- dets(:,3)/2 + 0.5;
    dets(:,2) = dets(:,2)- dets(:,4)/2 + 0.5;
end

% w & h: image width & hight on predict
% ow & oh: actual image width & hight before predict but after resize
function bboxes = scaleBboxes(bboxes,imSz,w,h,ow,oh)
    dh = round((h - oh)/2+1);
    dw = round((w - ow)/2+1);

    bboxes(:,1) = (bboxes(:,1).*w - dw) .* (imSz(1,2)/ow);
    bboxes(:,2) = (bboxes(:,2).*h - dh) .* (imSz(1,1)/oh);
    bboxes(:,3) = (bboxes(:,3).*w ) .* (imSz(1,2)/ow);
    bboxes(:,4) = (bboxes(:,4).*h ) .* (imSz(1,1)/oh);

end
