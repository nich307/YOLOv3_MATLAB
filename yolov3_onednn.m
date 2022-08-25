%% Object Detection Using YOLO v3 608x608
function out = yolov3_onednn()
    %% Update buildinfo with the OpenCV library flags.
    %opencv_link_flags = '`pkg-config --cflags --libs opencv`'; % opencv 3
    %opencv_link_flags = '`pkg-config --cflags --libs opencv4`'; % opencv 4
    %coder.updateBuildInfo('addLinkFlags',opencv_link_flags);
    %coder.inline('never');
 
    % Connect to webcam
    videoFile = 'highway_accidents.avi';
    videoFreader = vision.VideoFileReader(videoFile,'VideoOutputDataType','uint8');
    player = vision.DeployableVideoPlayer('Size','Custom','CustomSize',[1280 720]);
    
    %%
    orgImg = step(videoFreader);
    orgImg = im2single(orgImg);
    [img_h, img_w, ~] = size(orgImg);
    %step(player, orgImg);
    
    %%
    imgSize = 608;
    out = zeros([img_h img_w 3], 'uint8');

    ratio = min(imgSize/img_w, imgSize/img_h);

    % Image height and width after resizing image
    w = round(img_w * ratio);
    h = round(img_h * ratio);
    st_h = round((imgSize - h)/2) + 1;
    st_w = round((imgSize - w)/2) + 1;

    fps = 0;
    cont = ~isDone(videoFreader);
    while cont
        orgImg = step(videoFreader);
        %orgImg = fliplr(orgImg);
        in = im2single(orgImg);
        % img = imadjust(img, stretchlim(img,[0.01,0.80]));
        % img = histeq(img);
        %Creating background
        in3 = ones(imgSize, imgSize, 3, 'like', in) * 0.5;
        in2 = imresize(in, [h, w]); %,'Method','bilinear','AntiAliasing',false);
        in3(st_h:st_h+h-1, st_w:st_w+w-1, :) = in2;

        tic; % Count FPS
        predictions = yolov3_detect(in3);
        elapsedTime = toc;
        fps = .9*fps + .1*(1/elapsedTime);

        % post-processing and display the results
        out = postProcess(predictions, orgImg, w, h);
        out = insertText(out, [1, 1],  sprintf('FPS %2.2f', fps), 'FontSize', 26, 'BoxColor', [0,150,0]);
        out = imresize(out, [img_h img_w]);
        step(player, out);
        cont = ~isDone(videoFreader);% && isOpen(player);
    end
    player.release();
end
