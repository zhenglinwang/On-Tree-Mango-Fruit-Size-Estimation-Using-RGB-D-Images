%% This code is to demonstrate the idea depicted in the paper "On-Tree Mango Fruit Size Estimation Using RGB-D Images"
%  if you use the code, please cite the paper:
%  Wang, Z., Walsh, K. B., & Verma, B. (2017). On-tree mango fruit size estimation using RGB-D images. Sensors, 17(12), 2738. 
%  this code has been tested on MATLAB R2017b (64bit) + windows 10

clc
close all
clear

addpath .\test_images

ret = 0;
a_len = 0;
b_width = 0;

%% the cascade detector was trained separately, using HOG or LBP
load('detector_LBP_box32_128_Merge_8.mat');

%% The following parameters come from camera calibration
fx_rgb =   1.0585e+03; %3.28135mm
fy_rgb =   1.1341e+03; %3.51571mm
cx_rgb =  965.1123;
cy_rgb =  583.2686;
fx_d =   1.1752e+03; %3.64132mm
fy_d =   1.2589e+03; %3.90295mm
cx_d =  849.6059;
cy_d =  726.8106;
extrinsics = [1.0000   -0.0055    0.0003  -78.2239;
    0.0055    1.0000    0.0002   31.3206;
   -0.0003   -0.0002    1.0000   -8.0845];

fx_rgb_len =   3.28135;
fy_rgb_len =   3.51571;
fx_d_len =   3.64132;
fy_d_len =   3.90295;


%% this trick is used to keep more decimals.
depthScale = 322;


jpegFiles = dir('test_images\\*.jpg'); 
numfiles = length(jpegFiles);
for k = 1:1: numfiles
    imName = fullfile(jpegFiles(k).folder,jpegFiles(k).name);
    RGB = imread(imName);
    figure, imshow(RGB)  
    
    filename = sprintf('%s.txt', jpegFiles(k).name(1:end-4));
    depName = fullfile(jpegFiles(k).folder, filename);
    fileID = fopen(depName);
    line1 = textscan(fileID, '%d','HeaderLines',0);
    fclose(fileID);
    depthArray = line1{1};
    A = imresize(reshape(depthArray, [512, 424])', [1365 1649], 'bilinear')*depthScale;

    % This is from camera calibration
    % refer to https://www.codefull.org/2016/03/align-depth-and-color-frames-depth-and-rgb-registration/
    [B, C] = depth_rgb_registration(A);
    
    figure, imagesc(B);

    BB = (B>2000 & B<3500);
    [BB] = filterStalk(BB, 2);
    HoleSize = 300;
    BB = bwareaopen(BB,HoleSize);
    BB = 1-BB;
    BB = bwareaopen(BB,HoleSize);
    BB = 1-BB;
    SE = strel('disk', 1);
    BB = imdilate(BB, SE);
    % RGB = fliplr(RGB);
    overlay = imoverlay(RGB, BB, 'yellow');
    figure, imagesc(overlay);

    % cut out margin areas
    A = B(:, 237:1717);
    RGB= RGB(:, 237:1717, :);
	A = fliplr(A);
    RGB= fliplr(RGB);
    
    RGB0 = RGB;
    img = RGB;
    [imWidth, imHeight, imChannel] = size(RGB);
    
    % detect potential fruit using cascade classifier
    bbox = step(detector,img); 
    % Insert bounding box rectangles and return the marked image.
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox, 'mango');
    figure; imagesc(detectedImg);
    title(['fruit: ', num2str(size(bbox, 1))]);
    
    bbox1 = bbox;
    remove_no = 0;
    for ii=1:1:size(bbox, 1)
        start_x = bbox(ii, 2);
        start_y = bbox(ii, 1);
        len_x = bbox(ii, 4);
        len_y = bbox(ii, 3);
        
        start_x = round(start_x - len_x*0.2);
        start_y = round(start_y - len_y*0.2);
        if (start_x<1)
            start_x = 1;
            continue;
        end
        if (start_y<1)
            start_y = 1;
            continue;
        end
        
        len_x = round(len_x*1.4);
        len_y = round(len_y*1.6);
        if ((start_x+len_x-1)>imWidth)
            len_x = imWidth-start_x-1;
            continue;
        end
        if ((start_y+len_y-1) > imHeight)
            len_y = imHeight-start_y-1;
            continue;
        end
           
        img_check = img(start_x:start_x+len_x-1, start_y:start_y+len_y-1, :);
        
        imGray = rgb2gray(img_check);
        level = graythresh(imGray);
        BW = im2bw(imGray,level*1.0);
        BW3 = uint8(cat(3, BW, BW, BW));
        RGB = img_check.*BW3;
        
        [BW, RGB] = createMask_universal_1_15_Lab(RGB);
        RGB = im2uint8(RGB);
        
        HoleSize = 200;
        BW = bwareaopen(BW,HoleSize);
        BW = 1-BW;
        BW = bwareaopen(BW,HoleSize);
        BW = 1-BW;
        BW3 = uint8(cat(3, BW, BW, BW));
        RGB = img_check.*BW3;
        
        [BW] = filterStalk(BW, 7);
        BW3 = uint8(cat(3, BW, BW, BW));
        RGB = img_check.*BW3;
        
        HoleSize = 400;
        BW = bwareaopen(BW,HoleSize);
        BW = 1-BW;
        BW = bwareaopen(BW,HoleSize);
        BW = 1-BW;
        
        SE = strel('disk', 1);
        BW = imdilate(BW, SE);
        BW3 = uint8(cat(3, BW, BW, BW));
        RGB = img_check.*BW3;
    
        CC = bwconncomp(BW);
        for i0 =1:CC.NumObjects
            BW1 = BW*0;
            BW1(CC.PixelIdxList{i0}) = 1;

            valid_area = length(find(BW1(:)));
            if length(find(BW1(:)))<500 || length(find(BW1(:)))>8000
                BW(CC.PixelIdxList{i0}) = 0;
                continue;
            end

            props = regionprops(BW1,'Area', 'MajorAxisLength','MinorAxisLength', ...
                'BoundingBox', 'Orientation', 'Eccentricity', 'Centroid');
            if length(props)<1
                continue;
            end

            for i1 = 1:1:length(props) 
                s = props(i1);
                cx = s.Centroid(1);
                cy = s.Centroid(2);

                BW2 = BW*0;
                if (ceil(s.BoundingBox(2)+s.BoundingBox(4)) > size(BW2, 1) ...
                        || ceil(s.BoundingBox(1)+s.BoundingBox(3)) > size(BW2, 2))
                    BW(ceil(s.BoundingBox(2)) : ceil(s.BoundingBox(2)+s.BoundingBox(4)), ...
                     ceil(s.BoundingBox(1)): ceil(s.BoundingBox(1)+s.BoundingBox(3)) ) = 0;
                    continue;
                end

                BW2(ceil(s.BoundingBox(2)) : ceil(s.BoundingBox(2)+s.BoundingBox(4)), ...
                     ceil(s.BoundingBox(1)): ceil(s.BoundingBox(1)+s.BoundingBox(3)) ) = 1;
                Area_e = round(pi*(s.MajorAxisLength)*(s.MinorAxisLength)/4);
                area_ratio = s.Area/Area_e;
                thisBB = s.BoundingBox;

                depth_thres = 1400;
                cy = start_x+round(cy);
                cx = start_y + round(cx);
                rec_x = start_y+thisBB(1);
                rec_y = start_x+thisBB(2);
                distance_dep = A(cy, cx);
                if(distance_dep < depth_thres)
                    distance_dep = A(cy-1, cx);
                end
                if(distance_dep < depth_thres)
                    distance_dep = A(cy+1, cx);
                end
                if(distance_dep < depth_thres)
                    distance_dep = A(cy, cx-1);
                end
                if(distance_dep < depth_thres)
                    distance_dep = A(cy, cx+1);
                end
                if((distance_dep < depth_thres) || (distance_dep > 3500))
                    disp(['Wrong depth: ',  num2str(distance_dep)]);
                    continue;
                end
                len_pxl = round(s.BoundingBox(4));
                wid_pxl = round(s.BoundingBox(3));
                 
                % fruit not satisfied with the conditions are excluded for
                % sizing
                if ((area_ratio >= 0.99) && (s(1).Eccentricity < 0.70)) %0.9 0.8 vs 0.98 0.75
                    if (len_pxl>= floor(s(1).MajorAxisLength)+2)
                        disp(['error: ', jpegFiles(k).name, ' len_pxl: ', num2str(len_pxl), ' MajorAxisLength: ', num2str(s(1).MajorAxisLength)])
                        continue;
                    end
                    
                    cal_len = round(len_pxl*3.1*1e-3*distance_dep/fx_rgb_len);
                    cal_width = round(wid_pxl*3.1*1e-3*distance_dep/fy_rgb_len);

                    figure('units','normalized','outerposition',[0 0 1 1]), imagesc(RGB0), axis square, hold on;
                    plot(round(cx), round(cy), 'r+', 'LineWidth', 1);
                    rectangle('Position', [rec_x+1,rec_y-1,wid_pxl,len_pxl],'EdgeColor','r','LineWidth',1 );
                    title([' Eccentricity: ', num2str(s.Eccentricity), ...
                       ' valid_area: ', num2str(valid_area),...
                       ' area_ratio: ', num2str(area_ratio),...
                       ' cental x: ', num2str(round(cx)), ...
                       ' cental y: ', num2str(round(cy)),...
                       ' length pixel: ', num2str(len_pxl), ...
                       ' widht pixel: ', num2str(wid_pxl),...
                      '  depth: ', num2str(distance_dep), ...
                       ' box length: ', num2str(round(len_pxl*3.1*1e-3*distance_dep/fx_rgb_len)), ...
                       ' box widht: ', num2str(round(wid_pxl*3.1*1e-3*distance_dep/fy_rgb_len))]);
                     hold off;
                end
            end
        end
    end
end

return;



