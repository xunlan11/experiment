% 定义函数test，参数为图像img1、img2
function result=test(img1,img2)

% 转为灰度图 
img1 = rgb2gray(img1);  
img2 = rgb2gray(img2);  
   
% surf特征检测，匹配并对齐图像  
points1 = detectSURFFeatures(img1);  
points2 = detectSURFFeatures(img2);  
[features1, validPoints1] = extractFeatures(img1, points1);  
[features2, validPoints2] = extractFeatures(img2, points2);  
indexPairs = matchFeatures(features1, features2);   
matchedPoints1 = validPoints1(indexPairs(:, 1));  
matchedPoints2 = validPoints2(indexPairs(:, 2));   
[tform, inlierIdx] = estimateGeometricTransform2D(matchedPoints2, matchedPoints1, 'rigid');  
%{
% 匹配图像
inliers1 = matchedPoints1(inlierIdx, :);  
inliers2 = matchedPoints2(inlierIdx, :);  
figure
showMatchedFeatures(img1, img2, inliers1, inliers2, 'montage');  
%}
img2_new = imwarp(img2, tform, 'OutputView', imref2d(size(img1))); 
%{
% 对齐后的图像
figure
imshow(img2_new)
%}

% 填补移动后剩下的背景，并差分，back统计黑色像素数量（包括移动后剩下的背景和原来就是黑色的像素）
[row,col] = size(img1);
back = 0;
for i=1:row
    for j=1:col
        if img2_new(i,j)==0
            img2_new(i,j)=img1(i,j);
            back = back + 1;
        end
    end
end
img_12 = img1 -img2_new;
%{
% 差分图像
figure
imshow(img_12)
%}

% 阈值化，counts统计白色像素数量
counts = 0;
for i=1:row
    for j=1:col
        if img_12(i,j)>70
            img_12(i,j)=255;
            counts = counts + 1;
        else
            img_12(i,j)=0;
        end
    end
end
%{
% 阈值化图像
figure
imshow(img_12)
%}

if counts>5000
    result = 0;
else
    result = 1;
end

end