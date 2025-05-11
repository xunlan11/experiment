clc;clear all;close all;

parent = 'testimg';
parentPath = dir(parent);
len = size(parentPath);
result = zeros(len-2);
for i = 1:len-2
    childPath = fullfile(parent, parentPath(i+2).name);
    img1 = imread(fullfile(childPath, '1'), 'jpg');
    img2 = imread(fullfile(childPath, '2'), 'jpg');
    result(i) = test(img1, img2);  
end
disp(result')