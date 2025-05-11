% 定义函数MyInteImg，参数为二值图像img 
function MyInteImg_result=MyInteImg(img)

img = double(img);
[rows, cols] = size(img); 

% 先行方向累加和
s = zeros(rows, cols);
for i = 1:rows
    s(i,1) = img(i,1);
    for j = 2:cols 
        s(i,j) = s(i,j - 1) + img(i,j);
    end 
end 

% 再列方向累加和
ii = zeros(rows, cols);
for j = 1:cols 
    ii(1,j) = s(1,j);
    for i = 2:rows
        ii(i,j) = ii(i - 1,j) + s(i,j);
    end 
end 

% 返回值
MyInteImg_result = ii; 

end