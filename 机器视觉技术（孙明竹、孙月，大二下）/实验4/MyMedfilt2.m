% 定义函数MyMedfilt2，参数为二值图像img
function Medfilt2_result=MyMedfilt2(img)

% 获取img的行数rows和列数cols 
[rows, cols] = size(img); 

% 3x3掩模
img_0 = zeros(rows, cols);
for i=2:rows - 1
    for j=2:cols - 1
        A=[img(i-1,j-1),img(i-1,j),img(i-1,j+1),img(i,j-1),img(i,j),img(i,j+1),img(i+1,j-1),img(i+1,j),img(i+1,j+1)];
        img_0(i,j)=median(A);
    end
end

% 返回值
Medfilt2_result=img_0(2:rows - 1,2:cols - 1);

end