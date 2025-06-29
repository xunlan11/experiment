% 定义函数MyDisTrans，参数为二值图像bw 
function DisTrans_result=MyDisTrans(bw) 

% 获取bw的行数rows和列数cols 
[rows, cols] = size(bw); 

% 在bw周围填充1圈1，将每个像素值乘100
padsize = [1 1]; 
bw_a = padarray(bw, padsize, 1); 
bw_a = bw_a * 100; 

% 从(2,2)到(rows+1,cols+1)，遍历bw_a的每个像素 
% 数组al包含当前像素及其左上、右上、左和上方的像素值，并根据距离加上对应值，其中最小值为当前像素值 
for j = 2:cols+1 
    for i = 2:rows+1 
        al = [bw_a(i,j) bw_a(i-1,j-1)+1 bw_a(i+1,j-1)+1 bw_a(i,j-1)+1 bw_a(i-1,j)+1]; 
        bw_a(i,j) = min(al);
    end 
end 

% 反向再遍历一次
for j = cols+1:-1:2 
    for i = rows+1:-1:2 
        bl = [bw_a(i,j) bw_a(i,j+1)+1 bw_a(i+1,j+1)+1 bw_a(i-1,j+1)+1 bw_a(i+1,j)+1]; 
        bw_a(i,j) = min(bl); 
    end 
end 

% 返回bw_a的中间部分
DisTrans_result = bw_a(2:rows+1,2:cols+1); 

end