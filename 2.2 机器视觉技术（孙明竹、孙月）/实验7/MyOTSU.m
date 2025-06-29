% 定义函数MyOTSU，参数为二值图像img 
function OTSU_result=MyOTSU(img)

% 归一化直方图  
counts = imhist(img); 
num = sum(counts);
probabilities = counts / num;

index = find(probabilities ~= 0);
min_index = min(index);
max_index = max(index);

% 遍历
maxv = -inf;  
tbest = 0; 
u = zeros(1,256);
for i = min_index : max_index
    u(i) = u(i-1) + probabilities(i);
end
u(max_index : 256) = 1;
for t = min_index : max_index  
    wB = sum(probabilities(min_index : t));
    wF = 1 - wB;   
    uB = u(t) / wB;  
    uF = (u(max_index) - u(t)) / wF;  
    v = wB *(uB - u(max_index)).^2 + wF *(uF - u(max_index)).^2; 
    if v > maxv  
        maxv = v;  
        tbest = t;  
    end  
end  

img(img > tbest) = 255; 
img(img < tbest+1) = 0; 

OTSU_result=img;

end