% ���庯��MyMedfilt2������Ϊ��ֵͼ��img
function Medfilt2_result=MyMedfilt2(img)

% ��ȡimg������rows������cols 
[rows, cols] = size(img); 

% 3x3��ģ
img_0 = zeros(rows, cols);
for i=2:rows - 1
    for j=2:cols - 1
        A=[img(i-1,j-1),img(i-1,j),img(i-1,j+1),img(i,j-1),img(i,j),img(i,j+1),img(i+1,j-1),img(i+1,j),img(i+1,j+1)];
        img_0(i,j)=median(A);
    end
end

% ����ֵ
Medfilt2_result=img_0(2:rows - 1,2:cols - 1);

end