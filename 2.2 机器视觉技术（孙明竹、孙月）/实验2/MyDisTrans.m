% ���庯��MyDisTrans������Ϊ��ֵͼ��bw 
function DisTrans_result=MyDisTrans(bw) 

% ��ȡbw������rows������cols 
[rows, cols] = size(bw); 

% ��bw��Χ���1Ȧ1����ÿ������ֵ��100
padsize = [1 1]; 
bw_a = padarray(bw, padsize, 1); 
bw_a = bw_a * 100; 

% ��(2,2)��(rows+1,cols+1)������bw_a��ÿ������ 
% ����al������ǰ���ؼ������ϡ����ϡ�����Ϸ�������ֵ�������ݾ�����϶�Ӧֵ��������СֵΪ��ǰ����ֵ 
for j = 2:cols+1 
    for i = 2:rows+1 
        al = [bw_a(i,j) bw_a(i-1,j-1)+1 bw_a(i+1,j-1)+1 bw_a(i,j-1)+1 bw_a(i-1,j)+1]; 
        bw_a(i,j) = min(al);
    end 
end 

% �����ٱ���һ��
for j = cols+1:-1:2 
    for i = rows+1:-1:2 
        bl = [bw_a(i,j) bw_a(i,j+1)+1 bw_a(i+1,j+1)+1 bw_a(i-1,j+1)+1 bw_a(i+1,j)+1]; 
        bw_a(i,j) = min(bl); 
    end 
end 

% ����bw_a���м䲿��
DisTrans_result = bw_a(2:rows+1,2:cols+1); 

end