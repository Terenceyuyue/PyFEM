function showMnist(data_x)
% data_x: 784 * N, ÿ�ж�Ӧһ��ͼƬ
% �� N>25, �����ʾǰ 25 ��ͼ

if nargin==0
    clc;clear;close all
    mdata = load('mnistdata');
    data_x = mdata.test_data_x;
end

N = size(data_x,2);

figure,
set(gcf,'Units','normal');
set(gcf,'Position',[0.0,0.0,0.35,0.55]);

m = 5; n = 5;
for i = 1:min(N,m*n)
    a = data_x(:,i);
    a = reshape(a,28,28)'; % transpose
    subplot(m,n,i), imshow(a);
end