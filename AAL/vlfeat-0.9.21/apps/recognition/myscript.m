for i = 1:100
    imgPath = ['./' mat2str(i) '.png']; % ��ϱ���·����ͼƬ����
    imwrite(A,imgPath);                 % A����������õ��Ĵ�����ͼƬ����
end