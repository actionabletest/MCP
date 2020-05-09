for i = 1:100
    imgPath = ['./' mat2str(i) '.png']; % 组合保存路径和图片名称
    imwrite(A,imgPath);                 % A假设就是所得到的待保存图片矩阵
end