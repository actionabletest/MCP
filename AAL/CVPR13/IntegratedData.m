% ΪʲôҪ�������ݣ��� -- ��ΪΪ�˳��ʹ�ð����Ʒ�������cpu������Դ������ÿ̨����������������matlab��ͬʱ��һ���������ݼ�
name_str = ["bim-a", "bim-b", "brightness", "contrast", "cw-l2", "fgsm",  "scale", "shear", "translation", "rotation" ,"jsma"];


% current_dir_1 = strcat("mnist_mutualInfo\mnist_", name_str(1), "_compound8_mutualInfo", num2str(1));
for i =1:10
    load(strcat("cifar_mutualInfo\cifar_", name_str(i), "_compound8_mutualinfo", num2str(1)));
    mutualinfo1 = mutualinfo;
    clearvars mutualinfo;
    load(strcat("cifar_mutualInfo\cifar_", name_str(i), "_compound8_mutualinfo", num2str(2)));
    mutualinfo2 = mutualinfo;
    clearvars mutualinfo;
    mutualinfo = [cell2mat(mutualinfo1), cell2mat(mutualinfo2)];
    save(strcat("cifar_finalmutualinfo\cifar_", name_str(i),"_compound8_mutualinfo"),"mutualinfo");
end

load(strcat("cifar_mutualInfo\cifar_", name_str(11), "_compound8_mutualinfo", num2str(1)));
mutualinfo1 = mutualinfo;
clearvars mutualinfo;
load(strcat("cifar_mutualInfo\cifar_", name_str(11), "_compound8_mutualinfo", num2str(2)));
mutualinfo2 = mutualinfo;
clearvars mutualinfo;
load(strcat("cifar_mutualInfo\cifar_", name_str(11), "_compound8_mutualinfo", num2str(3)));
mutualinfo3 = mutualinfo;
clearvars mutualinfo;
load(strcat("cifar_mutualInfo\cifar_", name_str(11), "_compound8_mutualinfo", num2str(4)));
mutualinfo4 = mutualinfo;
clearvars mutualinfo;

mutualinfo = [cell2mat(mutualinfo1), cell2mat(mutualinfo2),cell2mat(mutualinfo3),cell2mat(mutualinfo4)];
save(strcat("cifar_finalmutualinfo\cifar_", name_str(11),"_compound8_mutualinfo"),"mutualinfo");


