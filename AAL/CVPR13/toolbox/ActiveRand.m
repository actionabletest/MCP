function randresult=activerand(labelset,unlabelset,testset,bnum, repeatnum)
%
% Active learning with random selection strategy
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column

lambda = 0.1;
ut=size(unlabelset,1);
[lt,d]=size(labelset);
classes = unique(testset(:,d));
classnum = length(classes);

id=0;
    w=logregtrain(labelset,lambda);
    [accu,yvec]=logregclassifynew(testset,w,classnum,classes);
    randresult.activeid(1)=id;
    randresult.accu(1)=accu;

for i=2:repeatnum+1
 
for bi = 1:bnum
    tmp = randperm(ut);	
    id = tmp(1);
    lt=lt+1;
    ut = ut -1;
    labelset(lt,:)=unlabelset(id,:);
unlabelset(id,:)=[];
end

    w=logregtrain(labelset,lambda);
    [accu,yvec]=logregclassifynew(testset,w,classnum,classes);
    randresult.activeid(i)=id;
    randresult.accu(i)=accu;
   % randresult.yvec(i)=yvec;

end

