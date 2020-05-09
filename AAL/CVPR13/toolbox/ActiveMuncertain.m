function result=activeuncertain(labelset,unlabelset,testset,bnum, repeatnum)
%
% Active learning while selecting query that has largest entropy on P(y|x)
%
% Y. Guo
% Jan. 8, 2006
%
% data: t*(d+1), the last column is the label column
lambda = 0.1;
ut=size(unlabelset,1);
[lt,d]=size(labelset);
classes=unique(testset(:,d));
classnum = length(classes);
ulist=1:ut;

id=0;

 w=logregtrain(labelset,lambda);
    [accu,yvec]=logregclassifynew(testset,w,classnum,classes);
    result.activeid(1)=id;
    result.accu(1)=accu;
for i=2:repeatnum+1
    for bi = 1:bnum
        id=pickuncertainquery(ulist,unlabelset,w,classnum);

        lt=lt+1;
        labelset(lt,:)=unlabelset(id,:);

        ulist(id)=0;
    end
    w=logregtrain(labelset,lambda);
    [accu,yvec]=logregclassifynew(testset,w,classnum,classes);
    result.activeid(i)=id;
    result.accu(i)=accu;

end

