function [result selections]=activecombsample(Lset,Uset,Tset, beta, options, repeatnum, subsampling, indicator)
%
% combine most uncertain with mutual information weight
% can address multiclass problem, but only for non-batch mode active learning
%
% input:
% Lset: lt x (d+1), labeled set, the last column is the label column
% Uset: ut x (d+1), unlabeled set, the last column is the label column
% Tset: tt x (d+1), test set, the last column is the label column
% beta: (0,1)  the trade off between the uncertainty term and the mutual information term
% options: parameter used for produce kernel matrix
% repeatnum: how many active selections to conduct
%
% subsampling: method to select a subset as candidate
% subsampling.method
%       0: use all
%	1: top most uncertain ones
%	2: clustering
%	3: random selection
% subsampling.num
%	number of candidates
%
% Xin Li
%

lambda = 0.1;
[ut, d] = size(Uset);
d = d-1; % num. of features
lt = size(Lset, 1);
classes = unique(Tset(:,d+1));
classnum = length(classes);
uK = getkernel(Uset(:,1:d), Uset(:,1:d),options);

ulist = 1:ut;
U = uK;
%invU = inv(U+1e-8*eye(ut));
invU = pinv(U);

%%initial evaluate%%%%%%%%%%
w=logregtrain(Lset,lambda);
[accu,~,~]=logregclassifynew(Tset,w,classnum,classes);
result.accu(1)=accu;
result.subsampling = subsampling;

if indicator==0
    infodensity = zeros(ut,1);
    for u_i=1:ut
        for u_j = 1:ut
            infodensity(u_i) = infodensity(u_i) + ...
                Uset(u_j,:)*Uset(u_i,:)'/(norm(Uset(u_j,:))*norm(Uset(u_i,:)));
        end
    end
    infodensity = infodensity/ut;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 2:repeatnum+1
    %%%query selection%%%%%%%%%%%%%%%%%%
    %decide the candidate set
    if subsampling.method == 1
        entropyvec = getentropy(Uset,w,classnum);
        [v,eids]=sort(entropyvec,'descend');
        cand_ids = eids(1:subsampling.num);
        entropyvec = entropyvec(cand_ids);
    elseif subsampling.method == 2
        [idx, cts, sumd, Dist] = kmeans(Uset(:,1:d),subsampling.num,'Replicates',...
            10,'Maxiter',200,'EmptyAction','singleton','Display','off');
        [v, ids] = min(Dist);
        cand_ids = unique(ids);
        % 随机采样一个子集  从393抽取200子集
    elseif subsampling.method == 3
        tmp = randperm(ut);
        cand_ids = tmp(1:subsampling.num);
    else
        cand_ids = 1:ut;
    end
    if indicator>0
        %compute measure
        if subsampling.method ~= 1 % not random
            entropyvec = getentropy(Uset(cand_ids,:),w,classnum);
        end
        
        if length(beta) == 1 %single beta
            if beta~=1
                mutualinfo = getmutualinfo(cand_ids, U, invU);
                measures = (entropyvec.^beta).*(mutualinfo.^(1-beta));
            else
                measures = entropyvec;
            end
            
            [~,id] = max(measures);
            id = cand_ids(id);
            
            %%%%get the label and move to labeled set, and post process
            ut = ut - 1;
            lt = lt + 1;
            Lset(lt,:) = Uset(id,:);
            Uset(id,:) = [];
            ulist(id) = [];
            
            invU = calInvSubmat(U,invU,id);
            U = uK(ulist, ulist);
            %invU = inv(U+1e-8*eye(ut));
            
            %%% evaluate%%%%%%%%%%%%%%%
            w=logregtrain(Lset,lambda);
            [accu,~,~]=logregclassifynew(Tset,w,classnum,classes);
            result.accu(iter)=accu;
        else
            mutualinfo = getmutualinfo(cand_ids, U, invU);
            measures = zeros(length(cand_ids),length(beta));
            for b_i=1:length(beta)
                measures(:,b_i) = (entropyvec.^beta(b_i)).*(mutualinfo.^(1-beta(b_i)));
            end
            [~, indice] = max(measures,[],1);
            indice = unique(indice); % get rid of redundancy
            
            [~,~,prob] = logregclassifynew(Uset(cand_ids(indice),:),w,classnum,classes);
            costs = [];
            for j=1:length(indice)
                expecterr = 0;
                for k=1:classnum
                    %add this unlabeled data into labeled set
                    tempLdata = [Lset;[Uset(cand_ids(indice(j)),1:end-1) classes(k)]];
                    %retrain
                    weights = logregtrain(tempLdata, lambda);
                    %remove this unlabeled data from unlabeled set
                    tempUdata = Uset;
                    tempUdata(cand_ids(indice(j)),:) = [];
                    %predicting on unlabeled set
                    [~,pred_temp,probnew] = logregclassifynew(tempUdata,weights,classnum,classes);
                    %calculate measures
                    index = sub2ind(size(probnew),pred_temp,1:size(tempUdata,1));
                    expecterr = expecterr+prob(k,j)*sum(1-probnew(index));
                end
                costs = [costs;expecterr];
            end
            [~,id] = min(costs); % min error
            id = cand_ids(indice(id));
            ut = ut - 1;
            lt = lt + 1;
            Lset(lt,:) = Uset(id,:);
            Uset(id,:) = [];
            ulist(id) = [];
            
            invU = calInvSubmat(U,invU,id);
            U = uK(ulist, ulist);
            %invU = inv(U+1e-8*eye(ut));
            
            %%% evaluate%%%%%%%%%%%%%%%
            w=logregtrain(Lset,lambda);
            [accu,~,~]=logregclassifynew(Tset,w,classnum,classes);
            result.accu(iter)=accu;
        end
    elseif indicator==0 % density
        entropyvec = getentropy(Uset(cand_ids,:),w,classnum);
        measures = entropyvec.*infodensity(cand_ids);
        
        [~,id] = max(measures);
        id = cand_ids(id);
        
        %%%%get the label and move to labeled set, and post process
        ut = ut - 1;
        lt = lt + 1;
        Lset(lt,:) = Uset(id,:);
        Uset(id,:) = [];
        ulist(id) = [];
        
        %%% evaluate%%%%%%%%%%%%%%%
        w=logregtrain(Lset,lambda);
        [accu,~,~]=logregclassifynew(Tset,w,classnum,classes);
        result.accu(iter)=accu;
    else
        [~,~,prob] = logregclassifynew(Uset(cand_ids,:),w,classnum,classes);
        costs = zeros(length(cand_ids),1);
        matlabpool open;
        parfor j=1:length(cand_ids)
            expecterr = 0;
            for k=1:classnum
                %add this unlabeled data into labeled set
                tempLdata = [Lset;[Uset(cand_ids(j),1:end-1) classes(k)]];
                %retrain
                weights = logregtrain(tempLdata, lambda);
                %remove this unlabeled data from unlabeled set
                tempUdata = Uset;
                tempUdata(cand_ids(j),:) = [];
                %predicting on unlabeled set
                [~,pred_temp,probnew] = logregclassifynew(tempUdata,weights,classnum,classes);
                %calculate measures
                index = sub2ind(size(probnew),pred_temp,1:size(tempUdata,1));
                expecterr = expecterr+prob(k,j)*sum(1-probnew(index));
            end
            costs(j) = expecterr;
        end
        matlabpool close;
        [~,id] = min(costs); % min error
        id = cand_ids(id);
        ut = ut - 1;
        lt = lt + 1;
        Lset(lt,:) = Uset(id,:);
        Uset(id,:) = [];
        ulist(id) = [];
        
        invU = calInvSubmat(U,invU,id);
        U = uK(ulist, ulist);
        %invU = inv(U+1e-8*eye(ut));
        
        %%% evaluate%%%%%%%%%%%%%%%
        w=logregtrain(Lset,lambda);
        [accu,~,~]=logregclassifynew(Tset,w,classnum,classes);
        result.accu(iter)=accu;
    end
end
selections = Lset(end-repeatnum+1:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
