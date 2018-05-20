close all;
clear;

load("YaleB_32x32.mat");
TrainData=[];
TrainLabel=[];
TestData=[];
TestLabel=[];
GNDlength = length(unique(gnd));
p=10;

breakValue=61;
f=5;
lambda=0.1;

for j=1:GNDlength
   X=fea(gnd==j,:);
   l=size(X,1);
   ind1=randperm(l);
   TrainData = [TrainData ; X(ind1(1:p),:)];
   TestData = [TestData ; X(ind1(p+1:end),:)];
   TrainLabel=[TrainLabel ;j*ones(p,1)];
   TestLabel=[TestLabel ;j*ones(l-p,1)];  
end

testDataSize = size(TestData,1);
trainDataSize = size(TrainData,1);
trainLabelLength=length(unique(TrainLabel));

mu=mean(TrainData);
PHI = TrainData-repmat(mu,trainDataSize,1);
STMatrix=PHI'*PHI;
[Val,~] = eig(STMatrix);
Wdata=Val(:,end-breakValue+1-f:end-f);
Wdata=normc(Wdata);
TrainData = (TrainData - repmat(mu,trainDataSize,1))*Wdata;
TestData = (TestData-repmat(mu,testDataSize,1))*Wdata;

for i=1:testDataSize
    YT = TestData(i,:);
    B=lasso(TrainData',YT,'Lambda',lambda,'RelTol',1e-2);
    
    for j=1:trainLabelLength
       ind=TrainLabel==j;
       Y=TrainData(ind,:);
       XT=B(ind);
       recVal=XT'*Y;
       Ndata(j)=norm(YT-recVal);
    end
    [~,index] = min(Ndata);
    output(i) = index;
end
acc = sum((output)==TestLabel')/testDataSize;
disp(acc);