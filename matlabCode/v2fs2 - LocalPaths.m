trainfileID = fopen('normalized training data.txt','r');
AllDataSet = fscanf(trainfileID,'%d',[16 480]);
targetfileID = fopen('targets.txt','r'); 

AllTargets = fscanf(targetfileID,'%d',[1 480]);
All = cat(1,AllDataSet,AllTargets);
% AllDataSet = AllDataSet';
% AllTargets = AllTargets';
%random_All = All(randperm(size(All, 1)), :);
cols = size(All,2);
P = randperm(cols);
Allshuffled = All(:,P);

random_AllDataSet = Allshuffled(1:16,:);
random_AllTargets = Allshuffled(17:17,:);
FSrandom_AllDataSet = random_AllDataSet(9:12,:);
[targettrain,targettest] = divideblock(random_AllTargets,.9,.1);
targettrain = targettrain.'; 
targettest = targettest.';
[trainA,testA] = divideblock(random_AllDataSet, .9, .1); % 90% for training 10% for testing.
trainData = trainA.'; 
testData = testA.';
%P = trainData; 

% P = [0   1   0   1;
%        0    0   1   1];
% T = [0   1   1   0];
net = feedforwardnet(2, 'trainscg');
net.performFcn = 'mse';   
net.trainParam.epochs = 100;
net.divideFcn = ''; 
[net, tr] = train(net, trainData',targettrain' );

y = sim(net, testData');
y = round(y);
cnttrue=0;
cnt1diff=0;
cnt2diff=0;
cnt3diff=0;
cnt4diff=0;
for (i=1:41)
    
        %if abs(targettest(i)-y(i))<0.5
        if y(i)==targettest(i)
           cnttrue = cnttrue+1;
        else if abs(y(i)-targettest(i))==1
                cnt1diff = cnt1diff+1;
            
        else if abs(y(i)-targettest(i))==2
                cnt2diff = cnt2diff+1;
        else if abs(y(i)-targettest(i))==3
                cnt3diff = cnt3diff+1;
        else if abs(y(i)-targettest(i))==4
                cnt4diff = cnt4diff+1;
            
            end
            end
            end
            end
        end
end
res = [cnttrue;cnt1diff;cnt2diff;cnt3diff;cnt4diff]
           