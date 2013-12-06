% This is the script to process additional features that you want to use

% Initialize
trainNum    = numel(train_metadata);
trainBuzId  = cell(trainNum,1);
quizNum     = numel(quiz_metadata);
quizBuzId   = cell(quizNum,1);
bestNum     = 500;

% Load data
if ~exist('train_metadata','var')
    load ../data/metadata.mat
end

for i = 1 : trainNum
   trainBuzId(i) = train_metadata(i).business_id;
end

for i = 1 : quizNum
   quizBuzId(i) = quiz_metadata(i).business_id;
end

% Find the most frequent $bestNum business id
[trainBuzIdBest, ~, trainBest] = unique(trainBuzId);
[quizBuzIdBest, ~, quizBest]   = unique(quizBuzId);

trainBuzFreq  = hist(trainBest, unique(trainBest)); 
quizBuzFreq   = hist(quizBest, unique(quizBest));

[~, trainIdx] = sort(trainBuzFreq, 'descend');
[~, quizIdx]  = sort(quizBuzFreq, 'descend');

% Record data
Xt_additional_features = struct;
Xq_additional_features = struct;

Xt_additional_features.buzId = trainBuzIdBest(trainIdx(1:bestNum));
Xt_additional_features.index = trainIdx(1:bestNum);
Xq_additional_features.buzId = quizBuzIdBest(quizIdx(1:bestNum));
Xq_additional_features.index = quizIdx(1:bestNum);