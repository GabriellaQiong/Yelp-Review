function model = init_model(vocab)

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
% 
training = false;
addpath('../libs/libsvm_svdd');

if training
    %% load data
    tmp = load('../../data/review_dataset.mat');
    Xt = tmp.train.counts;
    Yt = tmp.train.labels;
    Xq = tmp.quiz.counts;
    clear tmp
    
    %% train model
    model = svmtrain(Yt, Xt, '-t 3 -c 0.1 -h 0');
    save('model.mat', 'model');
    
    %% predict on quiz
    Yq = zeros(size(Xq,1),1);
    Yq = svmpredict(Yq, Xq, model); 
    dlmwrite('submitSVM.txt', Yq, 'precision','%d');
    
else
    tmp   = load('model.mat');
    model = tmp.model;
end;

end