function model = init_model(vocab)

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
% 
training = false;

if training,
    %% load data
    tmp = load('../data/review_dataset.mat');
    Xt = tmp.train.counts;
    Yt = tmp.train.labels;
    Xq = tmp.quiz.counts;
    clear tmp
    %% train nb model
    nb = NaiveBayes.fit(Xt, Yt, 'distribution', 'mn');
    save('model.mat', 'nb');
    %% predict on quiz
    Yq = nb.predict(Xq);
    dlmwrite('submitNB.txt', Yq, 'precision','%d');
else
    tmp = load('model.mat');
    nb = tmp.nb;
end;

model.nb = nb;

end