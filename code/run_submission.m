clear;
load ../data/review_dataset.mat

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

initialize_additional_features;
%% Run algorithm
rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                       Xq_additional_features, Yt);

%% Save results to a text file for submission
dlmwrite('submit.txt',rates,'precision','%d');