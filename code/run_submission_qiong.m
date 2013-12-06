% Run script of Yelp review
% Written by Qiong Wang at University of Pennsylvania
% 11/27/2013

%% Clear Up
clc, close all;
% clear;

%% Initialize and Load Data
% Parameters
bigramFlag = false;                      % Whether to use bigram data

% Path
addpath ./libsvm_svdd;
addpath ./liblinear;

p          = mfilename('fullpath');
scriptDir  = fileparts(p);
outputDir  = fullfile(scriptDir, '/results');
if ~exist(outputDir, 'dir')
   mkdir(outputDir);
end
outputFile = fullfile(outputDir, 'submit.txt');

% Load data
if ~exist('train','var')
    load ../data/review_dataset.mat;
end

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

initialize_additional_features;

%% Run algorithm
% Training
tic
model0 = liblinear_train(Yt, Xt_counts, '-s 0 -c 0.1');
model1 = liblinear_train(Yt, Xt_counts, '-s 1 -c 0.1');
model2 = liblinear_train(Yt, Xt_counts, '-s 2 -c 0.1');
model3 = liblinear_train(Yt, Xt_counts, '-s 3 -c 0.1');
model4 = liblinear_train(Yt, Xt_counts, '-s 4 -c 0.1');
model5 = liblinear_train(Yt, Xt_counts, '-s 5 -c 0.1');
model6 = liblinear_train(Yt, Xt_counts, '-s 6 -c 0.1');
model7 = liblinear_train(Yt, Xt_counts, '-s 7 -c 0.1');

% Predicting
Yq_counts = zeros(size(Xq_counts,1),1);

Yhat0 = liblinear_predict(Yq_counts, Xq_counts, model0);
Yhat1 = liblinear_predict(Yq_counts, Xq_counts, model1);
Yhat2 = liblinear_predict(Yq_counts, Xq_counts, model2);
Yhat3 = liblinear_predict(Yq_counts, Xq_counts, model3);
Yhat4 = liblinear_predict(Yq_counts, Xq_counts, model4);
Yhat5 = liblinear_predict(Yq_counts, Xq_counts, model5);
Yhat6 = liblinear_predict(Yq_counts, Xq_counts, model6);
Yhat7 = liblinear_predict(Yq_counts, Xq_counts, model7); 

% rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features, Xq_additional_features, Yt);
                   
% RMSEs = sqrt(sum((Yh - Yt).^2) ./ numel(Yt));

% rates = Yhat0;
rates = (Yhat0 + Yhat1 + Yhat2 + Yhat3 + Yhat4 + 2 * Yhat5 + 4 * Yhat6 + 3 * Yhat7) / 14;
toc

%% Save results to a text file for submission
dlmwrite(outputFile, rates, 'precision','%d');