% % Run script of Yelp review
% % Written by Qiong Wang at University of Pennsylvania
% % 11/27/2013
% 
% %% Clear Up
% clc, close all;
% % clear;
% 
% %% Initialize and Load Data
% % Parameters
% bigramFlag = false;                      % Whether to use bigram data
% 
% % Path
% addpath ./libsvm_svdd;
% addpath ./liblinear;
% 
% p          = mfilename('fullpath');
% scriptDir  = fileparts(p);
% outputDir  = fullfile(scriptDir, '/results');
% if ~exist(outputDir, 'dir')
%    mkdir(outputDir);
% end
% outputFile = fullfile(outputDir, 'submit.txt');
% 
% % Load data
% if ~exist('train','var')
%     load ../data/review_dataset.mat;
% end
% 
% % Load bigram data
% if ~exist('train_bigram','var') && bigramFlag
%     if ~exist('train_metadata','var')
%         load ../data/metadata.mat;
%     end
%     if ~exist('../data/review_dataset_bigram.mat','file')
%         [Xt_counts_bi, Xq_counts_bi] = make_sparse_bigram(train_metadata, quiz_metadata);
%     else
%         load ../data/review_dataset_bigram.mat;
%     end
% end
% 
% Xt_counts = train.counts;
% Yt = train.labels;
% Xq_counts = quiz.counts;
% 
% initialize_additional_features;

%% Run algorithm
% Training
% try boosting with (non)cross-validation
Xt = Xt_counts;
models = {};
%weights = ones(length(Yt), 1) / length(Yt);
% rounds = 20, samc = 5 --> .84 version

for rounds = 1:20
    tic
    max_acc = 0;
    mod_acc = [];
    for samc = 1:5,
        sel = randperm(length(Yt), 5000);
        nXt = Xt(sel, :);
        nYt = Yt(sel);
        
        for type = 0:7
            cmod = liblinear_train(nYt, nXt, sprintf('-s %d -c 0.1 -q', type));
            [Yhat, acc, ~] = liblinear_predict(Yt, Xt, cmod);
            acc = acc(1) / 100;
            if (acc > max_acc)
                max_acc = acc;
                mod_acc = cmod;
                disp(acc);
            end;
        end;
    end;
    models{rounds} = mod_acc;
    weight(rounds) = 0.5 * log(max_acc / (1-max_acc));
    fprintf('MaxxAcc = %f, round %d\n', max_acc, rounds);
    toc;
end;

% 
%% Predicting
Yq = zeros(size(Xq_counts,1),1);
for rounds = 1:length(weight),
    cYq = liblinear_predict(Yq_counts, Xq_counts, models{rounds});
    Yq = Yq + cYq * weight(rounds);
end;
Yq = Yq / sum(weight);
rates = Yq;

% tic;
% [Yhat0, ~, descivalues] = liblinear_predict(Yq_counts, Xq_counts, model0);
% Yhat1 = liblinear_predict(Yq_counts, Xq_counts, model1);
% Yhat2 = liblinear_predict(Yq_counts, Xq_counts, model2);
% Yhat3 = liblinear_predict(Yq_counts, Xq_counts, model3);
% Yhat4 = liblinear_predict(Yq_counts, Xq_counts, model4);
% Yhat5 = liblinear_predict(Yq_counts, Xq_counts, model5);
% Yhat6 = liblinear_predict(Yq_counts, Xq_counts, model6);
% Yhat7 = liblinear_predict(Yq_counts, Xq_counts, model7); 
% 
% end;
% % rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features, Xq_additional_features, Yt);
%                    
% % RMSEs = sqrt(sum((Yh - Yt).^2) ./ numel(Yt));
% 
% % rates = Yhat0;
% rates = (Yhat0 + Yhat1 + Yhat2 + Yhat3 + Yhat4 + 2 * Yhat5 + 4 * Yhat6 + 3 * Yhat7) / 14;
% toc

%% Save results to a text file for submission
%dlmwrite(outputFile, rates, 'precision','%d');
dlmwrite('~/submit.txt', rates, 'precision','%d');