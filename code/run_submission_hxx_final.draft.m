% % Run script of Yelp review
% % Written by Qiong Wang at University of Pennsylvania
% % 11/27/2013
% 
% % Clear Up
% clc, close all;
% clear;
% 
% % Initialize and Load Data
% Parameters
% bigramFlag = false;                      % Whether to use bigram data
% 
% Path
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
Load data
if ~exist('train','var')
    load ../data/review_dataset.mat;
end
% 
% Load bigram data
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
Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;
% 
% initialize_additional_features;

%% Run algorithm
% Training
% try boosting with (non)cross-validation
Xt = Xt_counts;
nrounds = 28;
samsweak = 5000;
feasweak = 30000;
models = cell(nrounds, 1);
modelweight = zeros(nrounds, 1);
modelfeatures = cell(nrounds, 1);
% initalize weight
weight = ones(length(Yt), 1) / length(Yt);
% rounds = 20, samc = 5 --> .84 version

%% loop start
for round = 1:nrounds
    tic;
    max_acc = 0;
    % sampling
    accumweights = cumsum(weight);
    if (round < 30)
        reps = 30;
    else% (round < 50)
        reps = 35;
    end;
    
    for i = 1:reps,
        samram = rand(samsweak, 1);
        [~, sams] = histc(samram, [0; accumweights]);
        sel = false(size(Yt));
        sel(sams) = true;
        
        Xtsam = Xt(sams, :);
        Ytsam = Yt(sams);
        
        Xtval = Xt(~sel, :);
        Ytval = Yt(~sel);

        feas = randperm(size(Xtsam, 2), feasweak);
        for type = [0,1,7]
            if type == 7,
                c = 0.3162;
            else
                c = 0.1;
            end;
            cmod = liblinear_train(Ytsam, Xtsam(:, feas), sprintf('-s %d -c %f -q', type, 10^c));
            [Yhat, acc, ~] = liblinear_predict(Ytval, Xtval(:, feas), cmod);
            acc = acc(1) / 100;
            if (acc > max_acc)
                max_acc = acc;
                mod_acc = cmod;
                hat_acc = Yhat;
                fea_acc = feas;
                dbgid = type;
                opt_c = 10^c;
                disp(dbgid);
            end;
        end;
    end;
    % boosting
    models{round} = mod_acc;
    alpha = 0.5 * log(max_acc / (1-max_acc));
    modelweight(round) = alpha;
    modelfeatures{round} = fea_acc;
    % debug
    dbtype(round) = dbgid;
    dboptc(round) = opt_c;
    % update sample weights
    weight(Yt == hat_acc) = weight(Yt == hat_acc) * exp(-alpha);
    weight(Yt ~= hat_acc) = weight(Yt ~= hat_acc) * exp(alpha);
    weight = weight / sum(weight);
    %
    fprintf('MaxxAcc = %f, round %d\n', max_acc, round);
    toc;
    if (mod(round, 5) == 0)
        save('mod.84_2.mat', 'models', 'modelweight', 'modelfeatures');
    end;
end;

%% validate
load('mod.84_2.mat');
clear round;
Yhat = zeros(size(Xt_counts,1),1);
for rounds = 1:nrounds,
    cYt = liblinear_predict(zeros(size(Xt_counts,1),1), Xt_counts(:, modelfeatures{rounds}), models{rounds});
    Yhat = Yhat + cYt * modelweight(rounds);
    ypre = round(Yhat);
  % acc(rounds) = mean(ypre ~= Yt);
    acc(rounds) = mean((Yhat - Yt).^2);
end;
% 
%% Predicting
Yq = zeros(size(Xq_counts,1),1);
for rounds = 1:nrounds,
    cYq = liblinear_predict(zeros(size(Xq_counts,1),1), Xq_counts(:, modelfeatures{rounds}), models{rounds});
    Yq = Yq + cYq * modelweight(rounds);
end;
Yq = Yq / sum(modelweight(1:nrounds));
rates = Yq;

%% Save results to a text file for submission
%dlmwrite(outputFile, rates, 'precision','%d');
dlmwrite('results/submitM1_28.txt', rates, 'precision','%d');