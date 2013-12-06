function model = init_model(vocab)

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
% 
training = false;
addpath('../libs/liblinear/');

if training,
    %% load data
    tmp = load('../../data/review_dataset.mat');
    Xt = tmp.train.counts;
    Yt = tmp.train.labels;
    Xq = tmp.quiz.counts;
    clear tmp
    %% train model
    % Training
    nrounds = 28;
    samsweak = 5000;
    feasweak = 30000;
    models = cell(nrounds, 1);
    modelweight = zeros(nrounds, 1);
    modelfeatures = cell(nrounds, 1);
    % initalize weight
    weight = ones(length(Yt), 1) / length(Yt);
    %% loop start
    for round = 1:nrounds
        tic;
        max_acc = 0;
        % sampling
        accumweights = cumsum(weight);
        for i = 1:30,
            samram = rand(samsweak, 1);
            [~, sams] = histc(samram, [0; accumweights]);

            Xtsam = Xt(sams, :);
            Ytsam = Yt(sams);

            feas = randperm(size(Xtsam, 2), feasweak);
            for type = [0,1,7]
                if type == 7,
                    c = 0.3162;
                else
                    c = 0.1;
                end;
                cmod = liblinear_train(Ytsam, Xtsam(:, feas), sprintf('-s %d -c %f -q', type, 10^c));
                [Yhat, acc, ~] = liblinear_predict(Yt, Xt(:, feas), cmod);
                acc = acc(1) / 100;
                if (acc > max_acc)
                    max_acc = acc;
                    mod_acc = cmod;
                    hat_acc = Yhat;
                    fea_acc = feas;
                end;
            end;
        end;
        % boosting
        models{round} = mod_acc;
        alpha = 0.5 * log(max_acc / (1-max_acc));
        modelweight(round) = alpha;
        modelfeatures{round} = fea_acc;
        % update sample weights
        weight(Yt == hat_acc) = weight(Yt == hat_acc) * exp(-alpha);
        weight(Yt ~= hat_acc) = weight(Yt ~= hat_acc) * exp(alpha);
        weight = weight / sum(weight);
        fprintf('MaxxAcc = %f, round %d\n', max_acc, round);
        toc;
    end;
    save('model.mat', 'models', 'modelweight', 'modelfeatures');
    %% predict on quiz
    Yq = zeros(size(Xq,1),1);
    for rounds = 1:nrounds,
        cYq = liblinear_predict(zeros(size(Xq,1),1), Xq(:, modelfeatures{rounds}), models{rounds});
        Yq = Yq + cYq * modelweight(rounds);
    end;
    Yq = Yq / sum(modelweight(1:nrounds));
    dlmwrite('submitBoost.txt', Yq, 'precision','%d');
else
    tmp = load('model.mat');
    models = tmp.models;
    modelweight = tmp.modelweight;
    modelfeatures = tmp.modelfeatures;
end;

model.models = models;
model.modelweight = modelweight;
model.modelfeatures = modelfeatures;

end