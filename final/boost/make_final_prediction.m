function prediction = make_final_prediction(model,test_words,test_meta)

% Input
% test_words : a 1xp vector representing "1" test sample.
% test_meta : a struct containing the metadata of the test sample.
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

nrounds = 28;
prediction = 0;
for rounds = 1:nrounds,
    cYq = liblinear_predict(0, test_words(:, model.modelfeatures{rounds}), model.models{rounds}, '-q');
    prediction = prediction + cYq * model.modelweight(rounds);
end;
prediction = prediction / sum(model.modelweight(1:nrounds));
