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
    %% train
    %% pca
     V0 = zeros(size(Xt, 2), 1);
%      for dim = 1:size(Xt, 2)
%          nonzeros = Xt(:, dim) ~= 0;
%          dimMean = mean(Xt(nonzeros, dim));
%          V0(dim) = full(dimMean);
%          Xt(nonzeros, dim) = Xt(nonzeros, dim) - dimMean;
%      end;
     tic;
     [~, ~, Va] = svds(Xt, 200);
     toc;
     Xpca = Xt * Va;
    %% knn
     disp('KNN...');
     %[idx, centers] = vl_kmeans(Xtrain2', 1000, 'algorithm', 'ann');
     knnTree = KDTreeSearcher(Xpca);
     save('model.mat', 'V0', 'Va', 'knnTree', 'Yt');
    %% predict on quiz
    %% pca on test
%      for dim = 1:size(Xq, 2)
%          nonzeros = Xt(:, dim) ~= 0;
%          Xq(nonzeros, dim) = Xq(nonzeros, dim) - V0(dim);
%      end;
     Qpca = Xq * Va;
     tic;
     [IDX, Dst] = knnsearch(knnTree, Qpca, 'K', 5);
     toc;
     Dst = 1 ./ Dst;
     Yq = zeros(size(Qpca,1), 1);
     for i = 1:size(Qpca, 1)
         Yq(i) = sum(Yt(IDX(i,:))' .* Dst(i,:)) / sum(Dst(i,:));
     end;
     dlmwrite('submitPCAKNN.txt', Yq, 'precision','%d');
else
    tmp = load('model.mat');
    Va = tmp.Va;
    knnTree = tmp.knnTree;
    Yt = tmp.Yt;
end;

model.Va = Va;
model.knnTree = knnTree;
model.Yt = Yt;

end