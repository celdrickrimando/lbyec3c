close all; clear; clc

%% ================= Import Data =================
% exp_5way_svm_mlp: single isolated change from the CONFIRMED 0.80198
% best (RF + LDA + SVM + RUS, 10-fold OOF, nTrees=300, original 14-
% feature set). This experiment ADDS a neural-network classifier
% (fitcnet, MLP) as a 5TH blend member alongside that confirmed base -
% deliberately NOT stacked onto the still-unconfirmed GB branch from
% exp_5way_svm_gb (that branch only has OOF signal, 0.7881 vs the
% 4-way's 0.7848 - not a real Kaggle confirmation, and this session's
% OOF proxy has predicted the wrong real-Kaggle direction on 2 of the
% last 3 changes tested this way). Keeping this isolated to ONE new
% variable (MLP) against the CONFIRMED base follows the same discipline
% as every other experiment in this session.
%
% Why MLP specifically: every model in the confirmed blend (RF, LDA,
% SVM, RUS) is either tree-based, linear, or margin-based. A fully-
% connected neural network has a genuinely different bias/variance
% profile - the same "real mechanism difference, not just another
% hyperparameter tweak" bar that made GB worth testing.
%
% CAVEAT - IMPORTANT, read before running: unlike the other experiments
% in this file, the rationale below has NOT been checked against real
% training data via a proxy pass before writing this script. There was
% no MATLAB environment or your train.csv/test.csv available to run
% that check, so this is an untested hypothesis, not a pre-validated
% one. Run it, look at the disagreement check and grid-search numbers
% below BEFORE trusting this direction, let alone submitting to Kaggle.
%
% Requires Statistics and Machine Learning Toolbox R2021a+ (fitcnet).
rng(42); % unchanged from confirmed-best seed
train_tbl = readtable('train.csv');
test_tbl  = readtable('test.csv');

train_ids = train_tbl.stud_id;
test_ids  = test_tbl.stud_id;
train_tbl.stud_id = [];
test_tbl.stud_id  = [];

y_train = categorical(train_tbl.status);
train_tbl.status = [];

classNames = categories(y_train);
nClasses   = numel(classNames);

%% ================= Feature Engineering =================
train_tbl = addEngineeredFeatures(train_tbl);
test_tbl  = addEngineeredFeatures(test_tbl);

%% ================= Representation 1: native categorical (for RF/RUS) =================
varNames = train_tbl.Properties.VariableNames;
combined_tbl = [train_tbl; test_tbl];
n_train = height(train_tbl);

for i = 1:numel(varNames)
    if ~isnumeric(combined_tbl.(varNames{i}))
        combined_tbl.(varNames{i}) = categorical(combined_tbl.(varNames{i}));
    end
end

train_tbl_native = combined_tbl(1:n_train, :);
test_tbl_native  = combined_tbl(n_train+1:end, :);

%% ================= Representation 2: one-hot + standardized numeric (for LDA/SVM/MLP) =================
catCols = varNames(~varfun(@isnumeric, combined_tbl, 'OutputFormat', 'uniform'));
numCols = varNames(varfun(@isnumeric, combined_tbl, 'OutputFormat', 'uniform'));

oneHotBlocks = cell(1, numel(catCols));
for i = 1:numel(catCols)
    oneHotBlocks{i} = dummyvar(combined_tbl.(catCols{i}));
end
X_cat_all = [oneHotBlocks{:}];

X_num_all = table2array(combined_tbl(:, numCols));
mu = mean(X_num_all(1:n_train, :), 1);
sigma = std(X_num_all(1:n_train, :), [], 1);
sigma(sigma == 0) = 1;
X_num_all = (X_num_all - mu) ./ sigma;

X_flat_all = [X_cat_all, X_num_all];
X_flat_train = X_flat_all(1:n_train, :);
X_flat_test  = X_flat_all(n_train+1:end, :);

fprintf('Native-categorical table: %d columns. One-hot/standardized matrix: %d columns.\n', ...
    width(train_tbl_native), size(X_flat_train, 2));

%% ================= Model templates =================
treeTemplate = templateTree('MinLeafSize', 2, 'MaxNumSplits', 200, ...
    'PredictorSelection', 'interaction-curvature', ...
    'Surrogate', 'on');
nTrees = 300;

svmTemplate = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 0.3, 'Standardize', false);

boostTreeTemplate = templateTree('MaxNumSplits', 20, ...
    'PredictorSelection', 'interaction-curvature', 'Surrogate', 'on');
nBoostCycles = 200;

% exp_5way_svm_mlp: MLP hyperparameters - a reasonable first pass, NOT
% tuned. Two modest hidden layers plus L2 regularization (Lambda) to
% keep it from just memorizing the training fold given the one-hot
% column count printed above. 'Standardize', false because the numeric
% block is already manually standardized above (same reasoning as the
% SVM template's identical choice). If the disagreement check below
% looks promising, this is the first thing worth tuning before a real
% Kaggle submission.
mlpLayerSizes = [40 20];
mlpLambda = 1e-3;

%% ================= 10-fold OOF predictions to tune blend weights =================
cv = cvpartition(y_train, 'KFold', 10);
oofRF  = zeros(n_train, nClasses);
oofLDA = zeros(n_train, nClasses);
oofSVM = zeros(n_train, nClasses);
oofRUS = zeros(n_train, nClasses);
oofMLP = zeros(n_train, nClasses);

for k = 1:cv.NumTestSets
    trIdx = training(cv, k);
    teIdx = test(cv, k);

    % --- RF on native-categorical fold ---
    rfFold = fitcensemble(train_tbl_native(trIdx, :), y_train(trIdx), ...
        'Method', 'Bag', 'NumLearningCycles', nTrees, 'Learners', treeTemplate);
    [~, scoreRF] = predict(rfFold, train_tbl_native(teIdx, :));
    oofRF(teIdx, :) = alignScoreColumns(scoreRF, rfFold.ClassNames, classNames);

    % --- LDA on one-hot fold ---
    ldaFold = fitcdiscr(X_flat_train(trIdx, :), y_train(trIdx), 'DiscrimType', 'pseudoLinear');
    [~, scoreLDA] = predict(ldaFold, X_flat_train(teIdx, :));
    oofLDA(teIdx, :) = alignScoreColumns(scoreLDA, ldaFold.ClassNames, classNames);

    % --- ECOC-SVM on one-hot fold ---
    svmFold = fitcecoc(X_flat_train(trIdx, :), y_train(trIdx), ...
        'Learners', svmTemplate, 'Coding', 'onevsone', 'FitPosterior', true);
    [~, ~, ~, postSVM] = predict(svmFold, X_flat_train(teIdx, :));
    oofSVM(teIdx, :) = alignScoreColumns(postSVM, svmFold.ClassNames, classNames);

    % --- RUSBoost on native-categorical fold ---
    rusFold = fitcensemble(train_tbl_native(trIdx, :), y_train(trIdx), ...
        'Method', 'RUSBoost', 'NumLearningCycles', nBoostCycles, 'Learners', boostTreeTemplate);
    [~, scoreRUS] = predict(rusFold, train_tbl_native(teIdx, :));
    oofRUS(teIdx, :) = alignScoreColumns(scoreRUS, rusFold.ClassNames, classNames);

    % --- MLP on one-hot fold - NEW 5th member, isolated test ---
    % fitcnet returns calibrated posterior probabilities directly from
    % predict() - no softmax workaround needed here, unlike the GB-ECOC
    % branch in exp_5way_svm_gb (FitPosterior there is SVM-only).
    mlpFold = fitcnet(X_flat_train(trIdx, :), y_train(trIdx), ...
        'LayerSizes', mlpLayerSizes, 'Lambda', mlpLambda, ...
        'Standardize', false);
    [~, scoreMLP] = predict(mlpFold, X_flat_train(teIdx, :));
    oofMLP(teIdx, :) = alignScoreColumns(scoreMLP, mlpFold.ClassNames, classNames);

    fprintf('Fold %d done.\n', k);
end

%% ================= Isolated MLP-vs-SVM disagreement check =================
% Same style of check that was run before folding GB into the full grid
% search in exp_5way_svm_gb (288/3318 disagreeing rows, split 162/126) -
% a cheap sanity check on whether MLP is contributing something SVM
% alone doesn't, before trusting the grid search below. If this split is
% heavily lopsided one direction, MLP is probably just a weaker SVM, not
% a complementary one, and the grid search is likely to hand it ~0
% weight the way RUSBoost got ~0 weight in the GB experiment.
predSVM = categorical(classNames(argmaxCols(oofSVM)));
predMLP = categorical(classNames(argmaxCols(oofMLP)));
disagreeIdx = predSVM ~= predMLP;
svmRightMlpWrong = sum(disagreeIdx & predSVM == y_train & predMLP ~= y_train);
mlpRightSvmWrong = sum(disagreeIdx & predMLP == y_train & predSVM ~= y_train);
fprintf('\nSVM vs MLP disagree on %d of %d rows.\n', sum(disagreeIdx), n_train);
fprintf('  SVM right / MLP wrong: %d\n', svmRightMlpWrong);
fprintf('  MLP right / SVM wrong: %d\n', mlpRightSvmWrong);

%% ================= Grid search blend weights on OOF predictions =================
bestAcc = 0;
bestW = [1 0 0 0 0];
step = 0.05;
for w1 = 0:step:1
    for w2 = 0:step:(1 - w1)
        for w3 = 0:step:(1 - w1 - w2)
            for w4 = 0:step:(1 - w1 - w2 - w3)
                w5 = 1 - w1 - w2 - w3 - w4;
                blend = w1 * oofRF + w2 * oofLDA + w3 * oofSVM + w4 * oofRUS + w5 * oofMLP;
                [~, predIdx] = max(blend, [], 2);
                predLabels = categorical(classNames(predIdx));
                acc = mean(predLabels == y_train);
                if acc > bestAcc
                    bestAcc = acc;
                    bestW = [w1 w2 w3 w4 w5];
                end
            end
        end
    end
end
fprintf('\nBest OOF blend weights [RF LDA SVM RUS MLP] = [%.2f %.2f %.2f %.2f %.2f], OOF accuracy = %.4f\n', ...
    bestW(1), bestW(2), bestW(3), bestW(4), bestW(5), bestAcc);
fprintf('(Reference) RF alone OOF accuracy:  %.4f\n', mean(categorical(classNames(argmaxCols(oofRF)))  == y_train));
fprintf('(Reference) LDA alone OOF accuracy: %.4f\n', mean(categorical(classNames(argmaxCols(oofLDA))) == y_train));
fprintf('(Reference) SVM alone OOF accuracy: %.4f\n', mean(categorical(classNames(argmaxCols(oofSVM))) == y_train));
fprintf('(Reference) RUSBoost alone OOF accuracy: %.4f\n', mean(categorical(classNames(argmaxCols(oofRUS))) == y_train));
fprintf('(Reference) MLP alone OOF accuracy: %.4f\n', mean(categorical(classNames(argmaxCols(oofMLP))) == y_train));

blendOOF = bestW(1) * oofRF + bestW(2) * oofLDA + bestW(3) * oofSVM + bestW(4) * oofRUS + bestW(5) * oofMLP;
[~, blendIdx] = max(blendOOF, [], 2);
predTrainBlend = categorical(classNames(blendIdx));

figure;
confusionchart(y_train, predTrainBlend);
title('Confusion Matrix - RF+LDA+SVM+RUS+MLP Blend (Out-of-Fold)');
xlabel('Predicted Class');
ylabel('True Class');

C = confusionmat(y_train, predTrainBlend);
scoreF1 = zeros(nClasses, 1);
for i = 1:nClasses
    TP = C(i,i);
    FP = sum(C(:,i)) - TP;
    FN = sum(C(i,:)) - TP;
    prec = TP / (TP + FP);
    rec  = TP / (TP + FN);
    scoreF1(i) = 2 * (prec * rec) / (prec + rec);
end
fprintf('\nPer-class F1 scores (blend):\n');
for i = 1:nClasses
    fprintf('  %s: %.4f\n', classNames{i}, scoreF1(i));
end

%% ================= Train final models on FULL training data =================
rfFinal = fitcensemble(train_tbl_native, y_train, ...
    'Method', 'Bag', 'NumLearningCycles', nTrees, 'Learners', treeTemplate);
ldaFinal = fitcdiscr(X_flat_train, y_train, 'DiscrimType', 'pseudoLinear');
svmFinal = fitcecoc(X_flat_train, y_train, ...
    'Learners', svmTemplate, 'Coding', 'onevsone', 'FitPosterior', true);
rusFinal = fitcensemble(train_tbl_native, y_train, ...
    'Method', 'RUSBoost', 'NumLearningCycles', nBoostCycles, 'Learners', boostTreeTemplate);
mlpFinal = fitcnet(X_flat_train, y_train, ...
    'LayerSizes', mlpLayerSizes, 'Lambda', mlpLambda, 'Standardize', false);

[~, scoreRFTest]  = predict(rfFinal, test_tbl_native);
[~, scoreLDATest] = predict(ldaFinal, X_flat_test);
[~, ~, ~, postSVMTest] = predict(svmFinal, X_flat_test);
[~, scoreRUSTest] = predict(rusFinal, test_tbl_native);
[~, scoreMLPTest] = predict(mlpFinal, X_flat_test);

scoreRFTest  = alignScoreColumns(scoreRFTest,  rfFinal.ClassNames,  classNames);
scoreLDATest = alignScoreColumns(scoreLDATest, ldaFinal.ClassNames, classNames);
postSVMTest  = alignScoreColumns(postSVMTest,  svmFinal.ClassNames, classNames);
scoreRUSTest = alignScoreColumns(scoreRUSTest, rusFinal.ClassNames, classNames);
scoreMLPTest = alignScoreColumns(scoreMLPTest, mlpFinal.ClassNames, classNames);

blendTest = bestW(1) * scoreRFTest + bestW(2) * scoreLDATest + bestW(3) * postSVMTest ...
    + bestW(4) * scoreRUSTest + bestW(5) * scoreMLPTest;
[~, testIdx] = max(blendTest, [], 2);
predTest = categorical(classNames(testIdx));

submission = table(test_ids, cellstr(predTest), 'VariableNames', {'stud_id', 'status'});
writetable(submission, 'submission.csv');
fprintf('\nSubmission written to submission.csv\n');

%% ================= Local Functions =================
function tbl = addEngineeredFeatures(tbl)
%ADDENGINEEREDFEATURES Adds derived academic-performance features.
    c1_enr  = tbl.curricular_units_1st_sem__enrolled_;
    c2_enr  = tbl.curricular_units_2nd_sem__enrolled_;
    c1_app  = tbl.curricular_units_1st_sem__approved_;
    c2_app  = tbl.curricular_units_2nd_sem__approved_;
    c1_eval = tbl.curricular_units_1st_sem__evaluations_;
    c2_eval = tbl.curricular_units_2nd_sem__evaluations_;
    c1_grade = tbl.curricular_units_1st_sem__grade_;
    c2_grade = tbl.curricular_units_2nd_sem__grade_;
    c1_cred = tbl.curricular_units_1st_sem__credited_;
    c2_cred = tbl.curricular_units_2nd_sem__credited_;

    tbl.total_enrolled = c1_enr + c2_enr;
    tbl.total_approved = c1_app + c2_app;
    tbl.total_eval     = c1_eval + c2_eval;
    tbl.total_credited = c1_cred + c2_cred;

    approval_rate_1st = c1_app ./ c1_enr;
    approval_rate_1st(c1_enr == 0) = 0;
    tbl.approval_rate_1st = approval_rate_1st;

    approval_rate_2nd = c2_app ./ c2_enr;
    approval_rate_2nd(c2_enr == 0) = 0;
    tbl.approval_rate_2nd = approval_rate_2nd;

    approval_rate_total = tbl.total_approved ./ tbl.total_enrolled;
    approval_rate_total(tbl.total_enrolled == 0) = 0;
    tbl.approval_rate_total = approval_rate_total;

    tbl.avg_grade      = (c1_grade + c2_grade) / 2;
    tbl.grade_diff     = c2_grade - c1_grade;
    tbl.enrolled_diff  = c2_enr - c1_enr;
    tbl.approved_diff  = c2_app - c1_app;

    tbl.zero_approved_1st  = double(c1_app == 0);
    tbl.zero_approved_2nd  = double(c2_app == 0);
    tbl.zero_enrolled_2nd  = double(c2_enr == 0);
end

function aligned = alignScoreColumns(scoreMat, modelClassNames, canonicalClassNames)
%ALIGNSCORECOLUMNS Reorders a model's posterior/score columns so column i
% always corresponds to canonicalClassNames{i}, regardless of the order
% the model itself assigned to its classes.
    nCanon = numel(canonicalClassNames);
    aligned = zeros(size(scoreMat, 1), nCanon);
    modelClassNames = cellstr(modelClassNames);
    for i = 1:nCanon
        colIdx = find(strcmp(modelClassNames, canonicalClassNames{i}));
        aligned(:, i) = scoreMat(:, colIdx);
    end
end

function idx = argmaxCols(scoreMat)
    [~, idx] = max(scoreMat, [], 2);
end
