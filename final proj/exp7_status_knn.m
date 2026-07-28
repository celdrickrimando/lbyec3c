close all; clear; clc

%% ================= Import Data =================
rng(0);
train_tbl = readtable('train.csv');
test_tbl  = readtable('test.csv');

% Keep IDs for the submission file, then drop them as a feature
train_ids = train_tbl.stud_id;
test_ids  = test_tbl.stud_id;
train_tbl.stud_id = [];
test_tbl.stud_id  = [];

% Separate out the target label, then remove it from the predictor table
y_train = categorical(train_tbl.status);
train_tbl.status = [];

fprintf('Training predictors before selection: %d columns\n', width(train_tbl));

%% ================= Encode Categorical Columns =================
% Any text column gets one-hot encoded; numeric columns are kept as-is.
% Train and test are combined first so both share the exact same
% encoding (same dummy columns), then split back apart afterward.

varNames = train_tbl.Properties.VariableNames;
n_train  = height(train_tbl);

combined_tbl = [train_tbl; test_tbl];

X_parts = {};
featureNames = {};

for i = 1:numel(varNames)
    col = combined_tbl.(varNames{i});
    if isnumeric(col)
        X_parts{end+1} = col;
        featureNames{end+1} = varNames{i};
    else
        % text / categorical predictor -> one-hot encode
        catCol = categorical(col);
        dummies = double(dummyvar(catCol));
        cats = categories(catCol);
        % drop the last dummy column to avoid perfect collinearity
        dummies(:, end) = [];
        cats(end) = [];
        X_parts{end+1} = dummies;
        for c = 1:numel(cats)
            featureNames{end+1} = sprintf('%s_%s', varNames{i}, cats{c});
        end
    end
end

X_all = cell2mat(X_parts);
X_train_full = X_all(1:n_train, :);
X_test_full  = X_all(n_train+1:end, :);

fprintf('Expanded feature matrix: %d columns (after one-hot encoding)\n', size(X_all,2));

%% ================= Automatic Feature Importance (weed out columns) =================
% Use Minimum-Redundancy-Maximum-Relevance (fscmrmr) to automatically
% rank every expanded feature by how useful it is for predicting status.
% This lets the script decide which columns are "unnecessary" instead
% of hand-picking them.

[rankedIdx, scores] = fscmrmr(X_train_full, y_train);

figure;
bar(scores(rankedIdx));
xlabel('Feature rank');
ylabel('MRMR Importance Score');
title('Feature Importance (fscmrmr)');
xticks(1:numel(rankedIdx));
xticklabels(featureNames(rankedIdx));
xtickangle(90);

% Keep only features whose importance score is above a small positive
% threshold -- this is the "weeding out" step the assignment asks for.
threshold = 0.02;
keepMask = scores > threshold;
selectedIdx = find(keepMask);

% Fallback: if the threshold is too strict, keep the top 15 features
if numel(selectedIdx) < 5
    selectedIdx = rankedIdx(1:15);
end

fprintf('\nSelected %d / %d expanded features:\n', numel(selectedIdx), numel(featureNames));
disp(featureNames(selectedIdx)');

X_train_sel = X_train_full(:, selectedIdx);
X_test_sel  = X_test_full(:, selectedIdx);

%% ================= Choose k via Cross-Validation =================
kRange = 1:2:25;
cvAcc = zeros(numel(kRange), 1);

for i = 1:numel(kRange)
    k = kRange(i);
    Mdl = fitcknn(X_train_sel, y_train, 'NumNeighbors', k, 'Standardize', true);
    cvMdl = crossval(Mdl, 'KFold', 5);
    cvAcc(i) = 1 - kfoldLoss(cvMdl);
end

figure;
plot(kRange, cvAcc, '-o');
xlabel('Number of Neighbors (k)');
ylabel('5-fold CV Accuracy');
title('k-NN Hyperparameter Search');
grid on;

[bestAcc, bestIdx] = max(cvAcc);
bestK = kRange(bestIdx);
fprintf('\nBest k = %d, CV accuracy = %.4f\n', bestK, bestAcc);

%% ================= Train Final Model & Evaluate =================
finalMdl = fitcknn(X_train_sel, y_train, 'NumNeighbors', bestK, 'Standardize', true);

cvFinal = crossval(finalMdl, 'KFold', 5);
predTrain = kfoldPredict(cvFinal);

figure;
confusionchart(y_train, predTrain);
title('Confusion Matrix - k-NN (Cross-Validated)');
xlabel('Predicted Class');
ylabel('True Class');

C = confusionmat(y_train, predTrain);
classes = categories(y_train);
scoreF1 = zeros(numel(classes), 1);
for i = 1:numel(classes)
    TP = C(i,i);
    FP = sum(C(:,i)) - TP;
    FN = sum(C(i,:)) - TP;
    prec = TP / (TP + FP);
    rec  = TP / (TP + FN);
    scoreF1(i) = 2 * (prec * rec) / (prec + rec);
end
fprintf('\nPer-class F1 scores:\n');
for i = 1:numel(classes)
    fprintf('  %s: %.4f\n', classes{i}, scoreF1(i));
end
fprintf('Cross-validated accuracy: %.4f\n', 1 - kfoldLoss(cvFinal));

%% ================= Predict on Test Set & Write Submission =================
predTest = predict(finalMdl, X_test_sel);

submission = table(test_ids, cellstr(predTest), 'VariableNames', {'stud_id', 'status'});
writetable(submission, 'submission.csv');
fprintf('\nSubmission written to submission.csv\n');
