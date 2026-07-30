close all; clear; clc

%% ================= Import Data =================
rng(0);
train_tbl = readtable('train.csv');
test_tbl  = readtable('test.csv');

train_ids = train_tbl.stud_id;
test_ids  = test_tbl.stud_id;
train_tbl.stud_id = [];
test_tbl.stud_id  = [];

y_train = categorical(train_tbl.status);
train_tbl.status = [];

%% ================= Feature Engineering =================
train_tbl = addEngineeredFeatures(train_tbl);
test_tbl  = addEngineeredFeatures(test_tbl);

%% ================= Convert Text Columns to Native Categorical =================
% Key difference from exp8/exp9: text columns are converted to MATLAB's
% 'categorical' type directly, NOT one-hot encoded. fitcensemble/templateTree
% detect categorical-typed table columns automatically and split on them
% natively (grouping categories flexibly on either side of a split),
% instead of being forced through ~250 sparse binary dummy columns. This
% avoids diluting the tree's attention across many near-empty one-hot
% columns and lets each split use the full information in a categorical
% column at once.

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

fprintf('Training predictors: %d columns (categorical columns kept native, not one-hot)\n', width(train_tbl_native));

%% ================= Train Random Forest (Native Categorical Splits) =================
% 'PredictorSelection','interaction-curvature' corrects a known bias where
% predictors with many possible categories/split points get favored purely
% because they offer more ways to split, not because they're more
% predictive -- this matters here since some categorical columns (course,
% occupations) have far more levels than others (gender, debtor).

nTrees = 300;
treeTemplate = templateTree('MinLeafSize', 2, 'MaxNumSplits', 200, ...
    'PredictorSelection', 'interaction-curvature', ...
    'Surrogate', 'on');

rfMdl = fitcensemble(train_tbl_native, y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', nTrees, ...
    'Learners', treeTemplate);

%% ================= Cross-Validated Evaluation =================
cvMdl = crossval(rfMdl, 'KFold', 5);
predTrain = kfoldPredict(cvMdl);

figure;
confusionchart(y_train, predTrain);
title('Confusion Matrix - Random Forest, Native Categorical Splits (Cross-Validated)');
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
fprintf('Cross-validated accuracy: %.4f\n', 1 - kfoldLoss(cvMdl));

%% ================= Predict on Test Set & Write Submission =================
predTest = predict(rfMdl, test_tbl_native);

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
