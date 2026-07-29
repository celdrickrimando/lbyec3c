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
% Same engineered features as the k-NN version. Random Forest tends to
% be far more robust to noisy/irrelevant raw columns than k-NN (it can
% simply ignore them via splits), so unlike the k-NN script we do NOT
% need a manual exclusion list here -- all 36 raw columns are kept.
train_tbl = addEngineeredFeatures(train_tbl);
test_tbl  = addEngineeredFeatures(test_tbl);

fprintf('Training predictors: %d columns (no manual exclusion needed for RF)\n', width(train_tbl));

%% ================= Encode Categorical Columns =================
% Text columns are one-hot encoded manually; numeric columns kept as-is.
% Train and test combined first so both share identical dummy columns.

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
        catCol = categorical(col);
        cats = categories(catCol);
        dummies = zeros(numel(catCol), numel(cats));
        for c = 1:numel(cats)
            dummies(:, c) = double(catCol == cats{c});
        end
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

%% ================= Train Random Forest (Bagged Trees) =================
% Random Forest via fitcensemble with the 'Bag' method. Numeric-code
% comparison in local Python testing (300 trees, MaxNumSplits ~ deep,
% MinLeafSize = 2) landed around 0.775 cross-validated accuracy on the
% full column set, vs. ~0.76 for the tuned k-NN version on a hand-pruned
% column subset -- so RF is used here on ALL columns, no exclusion list.

nTrees = 300;
treeTemplate = templateTree('MinLeafSize', 2, 'MaxNumSplits', 200);

rfMdl = fitcensemble(X_train_full, y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', nTrees, ...
    'Learners', treeTemplate);

%% ================= Cross-Validated Evaluation =================
cvMdl = crossval(rfMdl, 'KFold', 5);
predTrain = kfoldPredict(cvMdl);

figure;
confusionchart(y_train, predTrain);
title('Confusion Matrix - Random Forest (Cross-Validated)');
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

%% ================= Feature Importance (informational) =================
imp = predictorImportance(rfMdl);
[~, impOrder] = sort(imp, 'descend');
figure;
bar(imp(impOrder(1:min(20, numel(imp)))));
xticks(1:min(20, numel(imp)));
xticklabels(featureNames(impOrder(1:min(20, numel(imp)))));
xtickangle(90);
title('Top 20 Feature Importances (Random Forest)');
ylabel('Importance');

%% ================= Predict on Test Set & Write Submission =================
predTest = predict(rfMdl, X_test_full);

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