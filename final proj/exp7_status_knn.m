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

%% ================= Feature Engineering =================
% Add a handful of engineered numeric features summarizing academic
% performance across both semesters. This gives fscmrmr strong
% quantitative signal to compete against the many one-hot category
% columns, which otherwise dominate the top-ranked feature list.
train_tbl = addEngineeredFeatures(train_tbl);
test_tbl  = addEngineeredFeatures(test_tbl);

fprintf('Training predictors before selection: %d columns\n', width(train_tbl));

%% ================= Manual Column Exclusion (optional) =================
% List any raw column names here that you want to force out of the model,
% regardless of what fscmrmr would score them. This runs BEFORE the
% automatic importance ranking, so excluded columns never even get
% one-hot encoded or considered. Leave the cell array empty ({}) to skip
% this step and let fscmrmr decide everything on its own.
%
% Test 2: CV-search validated exclusion set (15 columns), found via a
% local greedy backward-elimination search using 5-fold CV accuracy as a
% proxy for leaderboard score (see feature_search_log.md for full history)
%  2  application_mode
%  3  application_order
%  4  course
%  5  daytime_evening_attendance
%  6  previous_qualification
%  7  previous_qualification_(grade)
%  9  mothers_qualification
%  10 fathers_qualification
%  11 mothers_occupation
%  12 fathers_occupation
%  15 educational_special_needs
%  20 age_at_enrollment
%  21 international
%  27 curricular_units_1st_sem_(without_evaluations)
%  30 curricular_units_2nd_sem_(evaluations)
manualExclude = {'application_mode', 'application_order', 'course', ...
                  'daytime_evening_attendance', 'previous_qualification', ...
                  'previous_qualification__grade_', 'mothers_qualification', ...
                  'fathers_qualification', 'mothers_occupation', 'fathers_occupation', ...
                  'educational_special_needs', 'age_at_enrollment', 'international', ...
                  'curricular_units_1st_sem__without_evaluations_', ...
                  'curricular_units_2nd_sem__evaluations_'};

dropCols = intersect(manualExclude, train_tbl.Properties.VariableNames);
if ~isempty(dropCols)
    fprintf('Manually excluding %d columns: %s\n', numel(dropCols), strjoin(dropCols, ', '));
    train_tbl(:, dropCols) = [];
    test_tbl(:, dropCols)  = [];
end

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
        % text / categorical predictor -> one-hot encode manually
        % (avoids dummyvar, which isn't available on every installation)
        catCol = categorical(col);
        cats = categories(catCol);
        dummies = zeros(numel(catCol), numel(cats));
        for c = 1:numel(cats)
            dummies(:, c) = double(catCol == cats{c});
        end
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

%% ================= Local Functions =================
function tbl = addEngineeredFeatures(tbl)
%ADDENGINEEREDFEATURES Adds derived academic-performance features.
% Column names below match what MATLAB's makeValidName produces from
% names like "curricular_units_1st_sem_(enrolled)".

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
