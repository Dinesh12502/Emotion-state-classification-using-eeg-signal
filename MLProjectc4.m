clc
clear all
close all

d1=load('DREAMER.mat');
d2=d1.DREAMER.Data;


ALL_HIGH_VAL_EEG=[];
ALL_LOW_VAL_EEG=[];

ALL_HIGH_ARO_EEG=[];
ALL_LOW_ARO_EEG=[];
for i=1:23  
    
 pat_d1=d2(i);
 
 %%%%% VALENCE EEG CLASSIFICATION DATABASE 
 
 %%%% Valence Score
 pat_val=pat_d1{1,1}.ScoreValence;
 
 PAT_HIGH_VAL_EEG=[];
 PAT_LOW_VAL_EEG=[];
 
 for k=1:18
     pat_EEG_ALL=pat_d1{1,1}.EEG.stimuli{k,1};
     pat_EEG1=pat_EEG_ALL(1:128*10,4);
     
     if(pat_val(k)>3)
          PAT_HIGH_VAL_EEG=[PAT_HIGH_VAL_EEG;[pat_EEG1' 1]];
     else
          PAT_LOW_VAL_EEG=[PAT_LOW_VAL_EEG;[pat_EEG1' 0]];
     end
 end
 
 ALL_HIGH_VAL_EEG=[ALL_HIGH_VAL_EEG;PAT_HIGH_VAL_EEG];
 ALL_LOW_VAL_EEG=[ALL_LOW_VAL_EEG;PAT_LOW_VAL_EEG];
 
 
 
 %%%% AROUSAL EEG CLASSIFICATION DATABASE 
 
 %%% Arousal Score
 pat_aro=pat_d1{1,1}.ScoreArousal;
 
 PAT_HIGH_ARO_EEG=[];
 PAT_LOW_ARO_EEG=[];
 
 for k=1:18
     %%% EEG DATA 2 Channel
     pat_EEG_ALL=pat_d1{1,1}.EEG.stimuli{k,1};
     pat_EEG1=pat_EEG_ALL(1:128*10,4);
     
     if(pat_aro(k)>3)
          PAT_HIGH_ARO_EEG=[PAT_HIGH_ARO_EEG;[pat_EEG1' 1]];
     else
          PAT_LOW_ARO_EEG=[PAT_LOW_ARO_EEG;[pat_EEG1' 0]];
     end
 end
 
 ALL_HIGH_ARO_EEG=[ALL_HIGH_ARO_EEG;PAT_HIGH_ARO_EEG];
 ALL_LOW_ARO_EEG=[ALL_LOW_ARO_EEG;PAT_LOW_ARO_EEG];
 
end


ALL_VAL_EEG_CLASS=[ALL_HIGH_VAL_EEG;ALL_LOW_VAL_EEG];
ALL_ARO_EEG_CLASS=[ALL_HIGH_ARO_EEG;ALL_LOW_ARO_EEG];


VALENCE=[];
AROUSAL=[];


%for VALENCE EEG DATABASE
for i=1:414
X1  = ALL_VAL_EEG_CLASS(i,1:1280);
label= ALL_VAL_EEG_CLASS(i,1281);
[c,l] = wavedec(X1,7,"db6");
[cd1,cd2,cd3,cd4,cd5,cd6,cd7] = detcoef(c,l,[1 2 3 4 5 6 7]);
delta=[cd7,cd6,cd5];
theta=cd4;
alpha=cd3;
beta=cd2;

%delta
delta_mean = mean(delta);
delta_std = std(delta);
delta_median = median(delta);
delta_skewness = skewness(delta);
delta_kurtosis = kurtosis(delta);
p = hist(delta,100);
p = p./sum(p);
delta_entropy = -sum(p.*log2(p + eps));
delta_power = mean(delta.^2);

%theta
theta_mean = mean(theta);
theta_std = std(theta);
theta_median = median(theta);
theta_skewness = skewness(theta);
theta_kurtosis = kurtosis(theta);
p = hist(theta,100);
p = p./sum(p);
theta_entropy = -sum(p.*log2(p + eps));
theta_power = mean(theta.^2);

%alpha
alpha_mean = mean(alpha);
alpha_std = std(alpha);
alpha_median = median(alpha);
alpha_skewness = skewness(alpha);
alpha_kurtosis = kurtosis(alpha);
p = hist(alpha,100);
p = p./sum(p);
alpha_entropy = -sum(p.*log2(p + eps));
alpha_power = mean(alpha.^2);

%beta
beta_mean = mean(beta);
beta_std = std(beta);
beta_median = median(beta);
beta_skewness = skewness(beta);
beta_kurtosis = kurtosis(beta);
p = hist(beta,100);
p = p./sum(p);
beta_entropy = -sum(p.*log2(p + eps));
beta_power = mean(beta.^2);



%normalised power;
sum_power= delta_power + theta_power + alpha_power + beta_power;
delta_nompower = delta_power/sum_power;
theta_nompower = theta_power/sum_power;
alpha_nompower = alpha_power/sum_power;
beta_nompower = beta_power/sum_power;


%create a database
database = [delta_mean,delta_std, delta_median, delta_skewness, delta_kurtosis, delta_entropy, delta_power,delta_nompower,theta_mean,theta_std, theta_median, theta_skewness, theta_kurtosis, theta_entropy, theta_power,theta_nompower,alpha_mean,alpha_std, alpha_median, alpha_skewness, alpha_kurtosis, alpha_entropy, alpha_power,alpha_nompower,beta_mean,beta_std, beta_median,beta_skewness, beta_kurtosis, beta_entropy, beta_power,beta_nompower,label];



VALENCE = [VALENCE; database];


end

%for AROUSAL EEG DATABASE
for i=1:414
X1  = ALL_ARO_EEG_CLASS(i,1:1280);
label= ALL_ARO_EEG_CLASS(i,1281);
[c,l] = wavedec(X1,7,"db6");
[cd1,cd2,cd3,cd4,cd5,cd6,cd7] = detcoef(c,l,[1 2 3 4 5 6 7]);
delta=[cd7,cd6,cd5];
theta=cd4;
alpha=cd3;
beta=cd2;

%delta
delta_mean = mean(delta);
delta_std = std(delta);
delta_median = median(delta);
delta_skewness = skewness(delta);
delta_kurtosis = kurtosis(delta);
p = hist(delta,100);
p = p./sum(p);
delta_entropy = -sum(p.*log2(p + eps));
delta_power = mean(delta.^2);

%theta
theta_mean = mean(theta);
theta_std = std(theta);
theta_median = median(theta);
theta_skewness = skewness(theta);
theta_kurtosis = kurtosis(theta);
p = hist(theta,100);
p = p./sum(p);
theta_entropy = -sum(p.*log2(p + eps));
theta_power = mean(theta.^2);

%alpha
alpha_mean = mean(alpha);
alpha_std = std(alpha);
alpha_median = median(alpha);
alpha_skewness = skewness(alpha);
alpha_kurtosis = kurtosis(alpha);
p = hist(alpha,100);
p = p./sum(p);
alpha_entropy = -sum(p.*log2(p + eps));
alpha_power = mean(alpha.^2);

%beta
beta_mean = mean(beta);
beta_std = std(beta);
beta_median = median(beta);
beta_skewness = skewness(beta);
beta_kurtosis = kurtosis(beta);
p = hist(beta,100);
p = p./sum(p);
beta_entropy = -sum(p.*log2(p + eps));
beta_power = mean(beta.^2);



%normalised power;
sum_power= delta_power + theta_power + alpha_power + beta_power;
delta_nompower = delta_power/sum_power;
theta_nompower = theta_power/sum_power;
alpha_nompower = alpha_power/sum_power;
beta_nompower = beta_power/sum_power;

%create a database
database = [delta_mean,delta_std, delta_median, delta_skewness, delta_kurtosis, delta_entropy, delta_power,delta_nompower,theta_mean,theta_std, theta_median, theta_skewness, theta_kurtosis, theta_entropy, theta_power,theta_nompower,alpha_mean,alpha_std, alpha_median, alpha_skewness, alpha_kurtosis, alpha_entropy, alpha_power,alpha_nompower,beta_mean,beta_std, beta_median,beta_skewness, beta_kurtosis, beta_entropy, beta_power,beta_nompower,label];



AROUSAL = [AROUSAL; database];


end

%% SVM valence
x = VALENCE(:, 1:32);
y = VALENCE(:,33);

rand = randperm(414);

xtr = x(rand(1:331), :);
ytr = y(rand(1:331), :);

xt = x(rand(332:end), :);
yt = y(rand(332:end), :);


%Training the model
model = fitcsvm(xtr, ytr, 'KernelFunction', 'rbf', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

%Test the model
resultsvm = predict(model, xt);
% Plot confusion matrix
cm = confusionmat(yt, resultsvm);
figure;
confusionchart(cm);

% Calculate Sensitivity
TP = cm(1,1); % True Positive
TN = cm(2,2); % True Negative
FP = cm(2,1); % False Positive
FN = cm(1,2); % False Negative
svmAccuracy = (TP + TN) / (TP + TN + FP + FN);

svmSensitivity = TP / (TP + FN);

% Calculate F1 score
Precision = TP / (TP + FP); % Precision
Recall = TP / (TP + FN); % Recall
svmF1_score = 2 * (Precision * Recall) / (Precision + Recall);

% Display results
disp(['Accuracy: ', num2str(svmAccuracy * 100), '%']);
disp(['Sensitivity: ', num2str(svmSensitivity)]);
disp(['F1 Score: ', num2str(svmF1_score)]);

%accuracy = sum(result == yt)/length(yt)*100;
%sp = sprintf("Test Accuracy = %.2f", accuracy);
%disp(sp);

%% randomforest valence


x = VALENCE(:, 1:32);
y = VALENCE(:,33);

rand = randperm(414);

xtr = x(rand(1:331), :);
ytr = y(rand(1:331), :);

xt = x(rand(332:end), :);
yt = y(rand(332:end), :);

%train the model
model = fitcensemble(xtr, ytr, 'Method', 'Bag', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));


%test the model
resultrdf=predict(model,xt);
% Plot confusion matrix
cm = confusionmat(yt, resultrdf);
figure;
confusionchart(cm);

% Calculate Sensitivity
TP = cm(1,1); % True Positive
TN = cm(2,2); % True Negative
FP = cm(2,1); % False Positive
FN = cm(1,2); % False Negative
Accuracy = (TP + TN) / (TP + TN + FP + FN);

Sensitivity = TP / (TP + FN);

% Calculate F1 score
Precision = TP / (TP + FP); % Precision
Recall = TP / (TP + FN); % Recall
F1_score = 2 * (Precision * Recall) / (Precision + Recall);

% Display results
disp(['SVMAccuracy: ', num2str(svmAccuracy * 100), '%']);
disp(['svmSensitivity: ', num2str(svmSensitivity)]);
disp(['svmF1 Score: ', num2str(svmF1_score)]);
disp(['Accuracy: ', num2str(Accuracy * 100), '%']);
disp(['Sensitivity: ', num2str(Sensitivity)]);
disp(['F1 Score: ', num2str(F1_score)]);


%ms=sum(resultrdf==yt)/length(yt)*100;
%rf = sprintf("Test Accuracy = %.2f", ms);
%disp(rf);
%% mlp valence
x = VALENCE(:, 1:32);
y = VALENCE(:, 33);

rng('default'); % For reproducibility
rand = randperm(414);

xtr = x(rand(1:331), :);
ytr = y(rand(1:331), :);

xt = x(rand(332:end), :);
yt = y(rand(332:end), :);

% Define and train the MLP model
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize); 

% Train the network
net = train(net, xtr', ytr');

% Test the model
resultmlp = net(xt'); 
resultmlp = round(resultmlp); 

cm = confusionmat(yt, resultrdf);
figure;
confusionchart(cm);

% Calculate Sensitivity
TP = cm(1,1); % True Positive
TN = cm(2,2); % True Negative
FP = cm(2,1); % False Positive
FN = cm(1,2); % False Negative
Accuracy = (TP + TN) / (TP + TN + FP + FN);

Sensitivity = TP / (TP + FN);
% Calculate F1 score
Precision = TP / (TP + FP); % Precision
Recall = TP / (TP + FN); % Recall
F1_score = 2 * (Precision * Recall) / (Precision + Recall);

% Display results
disp(['Accuracy: ', num2str(Accuracy * 100), '%']);
disp(['Sensitivity: ', num2str(Sensitivity)]);
disp(['F1 Score: ', num2str(F1_score)]);


%accuracy = sum(resultmlp' == yt) / length(yt) * 100; % Calculate accuracy
%disp(['Test Accuracy = ', num2str(accuracy), '%']);
%% SVM arousal
x = AROUSAL(:, 1:32);
y = AROUSAL(:,33);

rand = randperm(414);

xtr = x(rand(1:331), :);
ytr = y(rand(1:331), :);

xt = x(rand(332:end), :);
yt = y(rand(332:end), :);


%Training the model
model = fitcsvm(xtr, ytr, 'KernelFunction', 'rbf', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

%Test the model
resultsvm = predict(model, xt);
% Plot confusion matrix
cm = confusionmat(yt, resultsvm);
figure;
confusionchart(cm);

% Calculate Sensitivity
TP = cm(1,1); % True Positive
TN = cm(2,2); % True Negative
FP = cm(2,1); % False Positive
FN = cm(1,2); % False Negative
svmAccuracy = (TP + TN) / (TP + TN + FP + FN);

svmSensitivity = TP / (TP + FN);

% Calculate F1 score
Precision = TP / (TP + FP); % Precision
Recall = TP / (TP + FN); % Recall
svmF1_score = 2 * (Precision * Recall) / (Precision + Recall);
% Display results
disp(['SVMAccuracy: ', num2str(svmAccuracy * 100), '%']);
disp(['svmSensitivity: ', num2str(svmSensitivity)]);
disp(['svmF1 Score: ', num2str(svmF1_score)]);

%accuracy = sum(result == yt)/length(yt)*100;
%sp = sprintf("Test Accuracy = %.2f", accuracy);
%disp(sp);

%% randomforest arousal


x = AROUSAL(:, 1:32);
y = AROUSAL(:,33);

rand = randperm(414);

xtr = x(rand(1:331), :);
ytr = y(rand(1:331), :);

xt = x(rand(332:end), :);
yt = y(rand(332:end), :);

%train the model
model = fitcensemble(xtr, ytr, 'Method', 'Bag', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));


%test the model
resultrdf=predict(model,xt);
% Plot confusion matrix
cm = confusionmat(yt, resultrdf);
figure;
confusionchart(cm);

% Calculate Sensitivity
TP = cm(1,1); % True Positive
TN = cm(2,2); % True Negative
FP = cm(2,1); % False Positive
FN = cm(1,2); % False Negative
Accuracy = (TP + TN) / (TP + TN + FP + FN);

Sensitivity = TP / (TP + FN);

% Calculate F1 score
Precision = TP / (TP + FP); % Precision
Recall = TP / (TP + FN); % Recall
F1_score = 2 * (Precision * Recall) / (Precision + Recall);

% Display results
disp(['SVMAccuracy: ', num2str(svmAccuracy * 100), '%']);
disp(['svmSensitivity: ', num2str(svmSensitivity)]);
disp(['svmF1 Score: ', num2str(svmF1_score)]);
disp(['Accuracy: ', num2str(Accuracy * 100), '%']);
disp(['Sensitivity: ', num2str(Sensitivity)]);
disp(['F1 Score: ', num2str(F1_score)]);


%ms=sum(resultrdf==yt)/length(yt)*100;
%rf = sprintf("Test Accuracy = %.2f", ms);
%disp(rf);
%% mlp arousal
% Load your data
x = AROUSAL(:, 1:32);
y = AROUSAL(:, 33);

% Split the data into training and testing sets
rng('default'); % For reproducibility
rand = randperm(414);

xtr = x(rand(1:331), :);
ytr = y(rand(1:331), :);

xt = x(rand(332:end), :);
yt = y(rand(332:end), :);

% Define and train the MLP model
hiddenLayerSize = 10; % Choose the number of neurons in the hidden layer
net = patternnet(hiddenLayerSize); % Create the MLP model

% Train the network
net = train(net, xtr', ytr');

% Test the model
resultmlp = net(xt'); % Use the trained model to predict on test data
resultmlp = round(resultmlp); % Since this is a binary classification, round the output to 0 or 1

cm = confusionmat(yt, resultrdf);
figure;
confusionchart(cm);

% Calculate Sensitivity
TP = cm(1,1); % True Positive
TN = cm(2,2); % True Negative
FP = cm(2,1); % False Positive
FN = cm(1,2); % False Negative
Accuracy = (TP + TN) / (TP + TN + FP + FN);

Sensitivity = TP / (TP + FN);

% Calculate F1 score
Precision = TP / (TP + FP); % Precision
Recall = TP / (TP + FN); % Recall
F1_score = 2 * (Precision * Recall) / (Precision + Recall);
% Display results
disp(['Accuracy: ', num2str(Accuracy * 100), '%']);
disp(['Sensitivity: ', num2str(Sensitivity)]);
disp(['F1 Score: ', num2str(F1_score)]);