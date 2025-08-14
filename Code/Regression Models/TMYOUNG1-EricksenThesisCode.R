#install any packages not yet installed
install.packages("readxl")
install.packages("caret")
install.packages("earth")
install.packages("rsample")
install.packages("pls")
install.packages("arm")
install.packages("gbm")
install.packages("bartMachine")
install.packages("rJava")
install.packages("ranger")
install.packages("brnn")
install.packages("nnet")
install.packages("writexl")
install.packages("janitor")
install.packages("imputeMissings")
install.packages("beepr")
#Load Libraries, validation, and grids -----------------------------------
#Load Libraries
library(readxl)
library(caret)
library(earth)
library(rsample)
library(pls)
library(arm)
library(gbm)
#library(bartMachine)
library(rJava)
library(ranger)
library(brnn)
library(nnet)
library(writexl)
library(janitor)
library(imputeMissings)
library(beepr)
#validation/multiple model types
cv <- trainControl(method = "cv", number = 10)
bartGrid <- expand.grid(num_trees = c(5, 10,15, 20, 25), k = 2, alpha = 0.95, beta = 2, nu = 3)
brnnGrid <- expand.grid(neurons = seq(from = 1, to = 4, by = 1))
nnetGrid <- expand.grid(size = seq(from = 0, to = 6, by = 1), decay = seq(from = 0.2, to = 1, by = 0.2))
# BUILD MODELS -----------------------------------------------------------
#import dataset
IB <- read_excel("/Users/johnsonjeeva/internship/Rprogram/IB (March21 - Jan22).xlsx")
#exclude door core
IB <- data.frame(subset(IB, AvgIB>42))
#split data 80/20
set.seed(314)
IB_split <- initial_split(IB,prop = 0.8)
IB_train <- training(IB_split)
IB_hold <- testing(IB_split)
rm("IB_split")
#clean dataset
IB_train <- clean_names(IB_train)
IB_hold <- clean_names(IB_hold)
#NOTE COLUMN NAMES, ADJUST ACCORDINGLY (above 2 functions may have changed column names)
#build models
SLR_IB <- lm(avg_ib ~., data = IB_train)
GLM_IB <- train(avg_ib ~ ., data = IB_train, method = 'glm', trControl = cv);beep(1)
MARS1_IB <- earth(avg_ib ~ ., data = IB_train, pmethod = "cv", nfold = 10, degree = 1);beep(1)
MARS2_IB <- earth(avg_ib ~ ., data = IB_train, pmethod = "cv", nfold = 10, degree = 2);beep(1)
MARS3_IB <- earth(avg_ib ~ ., data = IB_train, pmethod = "cv", nfold = 10, degree = 3);beep(1)
MARS4_IB <- earth(avg_ib ~ ., data = IB_train, pmethod = "cv", nfold = 10, degree = 4);beep(1)
bayesianGLM_IB <- train(avg_ib ~ ., data = IB_train, method = 'bayesglm', trControl = cv);beep(1)
PCR_IB <- pcr(avg_ib ~ ., data=IB_train, validation= "CV");beep(1)
boostedGLM_IB <- train(avg_ib ~ ., data = IB_train, method = 'gbm', trControl = cv);beep(1)
#partitionTree_IB <- train(avg_ib ~ ., data = IB_train, method = 'rpart', trControl = cv);beep(1)
#BART_IB <- train(avg_ib ~ ., data = IB_train, method = 'bartMachine', trControl = cv, tuneGrid = bartGrid);beep(1)
baggedRegressionTree_IB <- train(avg_ib ~ ., data = IB_train, method = 'treebag', trControl = cv, importance = TRUE);beep(1)
randomForest_IB <- train(avg_ib ~ ., data = IB_train, method = 'ranger', trControl = cv, importance = 'permutation');beep(1)
bayesianRegularizedANN_IB <- train(avg_ib ~ ., data = IB_train, method = 'brnn', trControl = cv, tuneGrid = brnnGrid, importance = TRUE);beep(1)
singleLayerANN_IB <- train(avg_ib ~ ., data = IB_train, method = 'nnet', trControl = cv, linout=TRUE, tuneGrid = nnetGrid, importance = TRUE);beep(1)
singleLayerSkipANN_IB <- train(avg_ib ~ ., data = IB_train, method = 'nnet', trControl = cv, linout=TRUE, tuneGrid = nnetGrid, skip = TRUE); beep(4)
#plots
plot(predict(SLR_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Simple Linear Regression)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(SLR_IB, newdata = IB_train))))
plot(predict(GLM_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Generalized Linear Model)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(GLM_IB, newdata = IB_train))))
plot(predict(MARS1_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [1])",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(MARS1_IB, newdata = IB_train))))
plot(predict(MARS2_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [2])",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(MARS2_IB, newdata = IB_train))))
plot(predict(MARS3_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [3])", 
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(MARS3_IB, newdata = IB_train))))
plot(predict(MARS4_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [4])",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(MARS4_IB, newdata = IB_train))))
plot(predict(bayesianGLM_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bayesian Generalized Linear Model)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(bayesianGLM_IB, newdata = IB_train))))
plot(predict(boostedGLM_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Boosted Generalized Linear Model)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(boostedGLM_IB, newdata = IB_train))))
#plot(predict(partitionTree_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Partition Tree)",
     #main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(partitionTree_IB, newdata = IB_train))))
#plot(predict(BART_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bayesian Additive Regression Trees)",
     #main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(BART_IB, newdata = IB_train))))
plot(predict(baggedRegressionTree_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bagged Regression Tree)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(baggedRegressionTree_IB, newdata = IB_train))))
plot(predict(randomForest_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Random Forest)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(randomForest_IB, newdata = IB_train))))
plot(predict(bayesianRegularizedANN_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bayesian Regularized ANN)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(bayesianRegularizedANN_IB, newdata = IB_train))))
plot(predict(singleLayerANN_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Monotone Multilayer Perceptron [1])",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(singleLayerANN_IB, newdata = IB_train))))
plot(predict(singleLayerSkipANN_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Monotone Multilayer Perceptron [skip])",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ predict(singleLayerSkipANN_IB, newdata = IB_train))))

plot(predict(SLR_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Simple Linear Regression)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(SLR_IB, newdata = IB_hold))))
plot(predict(GLM_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Generalized Linear Model)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(GLM_IB, newdata = IB_hold))))
plot(predict(MARS1_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [1])",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(MARS1_IB, newdata = IB_hold))))
plot(predict(MARS2_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [2])",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(MARS2_IB, newdata = IB_hold))))
plot(predict(MARS3_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [3])",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(MARS3_IB, newdata = IB_hold))))
plot(predict(MARS4_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Multivariate Adaptive Regression Splines [4])",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(MARS4_IB, newdata = IB_hold))))
plot(predict(bayesianGLM_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bayesian Generalized Linear Model)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(bayesianGLM_IB, newdata = IB_hold))))
plot(predict(boostedGLM_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Boosted Generalized Linear Model)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(boostedGLM_IB, newdata = IB_hold))))
#plot(predict(partitionTree_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Partition Tree)",
     #main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(partitionTree_IB, newdata = IB_hold))))
#plot(predict(BART_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bayesian Additive Regression Trees)",
     #main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(BART_IB, newdata = IB_hold))))
plot(predict(baggedRegressionTree_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bagged Regression Tree)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(baggedRegressionTree_IB, newdata = IB_hold))))
plot(predict(randomForest_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Random Forest)", 
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(randomForest_IB, newdata = IB_hold))))
plot(predict(bayesianRegularizedANN_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Bayesian Regularized ANN)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(bayesianRegularizedANN_IB, newdata = IB_hold))))
plot(predict(singleLayerANN_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Monotone Multilayer Perceptron [1])",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(singleLayerANN_IB, newdata = IB_hold))))
plot(predict(singleLayerSkipANN_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Monotone Multilayer Perceptron [skip])",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ predict(singleLayerSkipANN_IB, newdata = IB_hold))))
#compile results
tn_IB = nrow(IB_train)
hn_IB = nrow(IB_hold)
tact_IB = IB_train$avg_ib
hact_IB = IB_hold$avg_ib
#SLR_IB
tRMSE_SLR_IB <- sqrt(sum((((predict(SLR_IB, newdata = IB_train))-tact_IB)^2)/tn_IB))
tRsquared_SLR_IB <- 1-((sum((predict(SLR_IB,newdata = IB_train)-tact_IB)^2))/(sum((mean(predict(SLR_IB, newdata = IB_train))-tact_IB)^2)))
tMAE_SLR_IB <- (sum(abs(predict(SLR_IB, newdata = IB_train)-tact_IB)))/tn_IB
hRMSE_SLR_IB <- sqrt(sum((((predict(SLR_IB, newdata = IB_hold))-hact_IB)^2)/hn_IB))
hRsquared_SLR_IB <- 1-((sum((predict(SLR_IB,newdata = IB_hold)-hact_IB)^2))/(sum((mean(predict(SLR_IB, newdata = IB_hold))-hact_IB)^2)))
hMAE_SLR_IB <- (sum(abs(predict(SLR_IB, newdata = IB_hold)-hact_IB)))/hn_IB
#GLM_IB
tRMSE_GLM_IB <- postResample(predict(GLM_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_GLM_IB <- postResample(predict(GLM_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_GLM_IB <- postResample(predict(GLM_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_GLM_IB <- postResample(predict(GLM_IB, newdata = data.frame(IB_hold)), IB_hold$avg_ib)[1]
hRsquared_GLM_IB <- postResample(predict(GLM_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_GLM_IB <- postResample(predict(GLM_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#MARS1_IB
tRMSE_MARS1_IB <- sqrt(sum((((predict(MARS1_IB, newdata = IB_train))-tact_IB)^2)/tn_IB))
tRsquared_MARS1_IB <- 1-((sum((predict(MARS1_IB,newdata = IB_train)-tact_IB)^2))/(sum((mean(predict(MARS1_IB, newdata = IB_train))-tact_IB)^2)))
tMAE_MARS1_IB <- (sum(abs(predict(MARS1_IB, newdata = IB_train)-tact_IB)))/tn_IB
hRMSE_MARS1_IB <- sqrt(sum((((predict(MARS1_IB, newdata = IB_hold))-hact_IB)^2)/hn_IB))
hRsquared_MARS1_IB <- 1-((sum((predict(MARS1_IB,newdata = IB_hold)-hact_IB)^2))/(sum((mean(predict(MARS1_IB, newdata = IB_hold))-hact_IB)^2)))
hMAE_MARS1_IB <- (sum(abs(predict(MARS1_IB, newdata = IB_hold)-hact_IB)))/hn_IB
#MARS2_IB
tRMSE_MARS2_IB <- sqrt(sum((((predict(MARS2_IB, newdata = IB_train))-tact_IB)^2)/tn_IB))
tRsquared_MARS2_IB <- 1-((sum((predict(MARS2_IB,newdata = IB_train)-tact_IB)^2))/(sum((mean(predict(MARS2_IB, newdata = IB_train))-tact_IB)^2)))
tMAE_MARS2_IB <- (sum(abs(predict(MARS2_IB, newdata = IB_train)-tact_IB)))/tn_IB
hRMSE_MARS2_IB <- sqrt(sum((((predict(MARS2_IB, newdata = IB_hold))-hact_IB)^2)/hn_IB))
hRsquared_MARS2_IB <- 1-((sum((predict(MARS2_IB,newdata = IB_hold)-hact_IB)^2))/(sum((mean(predict(MARS2_IB, newdata = IB_hold))-hact_IB)^2)))
hMAE_MARS2_IB <- (sum(abs(predict(MARS2_IB, newdata = IB_hold)-hact_IB)))/hn_IB
#MARS3_IB
tRMSE_MARS3_IB <- sqrt(sum((((predict(MARS3_IB, newdata = IB_train))-tact_IB)^2)/tn_IB))
tRsquared_MARS3_IB <- 1-((sum((predict(MARS3_IB,newdata = IB_train)-tact_IB)^2))/(sum((mean(predict(MARS3_IB, newdata = IB_train))-tact_IB)^2)))
tMAE_MARS3_IB <- (sum(abs(predict(MARS3_IB, newdata = IB_train)-tact_IB)))/tn_IB
hRMSE_MARS3_IB <- sqrt(sum((((predict(MARS3_IB, newdata = IB_hold))-hact_IB)^2)/hn_IB))
hRsquared_MARS3_IB <- 1-((sum((predict(MARS3_IB,newdata = IB_hold)-hact_IB)^2))/(sum((mean(predict(MARS3_IB, newdata = IB_hold))-hact_IB)^2)))
hMAE_MARS3_IB <- (sum(abs(predict(MARS3_IB, newdata = IB_hold)-hact_IB)))/hn_IB
#MARS4_IB 
tRMSE_MARS4_IB <- sqrt(sum((((predict(MARS4_IB, newdata = IB_train))-tact_IB)^2)/tn_IB))
tRsquared_MARS4_IB <- 1-((sum((predict(MARS4_IB,newdata = IB_train)-tact_IB)^2))/(sum((mean(predict(MARS4_IB, newdata = IB_train))-tact_IB)^2)))
tMAE_MARS4_IB <- (sum(abs(predict(MARS4_IB, newdata = IB_train)-tact_IB)))/tn_IB
hRMSE_MARS4_IB <- sqrt(sum((((predict(MARS4_IB, newdata = IB_hold))-hact_IB)^2)/hn_IB))
hRsquared_MARS4_IB <- 1-((sum((predict(MARS4_IB,newdata = IB_hold)-hact_IB)^2))/(sum((mean(predict(MARS4_IB, newdata = IB_hold))-hact_IB)^2)))
hMAE_MARS4_IB <- (sum(abs(predict(MARS4_IB, newdata = IB_hold)-hact_IB)))/hn_IB
#bayesianGLM_IB
tRMSE_bayesianGLM_IB <- postResample(predict(bayesianGLM_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_bayesianGLM_IB <- postResample(predict(bayesianGLM_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_bayesianGLM_IB <- postResample(predict(bayesianGLM_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_bayesianGLM_IB <- postResample(predict(bayesianGLM_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_bayesianGLM_IB <- postResample(predict(bayesianGLM_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_bayesianGLM_IB <- postResample(predict(bayesianGLM_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#boostedGLM_IB 
tRMSE_boostedGLM_IB <- postResample(predict(boostedGLM_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_boostedGLM_IB <- postResample(predict(boostedGLM_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_boostedGLM_IB <- postResample(predict(boostedGLM_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_boostedGLM_IB <- postResample(predict(boostedGLM_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_boostedGLM_IB <- postResample(predict(boostedGLM_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_boostedGLM_IB <- postResample(predict(boostedGLM_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#partitionTree_IB
#tRMSE_partitionTree_IB <- postResample(predict(partitionTree_IB, newdata = IB_train), IB_train$avg_ib)[1]
#tRsquared_partitionTree_IB <- postResample(predict(partitionTree_IB, newdata = IB_train), IB_train$avg_ib)[2]
#tMAE_partitionTree_IB <- postResample(predict(partitionTree_IB, newdata = IB_train), IB_train$avg_ib)[3]
#hRMSE_partitionTree_IB <- postResample(predict(partitionTree_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
#hRsquared_partitionTree_IB <- postResample(predict(partitionTree_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
#hMAE_partitionTree_IB <- postResample(predict(partitionTree_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#BART_IB 
#tRMSE_BART_IB <- postResample(predict(BART_IB, newdata = IB_train), IB_train$avg_ib)[1]
#tRsquared_BART_IB <- postResample(predict(BART_IB, newdata = IB_train), IB_train$avg_ib)[2]
#tMAE_BART_IB <- postResample(predict(BART_IB, newdata = IB_train), IB_train$avg_ib)[3]
#hRMSE_BART_IB <- postResample(predict(BART_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
#hRsquared_BART_IB <- postResample(predict(BART_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
#hMAE_BART_IB <- postResample(predict(BART_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#baggedRegressionTree_IB
tRMSE_baggedRegressionTree_IB <- postResample(predict(baggedRegressionTree_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_baggedRegressionTree_IB <- postResample(predict(baggedRegressionTree_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_baggedRegressionTree_IB <- postResample(predict(baggedRegressionTree_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_baggedRegressionTree_IB <- postResample(predict(baggedRegressionTree_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_baggedRegressionTree_IB <- postResample(predict(baggedRegressionTree_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_baggedRegressionTree_IB <- postResample(predict(baggedRegressionTree_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#randomForest_IB
tRMSE_randomForest_IB <- postResample(predict(randomForest_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_randomForest_IB <- postResample(predict(randomForest_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_randomForest_IB <- postResample(predict(randomForest_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_randomForest_IB <- postResample(predict(randomForest_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_randomForest_IB <- postResample(predict(randomForest_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_randomForest_IB <- postResample(predict(randomForest_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#bayesianRegularizedANN_IB
tRMSE_bayesianRegularizedANN_IB <- postResample(predict(bayesianRegularizedANN_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_bayesianRegularizedANN_IB <- postResample(predict(bayesianRegularizedANN_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_bayesianRegularizedANN_IB <- postResample(predict(bayesianRegularizedANN_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_bayesianRegularizedANN_IB <- postResample(predict(bayesianRegularizedANN_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_bayesianRegularizedANN_IB <- postResample(predict(bayesianRegularizedANN_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_bayesianRegularizedANN_IB <- postResample(predict(bayesianRegularizedANN_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#singleLayerANN_IB
tRMSE_singleLayerANN_IB <- postResample(predict(singleLayerANN_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_singleLayerANN_IB <- postResample(predict(singleLayerANN_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_singleLayerANN_IB <- postResample(predict(singleLayerANN_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_singleLayerANN_IB <- postResample(predict(singleLayerANN_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_singleLayerANN_IB <- postResample(predict(singleLayerANN_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_singleLayerANN_IB <- postResample(predict(singleLayerANN_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#singleLayerSkipANN_IB
tRMSE_singleLayerSkipANN_IB <- postResample(predict(singleLayerSkipANN_IB, newdata = IB_train), IB_train$avg_ib)[1]
tRsquared_singleLayerSkipANN_IB <- postResample(predict(singleLayerSkipANN_IB, newdata = IB_train), IB_train$avg_ib)[2]
tMAE_singleLayerSkipANN_IB <- postResample(predict(singleLayerSkipANN_IB, newdata = IB_train), IB_train$avg_ib)[3]
hRMSE_singleLayerSkipANN_IB <- postResample(predict(singleLayerSkipANN_IB, newdata = IB_hold), IB_hold$avg_ib)[1]
hRsquared_singleLayerSkipANN_IB <- postResample(predict(singleLayerSkipANN_IB, newdata = IB_hold), IB_hold$avg_ib)[2]
hMAE_singleLayerSkipANN_IB <- postResample(predict(singleLayerSkipANN_IB, newdata = IB_hold), IB_hold$avg_ib)[3]
#IB Data Frame
#RMSE
tRMSE_IB <- c(tRMSE_SLR_IB, tRMSE_GLM_IB, tRMSE_MARS1_IB, tRMSE_MARS2_IB, 
              tRMSE_MARS3_IB, tRMSE_MARS4_IB, 
              tRMSE_bayesianGLM_IB, tRMSE_boostedGLM_IB, 
              #tRMSE_partitionTree_IB, tRMSE_BART_IB, 
              tRMSE_baggedRegressionTree_IB, tRMSE_randomForest_IB, 
              tRMSE_bayesianRegularizedANN_IB, tRMSE_singleLayerANN_IB, 
              tRMSE_singleLayerSkipANN_IB)
hRMSE_IB <- c(hRMSE_SLR_IB, hRMSE_GLM_IB, hRMSE_MARS1_IB, 
              hRMSE_MARS2_IB, hRMSE_MARS3_IB, 
              hRMSE_MARS4_IB, hRMSE_bayesianGLM_IB, 
              hRMSE_boostedGLM_IB, #hRMSE_partitionTree_IB, hRMSE_BART_IB, 
              hRMSE_baggedRegressionTree_IB, 
              hRMSE_randomForest_IB, hRMSE_bayesianRegularizedANN_IB, 
              hRMSE_singleLayerANN_IB, hRMSE_singleLayerSkipANN_IB)
#RSquared
tRsquared_IB <- c(tRsquared_SLR_IB, tRsquared_GLM_IB, tRsquared_MARS1_IB,
                  tRsquared_MARS2_IB, tRsquared_MARS3_IB,
                  tRsquared_MARS4_IB, tRsquared_bayesianGLM_IB,
                  tRsquared_boostedGLM_IB, #tRsquared_partitionTree_IB, tRsquared_BART_IB, 
                  tRsquared_baggedRegressionTree_IB,
                  tRsquared_randomForest_IB, tRsquared_bayesianRegularizedANN_IB, 
                  tRsquared_singleLayerANN_IB, tRsquared_singleLayerSkipANN_IB)
hRsquared_IB <- c(hRsquared_SLR_IB, hRsquared_GLM_IB, hRsquared_MARS1_IB,
                  hRsquared_MARS2_IB, hRsquared_MARS3_IB,
                  hRsquared_MARS4_IB, hRsquared_bayesianGLM_IB,
                  hRsquared_boostedGLM_IB, #hRsquared_partitionTree_IB,hRsquared_BART_IB, 
                  hRsquared_baggedRegressionTree_IB,
                  hRsquared_randomForest_IB, hRsquared_bayesianRegularizedANN_IB,
                  hRsquared_singleLayerANN_IB, hRsquared_singleLayerSkipANN_IB)
#MAE
tMAE_IB <- c(tMAE_SLR_IB, tMAE_GLM_IB, tMAE_MARS1_IB,
             tMAE_MARS2_IB, tMAE_MARS3_IB,
             tMAE_MARS4_IB, tMAE_bayesianGLM_IB,
             tMAE_boostedGLM_IB, #tMAE_partitionTree_IB,
             #tMAE_BART_IB, 
             tMAE_baggedRegressionTree_IB,
             tMAE_randomForest_IB, tMAE_bayesianRegularizedANN_IB,
             tMAE_singleLayerANN_IB, tMAE_singleLayerSkipANN_IB)
hMAE_IB <- c(hMAE_SLR_IB, hMAE_GLM_IB, hMAE_MARS1_IB,
             hMAE_MARS2_IB, hMAE_MARS3_IB,
             hMAE_MARS4_IB, hMAE_bayesianGLM_IB,
             hMAE_boostedGLM_IB, #hMAE_partitionTree_IB,
             #hMAE_BART_IB, 
             hMAE_baggedRegressionTree_IB,
             hMAE_randomForest_IB, hMAE_bayesianRegularizedANN_IB,
             hMAE_singleLayerANN_IB, hMAE_singleLayerSkipANN_IB)
vote_IB <- data.frame(tRMSE_IB,hRMSE_IB, tRsquared_IB, hRsquared_IB, tMAE_IB, hMAE_IB)

rownames(vote_IB) <- c("Simple Linear Regression", 
                       "Generalized Linear Model", 
                       "MARS (1)", "MARS (2)", "MARS (3)", "MARS (4)", 
                       "Bayesian GLM", "Boosted GLM", "Bagged Tree", 
                       "Random Forest", "Bayesian ANN", 
                       "Single Layer ANN", 
                       "Skip Layer ANN")
write_xlsx(vote_IB,"/Users/johnsonjeeva/internship/Rprogram/ModelResults20.xlsx")


# Weighted Prediction -----------------------------------------------------
#Top 3 IB
#1-RandomForest
#2-boostedGLM
#3-bayesANN
WP_IB_train = (predict(bayesianRegularizedANN_IB, newdata = IB_train))*(0.17)+(predict(boostedGLM_IB, newdata = IB_train))*(0.33)+(predict(randomForest_IB, newdata = IB_train))*(0.5)
WP_IB_hold = (predict(bayesianRegularizedANN_IB, newdata = IB_hold))*(0.17)+(predict(boostedGLM_IB, newdata = IB_hold))*(0.33)+(predict(randomForest_IB, newdata = IB_hold))*(0.5)
tRMSE_WP_IB <- sqrt(sum((WP_IB_train-tact_IB)^2)/tn_IB)
tRsquared_WP_IB <- 1-((sum((WP_IB_train-tact_IB)^2))/(sum((mean(WP_IB_train)-tact_IB)^2)))
tMAE_WP_IB <- (sum(abs(WP_IB_train-tact_IB)))/tn_IB
hRMSE_WP_IB <- sqrt(sum((WP_IB_hold-hact_IB)^2)/hn_IB)
hRsquared_WP_IB <- 1-((sum((WP_IB_hold-hact_IB)^2))/(sum((mean(WP_IB_hold)-hact_IB)^2)))
hMAE_WP_IB <- (sum(abs(WP_IB_hold-hact_IB)))/hn_IB
plot(WP_IB_train, IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Weighted Prediction)",
     main = "IB (all grades, excluding ULDC/ULD2)-Training Data (80%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_train$avg_ib ~ WP_IB_train)))
plot(WP_IB_hold, IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Weighted Prediction)",
     main = "IB (all grades, excluding ULDC/ULD2)-Holdout Data (20%)", xlim = c(50,130), ylim = c(50,130), abline(lm(IB_hold$avg_ib ~ WP_IB_hold)))


# Sample Loop -------------------------------------------------------------
#outside loop
i <- 3
#begin repeat loop
repeat {
  if (nrow(IB)>i ) {
    print("Code being executed");
    SLR_IB <- lm(avg_ib ~., data = IB_train);
    plot(predict(SLR_IB, newdata = IB_train), IB_train$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Simple Linear Regression)",
         main = "MOR (ULTR)-Training Data (80%)", xlim = c(1200,2400), ylim = c(1200,2600), abline(lm(IB_train$avg_ib ~ predict(SLR_IB, newdata = IB_train))));
    plot(predict(SLR_IB, newdata = IB_hold), IB_hold$avg_ib, col = "darkorange", ylab = "Actual (IB)", xlab = "Predicted (Simple Linear Regression)",
         main = "MOR (ULTR)-Holdout Data (20%)", xlim = c(1200,2400), ylim = c(1200,2600), abline(lm(IB_hold$avg_ib ~ predict(SLR_IB, newdata = IB_hold))));
    print(postResample(predict(SLR_IB, newdata = IB_train), IB_train$avg_ib)[1]);
    print(postResample(predict(SLR_IB, newdata = IB_train), IB_train$avg_ib)[2]);
    print(postResample(predict(SLR_IB, newdata = IB_train), IB_train$avg_ib)[3]);
    print(postResample(predict(SLR_IB, newdata = IB_hold), IB_hold$avg_ib)[1]);
    print(postResample(predict(SLR_IB, newdata = IB_hold), IB_hold$avg_ib)[2]);
    print(postResample(predict(SLR_IB, newdata = IB_hold), IB_hold$avg_ib)[3]);
    j=nrow(IB);
    print("From most recently recorded process parameters, predicted IB is...");
    print(predict(SLR_IB, newdata = IB)[j]);
    print("checking for new data");
    date_time<-Sys.time();
    i <- nrow(IB)
    while((as.numeric(Sys.time()) - as.numeric(date_time))<5){} #prints code running ever 5 sec
  } else {
    print("No new data, check back in 5 seconds")
    Sys.sleep(5) #5 seconds
    i <- nrow(IB)
  }
}

#varImp(MARS1_IB)
#varImp(GLM_IB)
#varImp(SLR_IB)

## MY CODE
# Model names (in order)
model_names <- c("SLR", "GLM", "MARS1", "MARS2", "MARS3", "MARS4",
                 "BayesianGLM", "BoostedGLM", "BaggedTree", "RandomForest",
                 "BayesianANN", "SingleLayerANN", "SkipLayerANN")

# Combine everything into a results table
results_summary <- data.frame(
  Model = model_names,
  Train_RMSE = round(tRMSE_IB, 2),
  Test_RMSE  = round(hRMSE_IB, 2),
  Train_R2   = round(tRsquared_IB, 4),
  Test_R2    = round(hRsquared_IB, 4)
)

# View in RStudio or R GUI
View(results_summary)  # Opens as a spreadsheet view
print(results_summary) # Console print

library(writexl)
write_xlsx(results_summary, "Model_Summary.xlsx")
