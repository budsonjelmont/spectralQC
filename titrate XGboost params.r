#Given a set of XGB model params passed in as a grid (provided by the caret package), this script trains the model
#using each combination of params and performs cross-validation. Then creates an output text file for the user
#to compare the performance of each model
#
#Input: training set data with accuracy labels (e.g. the '10to3.csv' file in '\\proteome\Judson\2_Projects\QC metrics\XGB parameter optimization\1.5-2-5-10-fold-difference as accuracy threshold')
#Output: xgb_train.txt: general summary of model performance for each XGB parameter combination in the grid sorted on AUC

library(xgboost)
library(caret)
library(pROC)
library(doParallel)

setwd('C:/Users/superuser/Documents/Spectral QC/1.5-2-5-10-fold-difference as accuracy threshold')
dat = read.table('10to3.csv',sep=',',header=TRUE)

datlist = list()
datlist[['labels']] = factor(make.names(dat$Accuracy2))
datlist[['data']] = as.matrix(dat[,!names(dat) %in% c('FoldDiff','Accuracy10','Accuracy5','Accuracy2','Accuracy1.5','pArea')])

#Using the approach discussed here:
#https://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees

xgb_grid_1 = expand.grid(
 nrounds = c(50, 100, 500),
 eta = c(0.01, 0.001),
 max_depth = c(2, 4, 6),
 gamma = c(0, 1),
 colsample_bytree = c(0.25, 0.5),
 min_child_weight = c(0, 1, 5)
)

xgb_grid_2 = expand.grid(
  nrounds = c(500, 1000, 2000),
  eta = c(0.01, 0.001),
  max_depth = c(2, 4, 6),
  gamma = c(0, 1, 2),
  colsample_bytree = c(0.25, 0.5),
  min_child_weight = c(1, 5, 10)
)

xgb_grid_3 = expand.grid(
  nrounds = c(500, 1000, 2000),
  eta = c(0.01, 0.001),
  max_depth = c(6, 8, 10),
  gamma = c(0, 1),
  colsample_bytree = c(0.5, 0.75),
  min_child_weight = c(1, 5, 7.5)
)

xgb_grid_4 = expand.grid(
  nrounds = c(500, 1000, 2000),
  eta = c(0.1, 0.01),
  max_depth = c(6, 8, 10),
  gamma = c(0, 1),
  colsample_bytree = c(0.75, 0.85, 0.95),
  min_child_weight = c(1, 3)
)

xgb_grid_5 = expand.grid(
  nrounds = c(800, 1000),
  eta = c(0.01, 0.05),
  max_depth = c(5, 6),
  gamma = c(0, 1, 2, 3),
  colsample_bytree = c(0.75),
  min_child_weight = c(1, 3)
)

xgb_grid_6 = expand.grid(
  nrounds = c(700, 800, 900),
  eta = c(0.01),
  max_depth = c(5, 6, 7),
  gamma = c(0, 1, 2),
  colsample_bytree = c(0.7, 0.75),
  min_child_weight = c(1, 3)
)

xgb_grid_7 = expand.grid(
  nrounds = c(700, 800, 900),
  eta = c(0.01),
  max_depth = c(6, 7),
  gamma = c(0, 1, 2),
  colsample_bytree = c(0.7, 0.75, 0.8),
  min_child_weight = c(0, 0.5, 1, 2)
)

xgb_grid_8 = expand.grid(
  nrounds = c(700, 800, 900),
  eta = c(0.01),
  max_depth = c(6, 7),
  gamma = c(0, 1, 2),
  colsample_bytree = c(0.65, 0.7, 0.75),
  min_child_weight = c(0, 0.5, 1)
)

xgb_grid_9 = expand.grid(
  nrounds = c(700, 800, 900, 1000),
  eta = c(0.01),
  max_depth = c(5, 6, 7, 8),
  gamma = c(0, 1, 2),
  colsample_bytree = c(0.65, 0.7, 0.75),
  min_child_weight = c(0, 0.5)
)

xgb_grid_10 = expand.grid(
  nrounds = c(700, 800),
  eta = c(0.005, 0.01),
  max_depth = c(5, 6),
  gamma = c(1),
  colsample_bytree = c(0.65),
  min_child_weight = c(0.5)
)

xgb_grid = xgb_grid_10

# grab all cores for parallel execution
cl = makeCluster(detectCores())
registerDoParallel(cl)

# pack the training control parameters
xgb_trcontrol = trainControl(
  method = 'cv',
  number = 5,
  verboseIter = TRUE,
  returnData = TRUE,
  savePredictions = TRUE,
  returnResamp = 'all',                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
# using CV to evaluate
xgb_train = train(
  x = datlist$data,
  y = datlist$labels,
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  method = 'xgbTree'
)

#release all cores
stopCluster(cl)

#write out results after sorting on increasing ROC
sortorder = order(-xgb_train$results[['ROC']])
write.table(xgb_train$results[sortorder,], 'xgb_train9.txt', row.names=FALSE)

# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = 'none')
