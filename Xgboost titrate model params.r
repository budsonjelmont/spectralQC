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
write.table(xgb_train$results[sortorder,], 'xgb_train10.txt', row.names=FALSE)

# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = 'none')
