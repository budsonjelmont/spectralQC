#Given a set of XGB model params passed in as a grid (provided by the caret package), this script trains the model
#using each combination of params and performs cross-validation. Then creates 3 output text files for the user
#to compare the performance of each model
#
#Input: training set data with accuracy labels (e.g. the '10to3.csv' file in '\\proteome\Judson\2_Projects\QC metrics\XGB parameter optimization\1.5-2-5-10-fold-difference as accuracy threshold')
#Output: the following 3 text files:
#			xgb_train.txt: general summary of model performance for each XGB parameter combination in the grid sorted on AUC
#			xgb_train.pred.txt: complete list of predicted & observed values in all the CV test sets for each combination of test SICs & XGB parameter combinations in the grid
#			xgb_trainresultsAUCs.txt: summary of AUCs @ 4 FPR thresholds (1%, 5%, 10%, 100%) for each XGB parameter combination in the grid sorted on AUC @ 0.05 FPR

library(xgboost)
library(caret)
library(doParallel)
library(AUC)

setwd('C:/Users/superuser/Documents/Spectral QC/1.5-2-5-10-fold-difference as accuracy threshold')
dat = read.table('10to3.csv',sep=',',header=TRUE) #1 = accurate @ given threshold, 0 = inaccurate

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
  nrounds = c(800, 900, 1000),
  eta = c(0.01, 0.1),
  max_depth = c(8, 10, 12),
  gamma = c(0, 1),
  colsample_bytree = c(0.75, 0.85),
  min_child_weight = c(0, 0.5)
)

xgb_grid_11 = expand.grid(
  nrounds = c(800, 900, 1000),
  eta = c(0.01, 0.1),
  max_depth = c(12, 14, 16, 18),
  gamma = c(0),
  colsample_bytree = c(0.75, 0.85),
  min_child_weight = c(0, 0.5)
)

xgb_grid_12 = expand.grid(
  nrounds = c(800, 900, 1000),
  eta = c(0.1, 0.5),
  max_depth = c(18, 20, 22, 24),
  gamma = c(0, 0.5),
  colsample_bytree = c(0.75, 0.85),
  min_child_weight = c(0, 0.5)
)

xgb_grid_13 = expand.grid(
  nrounds = c(800, 900, 1000),
  eta = c(0.1, 0.25, 0.5),
  max_depth = c(24, 26, 28, 30),
  gamma = c(0, 0.5),
  colsample_bytree = c(0.75, 0.85),
  min_child_weight = c(0, 0.5)
)

xgb_grid_14 = expand.grid(
  nrounds = c(800, 900, 1000),
  eta = c(0.1, 0.25, 0.5),
  max_depth = c(30, 50, 70),
  gamma = c(0, 0.5),
  colsample_bytree = c(0.75, 0.85),
  min_child_weight = c(0, 0.5)
)

xgb_grid_15 = expand.grid(
  nrounds = c(1000),
  eta = c(0.5, 0.75),
  max_depth = c(40, 50, 60),
  gamma = c(0),
  colsample_bytree = c(0.85),
  min_child_weight = c(0, 0.25)
)

xgb_grid_16 = expand.grid(
  nrounds = c(1000),
  eta = c(0.65, 0.75, 0.85),
  max_depth = c(40, 50, 60),
  gamma = c(0),
  colsample_bytree = c(0.85),
  min_child_weight = c(0, 0.05, 0.1)
)

xgb_grid_17 = expand.grid(
  nrounds = c(1000),
  eta = c(0.85, 0.95),
  max_depth = c(50, 60, 70),
  gamma = c(0),
  colsample_bytree = c(0.85, 0.9),
  min_child_weight = c(0)
)

xgb_grid_18 = expand.grid(
  nrounds = c(500, 1000),
  eta = c(0.05, 0.01),
  max_depth = c(8, 10, 12, 14),
  gamma = c(0, 1),
  colsample_bytree = c(0.5, 0.75),
  min_child_weight = c(1, 5, 7.5)
)

xgb_grid_19 = expand.grid(
  nrounds = c(500, 1000),
  eta = c(0.005, 0.01, 0.015, 0.02),
  max_depth = c(8, 10, 12),
  gamma = c(1),
  colsample_bytree = c(0.75, 0.8),
  min_child_weight = c(1, 5, 7.5)
)

xgb_grid_20 = expand.grid(
  nrounds = c(500, 1000),
  eta = c(0.005, 0.01, 0.015, 0.02),
  max_depth = c(8, 10, 12),
  gamma = c(1),
  colsample_bytree = c(0.7, 0.75),
  min_child_weight = c(4, 5, 6, 7)
)

xgb_grid_21 = expand.grid(
  nrounds = c(500, 750, 1000),
  eta = c(0.005, 0.01, 0.015),
  max_depth = c(8, 9, 10, 11),
  gamma = c(0.9, 1, 1.1),
  colsample_bytree = c(0.7, 0.75),
  min_child_weight = c(4.5, 5, 5.5, 6)
)

xgb_grid_22 = expand.grid(
  nrounds = c(750, 1000),
  eta = c(0.01, 0.015),
  max_depth = c(8, 9, 10, 11),
  gamma = c(0.9, 0.95),
  colsample_bytree = c(0.65, 0.7, 0.75),
  min_child_weight = c(4, 4.25, 4.5, 4.75)
)

xgb_grid_23 = expand.grid( #*
  nrounds = c(750, 1000),
  eta = c(0.01, 0.0125),
  max_depth = c(8, 9, 10, 11),
  gamma = c(0.925, 0.95, 0.975),
  colsample_bytree = c(0.7, 0.75),
  min_child_weight = c(3, 3.25, 3.5, 3.75, 4)
)

xgb_grid_24 = expand.grid(
  nrounds = c(750, 1000),
  eta = c(0.0095, 0.01, 0.011),
  max_depth = c(8, 9, 10),
  gamma = c(0.95, 0.975),
  colsample_bytree = c(0.725, 0.75, 0.775),
  min_child_weight = c(3.5)
)

xgb_grid = xgb_grid_24

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

#get predictions table
pred = xgb_train$pred

#write out results after sorting on increasing ROC
sortorder = order(-xgb_train$results[['ROC']])
write.table(xgb_train$results[sortorder,], 'xgb_train24.txt', row.names=FALSE)
write.table(pred, 'xgb_train24.pred.txt', row.names=FALSE)

# function to get partial AUC for select refion of ROC curve bounded by specific FPRs. Modified from AUC::auc.
getPartialAUC = function (x, min = 0, max = 1) 
{
	fprs = c()
	aucs = c()
    if (any(class(x) == "roc")) {
        if (min != 0 || max != 1) {
            x$tpr <- x$tpr[x$fpr >= min & x$fpr <= max]
            x$fpr <- x$fpr[x$fpr >= min & x$fpr <= max]
        }
        ans <- 0
        for (i in 2:length(x$fpr)) {
            ans <- ans + 0.5 * abs(x$fpr[i] - x$fpr[i - 1]) * 
                (x$tpr[i] + x$tpr[i - 1])
        }
    }
    return(ans)
}
# iterate through parameter set and get the AUCs @ various FPRs
doParamSummary = function(paramset,predictions){
	#grab params from xgb_grid matrix and get the relevant rows from the predictions data frame
	nrounds = unlist(paramset['nrounds'])
	eta = unlist(paramset['eta'])
	max_depth = unlist(paramset['max_depth'])
	gamma = unlist(paramset['gamma'])
	colsample_bytree = unlist(paramset['colsample_bytree'])
	min_child_weight = unlist(paramset['min_child_weight'])
	
	nrounds_index = which(predictions$nrounds == nrounds)
	eta_index = which(predictions$eta == eta)
	md_index = which(predictions$max_depth == max_depth)
	gamma_index = which(predictions$gamma == gamma)
	cs_index = which(predictions$colsample_bytree == colsample_bytree)
	mcw_index = which(predictions$min_child_weight == min_child_weight)
	
	index = Reduce(intersect, list(nrounds_index, eta_index, md_index, gamma_index, cs_index, mcw_index))
	
	#r = pROC::roc(predictions$obs[index], predictions$X1[index]) #pROC call to make ROC curve
	r = AUC::roc(predictions$X1[index],predictions$obs[index]) #AUC call to make ROC curve
	
	#AUC @ 0.01 FPR
    AUC01 = getPartialAUC(r, min=0, max=0.01)
	if(length(AUC01) == 0){AUC01 = 0}
	
	#AUC @ 0.05 FPR
    AUC05 = getPartialAUC(r, min=0, max=0.05)
	if(length(AUC05) == 0){AUC05 = 0}
	
	#AUC @ 0.10 FPR
    AUC10 = getPartialAUC(r, min=0, max=0.1)
	if(length(AUC10) == 0){AUC10 = 0}
	
	#Total AUC
	AUC = AUC::auc(r)
	
	return(data.frame(nrounds=nrounds, eta=eta, max_depth=max_depth, gamma=gamma,
		colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
		AUC01=AUC01, AUC05=AUC05, AUC10=AUC10, AUC=AUC))
}

results = apply(xgb_grid, 1, doParamSummary, pred)
results = do.call(rbind,results)

sortorder = order(-results$AUC05)
write.table(results[sortorder,], 'xgb_train24resultsAUCs.txt', row.names=FALSE)
