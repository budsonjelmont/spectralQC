# Reads in all 4 accuracy datasets and their assigned labels (0 = inaccurate, 1 = accurate). Trains a gradient-boosted tree
# on the 10X vs. 3X datasets and simultaneously tests the tree against each of the other 3 datasets at each iteration,
# then plots the error obtained for each dataset as a function of the iteration number.

library(xgboost)

setwd('C:/Users/Judson/Documents/QC metrics/1.5-2-5-10-fold')

#read and reformat data  
traindat = read.table('10to3.csv',header=TRUE,sep=",")
traindat = traindat[,!names(traindat) %in% c("FoldDiff","Accuracy10","Accuracy2","Accuracy1.5","pArea")]
  
traindatlist = list()
traindatlist[['data']] = data.matrix(traindat[,-1])
traindatlist[['label']] = data.matrix(traindat[,1])


dat10to1 = read.table('10to1.csv',header=TRUE,sep=",")
dat10to1 = dat10to1[,!names(dat10to1) %in% c("FoldDiff","Accuracy10","Accuracy2","Accuracy1.5","pArea")]
dat3to1 = read.table('3to1.csv',header=TRUE,sep=",")
dat3to1 = dat3to1[,!names(dat3to1) %in% c("FoldDiff","Accuracy10","Accuracy2","Accuracy1.5","pArea")]  
datALL = read.table('all.csv',header=TRUE,sep=",")
datALL = datALL[,!names(datALL) %in% c("FoldDiff","Accuracy10","Accuracy2","Accuracy1.5","pArea")]  

dat10to1list = list()
dat10to1list[['data']] = data.matrix(dat10to1[,-1])
dat10to1list[['label']] = data.matrix(dat10to1[,1])
dat3to1list = list()
dat3to1list[['data']] = data.matrix(dat3to1[,-1])
dat3to1list[['label']] = data.matrix(dat3to1[,1])  
datALLlist = list()
datALLlist[['data']] = data.matrix(datALL[,-1])
datALLlist[['label']] = data.matrix(datALL[,1])  


#Make watchlist of train and test data matrices
dtrain = xgb.DMatrix(data = data.matrix(traindat[,-1]), label = data.matrix(traindat[,1]))
d10to1 = xgb.DMatrix(data = data.matrix(dat10to1[,-1]), label = data.matrix(dat10to1[,1]))
d3to1 = xgb.DMatrix(data = data.matrix(dat3to1[,-1]), label = data.matrix(dat3to1[,1]))
dALL = xgb.DMatrix(data = data.matrix(datALL[,-1]), label = data.matrix(datALL[,1]))
  
watchlist = list(train = dtrain, test10to1 = d10to1, test3to1 = d3to1, testALL = dALL)

# function to get partial AUC for select refion of ROC curve bounded by specific FPRs. Modified from AUC::auc.
getPartialAUC = function (x, min = 0, max = 1) 
{
  fprs = c()
  aucs = c()
  if (any(class(x) == "roc")) {
    if (min != 0 || max != 1) {
      x$tpr = x$tpr[x$fpr >= min & x$fpr <= max]
      x$fpr = x$fpr[x$fpr >= min & x$fpr <= max]
    }
    ans = 0
    for (i in 2:length(x$fpr)) {
      ans = ans + 0.5 * abs(x$fpr[i] - x$fpr[i - 1]) * 
        (x$tpr[i] + x$tpr[i - 1])
    }
  }
  return(ans)
}

# custom evaluation metric -- AUC @ 0-0.05 FPR
AUC05FPR = function(preds, dtrain) {
  labels = factor(getinfo(dtrain, "label"))
  r = AUC::roc(preds,labels) #AUC call to make ROC curve
  AUC05 = getPartialAUC(r, min=0, max=0.05)
  print(paste('AUC05:', AUC05,sep=''))
  print(paste('AUC:', AUC::auc(r),sep=''))
  if(length(AUC05) == 0){AUC05 = 0}
  return(list(metric = "AUC05FPR", value = AUC05))
}

xgb.model = xgb.train(data = dtrain,
                      eta = 0.01,
                      max_depth = 9, 
                      nround = 2500, 
                      subsample = 0.5,
                      colsample_bytree = 0.7,
                      gamma = 0.9,
                      #seed = 1,
                      eval_metric = AUC05FPR,  #can be one of 'error', 'auc', etc...
                      objective = "binary:logistic",
                      nthread = 3,
                      watchlist = watchlist,
                      verbose = 1
)

#Plot outcome
eval = xgb.model$evaluation_log
evaldat = data.frame(AUC05FPR = c(eval$train_AUC05FPR, eval$test10to1_AUC05FPR, eval$test3to1_AUC05FPR, eval$testALL_AUC05FPR))
evaldat$Data = c(rep('Train_10to3', length(eval$train_AUC05FPR)), rep('Test_10to1', length(eval$test10to1_AUC05FPR)),
                 rep('Test_3to1', length(eval$test3to1_AUC05FPR)), rep('Test_ALL', length(eval$testALL_AUC05FPR)))
evaldat$Iteration = rep(eval$iter,4)

maxiter = range(evaldat$Iteration)[2]
maxerr = range(evaldat$AUC05FPR)[2]
maxyax = round(maxerr * 1.1,2)

ggplot(evaldat, aes(x=Iteration, y=AUC05FPR, group=Data)) +
  geom_line(aes(y=AUC05FPR, color=Data, group=Data), size=0.85) +
  scale_x_continuous(breaks=seq(0,maxiter,by=200), limits=c(0,maxiter), labels=seq(0,maxiter,by=200)) +
  scale_y_continuous(breaks=seq(0,maxyax,by=0.01), limits=c(0,maxyax), labels=seq(0,maxyax,by=0.01))
