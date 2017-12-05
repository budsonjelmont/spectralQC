# Given the 4 data sets representing different ratio comparisons, this script iterates through all 16 combinations
# of data sets, using one as the training set and the other as the test (can be the same dataset), and assesses the 
# performance of both an SVM and a gradient-boosted regression tree approach to classifying the data
#
#	Inputs: 4 csvs--10to3.csv, 10to1.csv, 3to1.csv, and all.csv--containing accuracy classifications and measured QC metrics for every peptide in the given ratio comparison
#	Outputs: None. Results are stored in the svm.AUCdf and xgb.AUCdf objects.

library(e1071) #for SVM
library(xgboost) #for gradient-boosted regression trees
library(pROC)

setwd('C:/Users/Judson/Documents/QC metrics/1.5-2-5-10-fold')

datasets = c('10to1.csv','10to3.csv','3to1.csv','all.csv')

svm.AUCdf = data.frame(matrix(nrow = 4, ncol = 4))
colnames(svm.AUCdf) = rownames(svm.AUCdf) = c('tenVSone','tenVSthree','threeVSone','All')

svm.TPRdf = svm.AUCdf
svm.TNRdf = svm.AUCdf
xgb.AUCdf = svm.AUCdf
xgb.TPRdf = svm.AUCdf
xgb.TNRdf = svm.AUCdf
svm.preddf = list()
xgb.preddf = list()

#threshold to fix TPR/TNR at
threshold = 0.95

#Iterate over each data set, training SVM and XGB models on each 
for(i in 1:length(datasets)){
    
  traindat = read.table(datasets[i],header=TRUE,sep=",")
  traindat = traindat[,!names(traindat) %in% c("FoldDiff","Accuracy10","Accuracy2","Accuracy1.5","pArea")]
  trainlabels = traindat$Accuracy5

  #train the model and predict labels
  #SVM
  gamma = 0.05
  cost = 0.7
  svm.model = svm(Accuracy5 ~ ., data = traindat, cost = cost, gamma = gamma)
  
  #Gradient boosting
  traindatlist = list()
  traindatlist[['data']] = data.matrix(traindat[,-1])
  traindatlist[['label']] = data.matrix(traindat[,1])
  
  xgb.model = xgboost(data = traindatlist[['data']], 
                      label = traindatlist[['label']], 
                      eta = 0.01,
                      max_depth = 6, 
                      nround=900, 
                      subsample = 0.5,
                      colsample_bytree = 0.75,
                      gamma = 1,
                      #seed = 1,
                      eval_metric = "auc",
                      objective = "binary:logistic",
                      #objective = "multi:softmax",
                      #num_class = 2,   #only needed if objective="multi:softmax"
                      nthread = 3
                      )
    
  # Now iterate over datasets and apply the trained models to predict the labels of each
  for(j in 1:length(datasets)){
    #read and reformat test datasets
    testdat = read.table(datasets[j],header=TRUE,sep=",")
    testdat = testdat[,!names(testdat) %in% c("FoldDiff","Accuracy10","Accuracy2","Accuracy1.5","pArea")]
    testlabels = testdat$Accuracy5
    
    testdatlist = list()
    testdatlist[['data']] = data.matrix(testdat[,-1])
    testdatlist[['label']] = data.matrix(testdat[,1])
    
    #make SVM predictions
    svm.pred = predict(svm.model, testdat,decision.values=FALSE)
    svm.ROC = roc(testlabels,svm.pred)
    
    #make XGB predictions
    xgb.pred = predict(xgb.model, testdatlist$data)
    xgb.ROC = roc(testlabels,xgb.pred)
    
    #save predictions
    svm.preddf[[datasets[i]]][[datasets[j]]] = svm.pred
    xgb.preddf[[datasets[i]]][[datasets[j]]] = xgb.pred
    
    #save auc
    svm.AUCdf[i,j] = svm.ROC$auc
    xgb.AUCdf[i,j] = xgb.ROC$auc
    
    #save TPR @ 0.95 TNR
    index = range(which(svm.ROC$specificities-threshold >= 0))[1]
    svm.TPRdf[i,j] = svm.ROC$sensitivities[index]
    index = range(which(xgb.ROC$specificities-threshold >= 0))[1]
    xgb.TPRdf[i,j] = xgb.ROC$sensitivities[index]
    
    #save TNR @ 0.95 TPR
    index = range(which(svm.ROC$sensitivities-threshold >= 0))[2]
    svm.TNRdf[i,j] = svm.ROC$specificities[index]
    index = range(which(xgb.ROC$sensitivities-threshold >= 0))[2]
    xgb.TNRdf[i,j] = xgb.ROC$specificities[index]
  }
}
