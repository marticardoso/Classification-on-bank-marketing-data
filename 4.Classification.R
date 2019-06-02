####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 4: 
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

library(MASS)
library(class)
library(e1071)
library(TunePareto) # for generateCVRuns()
library(glmnet)

#Load preprocessed data
load("bank-processed-train-test.Rdata")
load("bank-processed-cat-train-test.Rdata")

set.seed (104)

#First we define some function, useful for the script

# Function that compute the accuracy given prediction and real values
compute.accuracy <- function (pred, real)
{
  ct <- table(Truth=real, Pred=pred)
  round(100*(1-sum(diag(ct))/sum(ct)),2)
}

# Function that computes the harm coeficient given prediction and real values
harm <- function (a,b) { 2/(1/a+1/b) }
compute.harm <- function (pred, real)
{
  ct <- table(Truth=real, Pred=pred)
  harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2])
}

# Function that runs a k-fold-CV using:
# - The generateModelAndPredict function to create the model and predict in each fold, 
# - The goodness.func to compute the goodness of the current fold 
run.k.fold.CV <- function(generateModelAndPredict, dataset, goodness.func= compute.accuracy, k = 10){
  set.seed(1234)
  CV.folds <- generateCVRuns (dataset.train$y, ntimes=1, nfold=k, stratified=TRUE)
  acc = numeric(k)
  for (j in 1:k)
  {
    print(j)
    va <- unlist(CV.folds[[1]][[j]])
    pred.va <- generateModelAndPredict(dataset[-va,], dataset[va,])
    acc[j]<-goodness.func(pred.va, dataset[va,]$y)
  }
  return(acc)
}


####################################################################
# Logistic Regression
####################################################################

run.logisticRegression <- function (dataset, P=0.5)
{
  generateModelAndPredict <- function(train, newdata){
    glm.model <- glm (y~., train, family = binomial) 
    preds = predict(glm.model, newdata, type="response")
    preds[preds<P] <- 0
    preds[preds>=P] <- 1
    pred <- factor(preds, labels=c("no","yes"))
  }
  
  error = run.k.fold.CV(generateModelAndPredict,dataset, compute.harm)
  
  mean(error)
}

(glm.model.r <- run.logisticRegression(dataset.train, 0.3))
(glm.model.cat.r <- run.logisticRegression(dataset.cat.train,0.3))

####################################################################
# Ridge Regression and Lasso (logistic)
####################################################################

#Alpha = 1 -> Lasso
#Alpha = 0 -> Ridge
run.glmnet <- function (dataset, newdata, alpha = 1, P = 0.5)
{
  #Create dummy variables for categorical
  x <- model.matrix(y~., dataset)[,-1]
  y <- ifelse(dataset.train$y == "yes", 1, 0)
  
  cv <- cv.glmnet(x, y, alpha = alpha, family = "binomial")
  plot(cv)
  # Fit the final model on the training data
  model <- glmnet(x, y, alpha = alpha, family = "binomial", lambda=cv$lambda.1se)
  
  ## Compute training accuracy
  train.pred <- predict(model, newx=x, type="response")
  train.pred[train.pred<P] <- 0
  train.pred[train.pred>=P] <- 1
  train.pred <- factor(train.pred, labels=c("no","yes"))
  print('Train')
  print(trainTable <- table(Truth=dataset$y,Pred=train.pred))
  print(trainError <- 100*(1-sum(diag(trainTable))/(sum(trainTable))))
  
  ## Compute test accuracy
  x.test <- model.matrix(y ~., newdata)[,-1]
  preds <- predict (model, newx=x.test, type="response")
  preds[preds<P] <- 0
  preds[preds>=P] <- 1
  preds <- factor(preds, labels=c("no","yes"))
  print('Test')
  print(testTable <- table(Truth=newdata$y,Pred=preds))
  print(testError <- 100*(1-sum(diag(testTable))/sum(testTable)))
  
  list(model=model,trainError=trainError, testError=testError)
}

ridge.model <- run.glmnet(dataset.train, dataset.test, alpha= 0)
ridge.model.cat <- run.glmnet(dataset.cat.train, dataset.cat.test, alpha=0)

lasso.model <- run.glmnet(dataset.train, dataset.test, alpha= 1)
lasso.model.cat <- run.glmnet(dataset.cat.train, dataset.cat.test, alpha=1)

####################################################################
# LDA
####################################################################

run.LDA <- function (dataset, newdata)
{
  lda.model <- lda(y~., dataset) 
  plot(lda.model)
  ## Compute training accuracy
  train.pred <- predict (lda.model, dataset)$class
  print('Train')
  print(trainTable <- table(Truth=dataset$y,Pred=train.pred))
  print(trainError <- 100*(1-sum(diag(trainTable))/(sum(trainTable))))
  
  ## Compute test accuracy
  test.pred <- predict (lda.model, newdata)$class
  print('Test')
  print(testTable <- table(Truth=newdata$y,Pred=test.pred))
  print(testError <- 100*(1-sum(diag(testTable))/sum(testTable)))
  
  list(model=lda.model,trainError=trainError, testError=testError)
}

lda.model <- run.LDA(dataset.train, dataset.test)
lda.model.cat <- run.LDA(dataset.cat.train, dataset.cat.test)

####################################################################
# Knn
####################################################################

run.knn <- function (dataset, test, ks=c(1,3,5,7,10,round(sqrt(N))))
{
  
  for (k in ks)
  {
    pred <- knn(train=dataset, test=test, cl=dataset$y, k=k)
    
    plot(train, xlab="X1", ylab="X2", xlim=XLIM, ylim=YLIM, type="n")
    points(test, col=nicecolors[as.numeric(predicted)], pch=".")
    contour(grid.x, grid.y, matrix(as.numeric(predicted),grid.size,grid.size), 
            levels=c(1,2), add=TRUE, drawlabels=FALSE)
    
    # add training points, for reference
    points(train, col=nicecolors[t+1], pch=16)
    title(paste(myk,"-NN classification",sep=""))
  }
}

pred <- knn(train=dataset.train[,-18], test=dataset.test[,-18], cl=dataset.train$y, k=1)

lda.model <- run.QDA(dataset.train, dataset.test)
lda.model.cat <- run.QDA(dataset.cat.train, dataset.cat.test)

####################################################################
# Naïve Bayes
####################################################################

run.NaiveBayes <- function (dataset, newdata)
{
  model <- naiveBayes(y ~ ., data = dataset)
  
  ## Compute training accuracy
  train.pred <- predict (model, dataset)
  print('Train')
  print(trainTable <- table(Truth=dataset$y,Pred=train.pred))
  print(trainError <- 100*(1-sum(diag(trainTable))/(sum(trainTable))))
  
  ## Compute test accuracy
  test.pred <- predict (model, newdata)
  print('Test')
  print(testTable <- table(Truth=newdata$y,Pred=test.pred))
  print(testError <- 100*(1-sum(diag(testTable))/sum(testTable)))
  
  list(model=lda.model,trainError=trainError, testError=testError)
}

lda.model <- run.NaiveBayes(dataset.train, dataset.test)
lda.model.cat <- run.NaiveBayes(dataset.cat.train, dataset.cat.test)

####################################################################
# Multilayer Perceptrons
####################################################################

 
####################################################################
# Radial Basis Function Network 
####################################################################

####################################################################
# SVM
####################################################################
