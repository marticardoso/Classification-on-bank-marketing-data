####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 4: Classification
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
library(class)
library(nnet)

#Load preprocessed data
load("bank-processed-train-test.Rdata")
load("bank-processed-cat-train-test.Rdata")

set.seed (104)

#First we load some useful function for the model selection task
source('modelSelectionUtils.R')


####################################################################
# Logistic Regression
####################################################################

run.logisticRegression <- function (dataset,P=0.5)
{
  generateModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    glm.model <- glm (y~., train, weights=weights,family = binomial) 
    preds = predict(glm.model, newdata, type="response")
    return(probabilityToFactor(preds,P))
  }
  
  F1.by.fold = run.k.fold.CV(generateModelAndPredict,dataset, compute.F1)
  
  return(mean(F1.by.fold))
}

logReg.F1 = run.logisticRegression(dataset.train)
logReg.cat.F1 = run.logisticRegression(dataset.cat.train)

####################################################################
# Ridge Regression and Lasso (logistic)
####################################################################

#Function that runs 10-fold-cv using glmnet
#Alpha = 1 -> Lasso
#Alpha = 0 -> Ridge
run.glmnet <- function (dataset, lambda, alpha = 1, P = 0.5)
{
  generateModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    #Create dummy variables for categorical
    x <- model.matrix(y~., train)[,-1]
    y <- ifelse(train$y == "yes", 1, 0)

    # Fit the final model on the training data
    model <- glmnet(x, y, alpha = alpha, weights = weights, family = "binomial", lambda=lambda)
    
    x.test <- model.matrix(y ~., newdata)[,-1]
    preds <- predict (model, newx=x.test, type="response")
    return(probabilityToFactor(preds,P))
  }
  
  F1 = run.k.fold.CV(generateModelAndPredict,dataset, compute.F1)
  return(mean(F1))
}

#Use default lambdas (glmnet)
get.default.lambdas <- function (dataset, alpha = 1, nlambda=25)
{
    x <- model.matrix(y~., dataset)[,-1]
    y <- ifelse(dataset$y == "yes", 1, 0)
    model <- glmnet(x, y, alpha = alpha,nlambda=nlambda, weights = compute.weights(dataset$y), family = "binomial")
    return(model$lambda)
}

# Try several lambdas
run.glmnet.find.best.lambda <- function (dataset, alpha = 1, P = 0.5)
{
  lambda.v <- get.default.lambdas(dataset,alpha)
  lambda.F1 = numeric(length(lambda.v))
  for(i in 1:(length(lambda.v))){
    print(paste("lambda ", lambda.v[i]))
    lambda.F1[i] <- run.glmnet(dataset.train, dataset.test, lambda=lambda.v[i], alpha= alpha, P=P)
  }
  max.lambda.id <- which.max(lambda.res)[1]
  
  return(list(lambda=lambda.v, 
              lambda.F1=lambda.F1, 
              max.lambda=lambda.v[max.lambda.id],
              max.F1 = lambda.F1[max.lambda.id]))
}

glmnet.Lasso = run.glmnet.find.best.lambda(dataset.train,alpha=1)
glmnet.ridge = run.glmnet.find.best.lambda(dataset.train,alpha=0)

glmnet.cat.Lasso = run.glmnet.find.best.lambda(dataset.cat.train,alpha=1)
glmnet.cat.ridge = run.glmnet.find.best.lambda(dataset.cat.train,alpha=0)


####################################################################
# LDA
####################################################################

run.lda <- function (dataset)
{
  generateModelAndPredict <- function(train, newdata){
    lda.model <- lda(y~., train) #Lda computes prior from data
    test.pred <- predict (lda.model, newdata)$class
    return(test.pred)
  }
  F1.by.fold = run.k.fold.CV(generateModelAndPredict,dataset, compute.F1)
  return(list(F1=mean(F1.by.fold),F1.sd=sd(F1.by.fold)))
}

lda.F1 = run.lda(dataset.train)
lda.cat.F1 = run.lda(dataset.cat.train)


####################################################################
# Naïve Bayes
####################################################################

run.NaiveBayes <- function (dataset, laplace=0)
{
  generateModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- naiveBayes(y ~ ., data = train, weights=weights,laplace=laplace)
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  F1.by.fold = run.k.fold.CV(generateModelAndPredict,dataset, compute.F1)
  
  return(list(F1=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}

(naive.F1 = run.NaiveBayes(dataset.train))
(naive.cat.F1 = run.logisticRegression(dataset.cat.train))

(naive.lapl.F1 = run.NaiveBayes(dataset.train,laplace=1))
(naive.lapl.cat.F1 = run.logisticRegression(dataset.cat.train,laplace=1))

####################################################################
# Multilayer Perceptrons
####################################################################


run.MLP <- function (dataset, nneurons, decay=0)
{
  generateModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- nnet(y ~., data = train, weights = weights, size=nneurons, maxit=200, decay=decay, MaxNWts=10000)
    test.pred <- predict (model, newdata)
    return(probabilityToFactor(test.pred))
  }
  
  F1.by.fold = run.k.fold.CV(generateModelAndPredict,dataset, compute.F1)
  
  return(list(F1=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}

# We fix a large number of hidden units in one hidden layer, and explore different regularization values
nneurons <- 30
decays <- 10^seq(-3,0,by=0.1)
result.d1 <- numeric(length(decays))
result.d2 <- numeric(length(decays))
for(i in 1:(length(decays))){
  print(paste("Decay ", decays[i]))
  result.d1[i] = run.MLP(dataset.train,nneurons,decay=decays[i])
  result.d2[i] = run.MLP(dataset.cat.train,nneurons,decay=decays[i])
}


####################################################################
# SVM
####################################################################

library(e1071)
run.SVM <- function (dataset, C=1, which.kernel="linear", gamma=0.5)
{
  generateModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    switch(which.kernel,
           linear={model <- svm(y~., train, type="C-classification", cost=C, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(y~., train, type="C-classification", cost=C, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(y~., train, type="C-classification", cost=C, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF=   {model <- svm(y~., train, type="C-classification", cost=C, kernel="radial", gamma=gamma, scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  F1.by.fold = run.k.fold.CV(generateModelAndPredict, dataset, compute.F1)
  
  return(list(F1=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}

#Run several C values
optimize.C <- function (dataset, Cs = 10^seq(-2,3), which.kernel="linear", gamma=0.5)
{
  results.F1 <- numeric(length(Cs))
  for(i in 1:(length(Cs))){
    print(paste("C ", Cs[i]))
    results.F1[i] = run.SVM(dataset,C=Cs[i], which.kernel=which.kernel, gamma=gamma)
  }
  max.C.idx <- which.max(results.F1)[1]
  
  return(list(Cs=Cs, 
              Cs.F1=results.F1, 
              max.C=Cs[max.C.idx],
              max.F1 = results.F1[max.C.idx]))
}

#Linear kernel
Cs <- 10^seq(-2,3)
svm.lin.d1.F1 <- optimize.C(dataset.train,     Cs, which.kernel="linear")
svm.lin.d2.F1 <- optimize.C(dataset.cat.train, Cs, which.kernel="linear")

#Polinomial 2
svm.poly2.d1.F1 <- optimize.C(dataset.train,     Cs, which.kernel="poly.2")
svm.poly2.d2.F1 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.2")

#Polinomial 3
svm.poly2.d1.F1 <- optimize.C(dataset.train,     Cs, which.kernel="poly.3")
svm.poly2.d2.F1 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.3")


#RBF
svm.RBF.d1.F1 <- optimize.C(dataset.train,     Cs, which.kernel="poly.3")
svm.RBF.d2.F1 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.3")

gammas <- 2^seq(-3,4)
svm.RBF.d1.g.F1 <- numeric(length(gammas))
svm.RBF.d2.g.F1 <- numeric(length(gammas))
for (i in 1:(length(gammas)))#Gamma
{
  print(paste("gamma ", gammas[i]))
  svm.RBF.d1.g.F1[i] = run.SVM(dataset.train,C=svm.RBF.d1.F1$max.C, "RBF", gamma= gammas[i])
  svm.RBF.d2.g.F1[i] = run.SVM(dataset.cat.train,C=svm.RBF.d2.F1$max.C, "RBF", gamma=gammas[i])
  
}

####################################################################
# Tree
####################################################################

library(tree)
run.tree <- function (dataset)
{
  generateModelAndPredict <- function(train, newdata){
    model <- tree(y ~ ., data=train)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  
  F1.by.fold = run.k.fold.CV(generateModelAndPredict, dataset, compute.F1)
  
  return(list(F1=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}
run.tree(dataset.train)
run.tree(dataset.cat.train)

####################################################################
# Random forest
####################################################################

library(randomForest)
run.randomForest <- function (dataset, ntree=100)
{
  generateModelAndPredict <- function(train, newdata){
    class.sampsize <- min(table(train$y))
    model <- randomForest(y ~ ., data=train, ntree=ntree, proximity=FALSE, 
                          sampsize=c(yes=class.sampsize, no=class.sampsize), strata=train$y)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  
  F1.by.fold = run.k.fold.CV(generateModelAndPredict, dataset, compute.F1)
  
  return(list(F1=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}

#Run severl ntrees
optimize.ntrees <- function(dataset){
  ntrees <- round(10^seq(1,3,by=0.2))
  results <- numeric(length(ntrees))
  for (i in 1:(length(ntrees)))
  { 
    print(paste("ntrees: ",nt))
    results[i] <- run.randomForest(dataset,ntrees[i])$F1
  }
  max.ntrees.idx <- which.max(results)[1]
  
  return(list(ntrees=ntrees, 
              F1= results, 
              max.ntrees= F1[max.ntrees.idx],
              max.F1 = results.F1[max.ntrees.idx]))
}
run.randomForest(dataset.train)
optimize.ntrees(dataset.train)

