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
library(ggplot2)

#Load preprocessed data
load("bank-processed-train-test.Rdata")
load("bank-processed-cat-train-test.Rdata")

set.seed (104)

#First we load some useful function for the model selection task
source('modelSelectionUtils.R')

mca.result <- mca(dataset.cat.train[,c(-1,-18)], 5)
ds.mca <- data.frame(mca.result$rs)
ds.mca$y <- dataset.cat.train$y
#predict(mca2,dataset.cat.test[,c(-1,-18)])


####################################################################
# Logistic Regression
####################################################################

run.logisticRegression <- function (dataset,P=0.5)
{
  createModelAndPredict <- function(train, newdata){
    weights <- compute.weights(train$y)
    glm.model <- glm (y~., train, weights=weights,family = binomial) 
    #glm.model <- step(glm.model, trace=FALSE)
    preds <- predict(glm.model, newdata, type="response")
    return(probabilityToFactor(preds,P))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

(logReg.d1 = run.logisticRegression(dataset.train))
(logReg.d2 = run.logisticRegression(dataset.cat.train))
(logReg.d3 = run.logisticRegression(df.mca))

####################################################################
# Ridge Regression and Lasso (logistic)
####################################################################

#Function that runs 10-fold-cv using glmnet
#Alpha = 1 -> Lasso
#Alpha = 0 -> Ridge
run.glmnet <- function (dataset, lambda, alpha = 1, P = 0.5)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    #Create dummy variables for categorical
    x <- model.matrix(y~., train)[,-1]

    # Fit the final model on the training data
    model <- glmnet(x, train$y, alpha = alpha, weights = weights, family = "binomial", lambda=lambda)
    
    x.test <- model.matrix(y ~., newdata)[,-1]
    preds <- predict (model, newx=x.test, type="response")
    return(probabilityToFactor(preds,P))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

# Try several lambdas
run.glmnet.find.best.lambda <- function (dataset, lambda.v, alpha = 1, P = 0.5)
{
  #lambda.v <- get.default.lambdas(dataset,alpha)
  lambda.F1 = numeric(length(lambda.v))
  lambda.F1.sd = numeric(length(lambda.v))
  for(i in 1:(length(lambda.v))){
    print(paste("lambda ", lambda.v[i]))
    tmp <- run.glmnet(dataset, lambda=lambda.v[i], alpha= alpha, P=P)
    lambda.F1[i] <- tmp$F1.mean
    lambda.F1.sd[i] <- tmp$F1.sd
  }
  max.lambda.id <- which.max(lambda.F1)[1]
  
  return(list(lambda=lambda.v, 
              lambda.F1=lambda.F1, 
              lambda.F1.sd = lambda.F1.sd,
              max.lambda=lambda.v[max.lambda.id],
              max.F1 = lambda.F1[max.lambda.id],
              max.F1.sd = lambda.F1.sd[max.lambda.id]))
}

lambda.max <- 100
lambda.min <- 1e-4
n.lambdas <- 50
lambda.v <- exp(seq(log(lambda.min),log(lambda.max),length=n.lambdas))

glmnet.Lasso = run.glmnet.find.best.lambda(dataset.train,lambda.v,alpha=1)
glmnet.ridge = run.glmnet.find.best.lambda(dataset.train,lambda.v,alpha=0)

glmnet.cat.Lasso = run.glmnet.find.best.lambda(dataset.cat.train,lambda.v,alpha=1)
glmnet.cat.ridge = run.glmnet.find.best.lambda(dataset.cat.train,lambda.v,alpha=0)

d4.mca.Lasso = run.glmnet.find.best.lambda(df.mca,lambda.v,alpha=1)
d4.mca.ridge = run.glmnet.find.best.lambda(df.mca,lambda.v,alpha=0)

save(glmnet.Lasso,glmnet.ridge,glmnet.cat.Lasso,glmnet.cat.ridge, file = "glmnet-results.Rdata")
load("glmnet-results.Rdata")
#Plot results
df <- data.frame(lambda=c(lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v),
                 F1=c(glmnet.Lasso$lambda.F1, glmnet.ridge$lambda.F1, 
                      glmnet.cat.Lasso$lambda.F1, glmnet.cat.ridge$lambda.F1,
                      d4.mca.Lasso$lambda.F1,d4.mca.ridge$lambda.F1), 
                 sd=c(glmnet.Lasso$lambda.F1.sd, glmnet.ridge$lambda.F1.sd, 
                      glmnet.cat.Lasso$lambda.F1.sd, glmnet.cat.ridge$lambda.F1.sd,
                      d4.mca.Lasso$lambda.F1.sd,d4.mca.ridge$lambda.F1.sd),
                 group=c(rep('D1 Lasso',n.lambdas),rep('D1 Ridge',n.lambdas),
                         rep('D2 Lasso',n.lambdas),rep('D2 Ridge',n.lambdas),
                         rep('D4 (MCA) Lasso',n.lambdas),rep('D4 (MCA) Ridge',n.lambdas)))

ggplot(df[df$F1>0.5,], aes(x=log(lambda), y=F1, group=group, color=group)) + 
  geom_errorbar(aes(ymin=F1-sd, ymax=F1+sd), width=.01) +
  scale_y_continuous(name = "F1", limits = c(0.6, 0.8)) +
  geom_line() + geom_point()+theme_minimal()

####################################################################
# LDA
####################################################################

run.lda <- function (dataset)
{
  createModelAndPredict <- function(train, newdata){
    lda.model <- lda(y~., train) #Lda computes prior from data
    test.pred <- predict (lda.model, newdata)$class
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(d1.lda = run.lda(dataset.train))
(d2.cat.lda = run.lda(dataset.cat.train))
(d4.mca.lda = run.lda(df.mca))


####################################################################
# Naïve Bayes
####################################################################

run.NaiveBayes <- function (dataset, laplace=0)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- naiveBayes(y ~ ., data = train, weights=weights,laplace=laplace)
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(naive.F1 = run.NaiveBayes(dataset.train))
(naive.cat.F1 = run.NaiveBayes(dataset.cat.train))

(naive.lapl.F1 = run.NaiveBayes(dataset.train,laplace=1))
(naive.lapl.cat.F1 = run.NaiveBayes(dataset.cat.train,laplace=1))

(d4.mca.naive = run.NaiveBayes(df.mca))
(d4.mca.naive.lapl = run.NaiveBayes(df.mca,laplace=1))

####################################################################
# Multilayer Perceptrons
####################################################################


run.MLP <- function (dataset, nneurons, decay=0)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- nnet(y ~., data = train, weights = weights, size=nneurons, maxit=200, decay=decay, MaxNWts=10000)
    test.pred <- predict (model, newdata)
    return(probabilityToFactor(test.pred))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

# We fix a large number of hidden units in one hidden layer, and explore different regularization values
nneurons <- 10
decays <- 10^seq(-3,0,by=0.1)
d1.mlp.F1 <- numeric(length(decays))
d1.mlp.F1.sd <- numeric(length(decays))
for(i in 1:(length(decays))){
  print(paste("Decay ", decays[i]))
  tmp = run.MLP(dataset.train,nneurons,decay=decays[i])
  d1.mlp.F1[i] = tmp$F1.mean
  d1.mlp.F1.sd[i] = tmp$F1.sd

}

d2.mlp.F1 <- numeric(length(decays))
d2.mlp.F1.sd <- numeric(length(decays))
for(i in 1:(length(decays))){
  print(paste("Decay ", decays[i]))
  tmp = run.MLP(dataset.cat.train,nneurons,decay=decays[i])
  d2.mlp.F1[i] = tmp$F1.mean
  d2.mlp.F1.sd[i] = tmp$F1.sd
}

d4.mlp.F1 <- numeric(length(decays))
d4.mlp.F1.sd <- numeric(length(decays))
for(i in 1:(length(decays))){
  print(paste("Decay ", decays[i]))
  tmp = run.MLP(df.mca,nneurons,decay=decays[i])
  d4.mlp.F1[i] = tmp$F1.mean
  d4.mlp.F1.sd[i] = tmp$F1.sd
}

####################################################################
# SVM
####################################################################

library(e1071)
run.SVM <- function (dataset, C=1, which.kernel="linear", gamma=0.5)
{
  createModelAndPredict <- function(train, newdata){
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
  
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

#Run several C values
optimize.C <- function (dataset, Cs = 10^seq(-2,3), which.kernel="linear", gamma=0.5)
{
  z <- list()
  z$Cs <- Cs
  z$F1 <- numeric(length(Cs))
  z$F1.sd <- numeric(length(Cs))
  z$accuracy <- numeric(length(Cs))
  z$accuracy.sd <- numeric(length(Cs))
  for(i in 1:(length(Cs))){
    print(paste("C ", Cs[i]))
    tmp <- run.SVM(dataset,C=Cs[i], which.kernel=which.kernel, gamma=gamma)
    z$F1 <- tmp$F1.mean
    z$F1.sd[i] <- tmp$F1.sd
    z$accuracy <- tmp$accuracy.mean
    z$accuracy.sd[i] <- tmp$accuracy.sd
  }
  
  
  max.C.idx <- which.max(z$results.F1)[1]
  z$max.C <-Cs[max.C.idx]
  z$max.F1 <- z$F1[max.C.idx]
  z$max.F1.sd <- z$F1.sd[max.C.idx]
  z
}

#Linear kernel
Cs <- 10^seq(-2,3)
svm.lin.d1.F1 <- optimize.C(dataset.train,     Cs, which.kernel="linear")
svm.lin.d2.F1 <- optimize.C(dataset.cat.train, Cs, which.kernel="linear")

d4.svm.lin.F1 <- optimize.C(df.mca, c(0.1), which.kernel="linear")

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
  createModelAndPredict <- function(train, newdata){
    model <- tree(y ~ ., data=train)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  
  F1.by.fold = run.k.fold.CV(createModelAndPredict, dataset, performance.metric=F1)
  
  return(list(F1.mean=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}
(run.tree(dataset.train))
(run.tree(dataset.cat.train))

####################################################################
# Random forest
####################################################################

library(randomForest)
run.randomForest <- function (dataset, ntree=100)
{
  createModelAndPredict <- function(train, newdata){
    class.sampsize <- min(table(train$y))
    model <- randomForest(y ~ ., data=train, ntree=ntree, proximity=FALSE, 
                          sampsize=c(yes=class.sampsize, no=class.sampsize), strata=train$y)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  
  F1.by.fold = run.k.fold.CV(createModelAndPredict, dataset, performance.metric=F1)
  
  return(list(F1.mean=mean(F1.by.fold), F1.sd=sd(F1.by.fold)))
}

#Run severl ntrees
optimize.ntrees <- function(dataset){
  ntrees <- round(10^seq(1,3,by=0.2))
  results <- numeric(length(ntrees))
  for (i in 1:(length(ntrees)))
  { 
    print(paste("ntrees: ",nt))
    results[i] <- run.randomForest(dataset,ntrees[i])$F1.mean
  }
  max.ntrees.idx <- which.max(results)[1]
  
  return(list(ntrees=ntrees, 
              F1= results, 
              max.ntrees= F1[max.ntrees.idx],
              max.F1 = results.F1[max.ntrees.idx]))
}
run.randomForest(dataset.train)
optimize.ntrees(dataset.train)


####################################################################
# knn
####################################################################

####################################################################
# RBF-NN
####################################################################


