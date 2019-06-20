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
library(TunePareto)
library(glmnet)
library(class)
library(nnet)
library(ggplot2)
library(e1071)
library(tree)
library(randomForest)

#Load preprocessed data
load("bank-processed-train-test.Rdata")
load("bank-processed-cat-train-test.Rdata")
dataset.cat.train$age <- scale(dataset.cat.train$age)
load("D3.PCAMCA.dataset.Rdata")
load("D4.MCA.dataset.Rdata")
set.seed (104)

#First we load some useful function for the model selection task
source('modelSelectionUtils.R')

#mca.result <- mca(dataset.cat.train[,c(-1,-18)], 5)
#d4.mca.train <- data.frame(mca.result$rs)
#d4.mca.train$y <- dataset.cat.train$y
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
(logReg.d3 = run.logisticRegression(d3.pcamca.train))
(logReg.d4 = run.logisticRegression(d4.mca.train))

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
  results = list()
  for(i in 1:(length(lambda.v))){
    print(paste("lambda ", lambda.v[i]))
    results[[i]] <- run.glmnet(dataset, lambda=lambda.v[i], alpha= alpha, P=P)
  }
  z <- list(lambda=lambda.v)
  z$lambda.F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$lambda.F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$lambda.accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  z$lambda.accuracy.sd <- unlist(lapply(results,function(t) t$accuracy.sd))
  
  max.lambda.id <- which.max(z$lambda.F1)[1]
  z$max.lambda=lambda.v[max.lambda.id]
  z$max.F1 = z$lambda.F1[max.lambda.id]
  z$max.F1.sd = z$lambda.F1.sd[max.lambda.id]
  z
}

lambda.max <- 100
lambda.min <- 1e-4
n.lambdas <- 50
lambda.v <- exp(seq(log(lambda.min),log(lambda.max),length=n.lambdas))

d1.Lasso = run.glmnet.find.best.lambda(dataset.train,lambda.v,alpha=1)
d1.ridge = run.glmnet.find.best.lambda(dataset.train,lambda.v,alpha=0)

d2.Lasso = run.glmnet.find.best.lambda(dataset.cat.train,lambda.v,alpha=1)
d2.ridge = run.glmnet.find.best.lambda(dataset.cat.train,lambda.v,alpha=0)

d3.Lasso = run.glmnet.find.best.lambda(d3.pcamca.train,lambda.v,alpha=1)
d3.ridge = run.glmnet.find.best.lambda(d3.pcamca.train,lambda.v,alpha=0)

d4.Lasso = run.glmnet.find.best.lambda(d4.mca.train,lambda.v,alpha=1)
d4.ridge = run.glmnet.find.best.lambda(d4.mca.train,lambda.v,alpha=0)

save(d1.Lasso,d1.ridge,d2.Lasso,d2.ridge,d3.Lasso,d3.ridge,d4.Lasso,d4.ridge, file = "tmp/glmnet-results.Rdata")
load("tmp/glmnet-results.Rdata")
#Plot results
df <- data.frame(lambda=c(lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v),
                 F1=c(d1.Lasso$lambda.F1, d1.ridge$lambda.F1, 
                      d2.Lasso$lambda.F1, d2.ridge$lambda.F1,
                      d3.Lasso$lambda.F1, d3.ridge$lambda.F1,
                      d4.Lasso$lambda.F1, d4.ridge$lambda.F1), 
                 sd=c(d1.Lasso$lambda.F1.sd, d1.ridge$lambda.F1.sd, 
                      d2.Lasso$lambda.F1.sd, d2.ridge$lambda.F1.sd,
                      d3.Lasso$lambda.F1.sd, d3.ridge$lambda.F1.sd,
                      d4.Lasso$lambda.F1.sd, d4.ridge$lambda.F1.sd),
                 group=c(rep('D1 Lasso',n.lambdas),rep('D1 Ridge',n.lambdas),
                         rep('D2 Lasso',n.lambdas),rep('D2 Ridge',n.lambdas),
                         rep('D3 (PCA+MCA) Lasso',n.lambdas),rep('D3 (PCA+MCA) Ridge',n.lambdas),
                         rep('D4 (MCA) Lasso',n.lambdas),rep('D4 (MCA) Ridge',n.lambdas)))

ggplot(df[df$F1>0.5,], aes(x=lambda, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1", limits = c(0.65, 0.75)) + scale_x_continuous(trans='log2')+
  geom_line() + theme_minimal()

ggplot(df[df$F1>0.5,], aes(x=lambda, y=sd, group=group, color=group)) + 
  scale_y_continuous(name = "sd(F1)",limits = c(0, 0.028)) + scale_x_continuous(trans='log2')+
  geom_line() + theme_minimal()

####################################################################
# LDA
####################################################################

run.lda <- function (dataset)
{
  createModelAndPredict <- function(train, newdata){
    lda.model <- lda(y~., train,prior=c(1,1)/2) 
    test.pred <- predict (lda.model, newdata)$class
    return(test.pred)
    #test.prob <- predict (lda.model, newdata)$posterior[,2]
    #return(probabilityToFactor(test.prob,P))
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(d1.lda = run.lda(dataset.train))
(d2.lda = run.lda(dataset.cat.train))
(d3.lda = run.lda(d3.pcamca.train))
(d4.lda = run.lda(d4.mca.train))

####################################################################
# QDA
####################################################################

run.qda <- function (dataset)
{
  createModelAndPredict <- function(train, newdata){
    qda.model <- qda(y~., train,prior=c(1,1)/2) 
    test.pred <- predict (qda.model, newdata)$class
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

#(d1.qda = run.qda(dataset.train))
#(d2.qda = run.qda(dataset.cat.train))
(d3.qda = run.qda(d3.pcamca.train))
(d4.qda = run.qda(d4.mca.train))

####################################################################
# knn
####################################################################

run.knn <- function (dataset, k)
{
  createModelAndPredict <- function(train, newdata){
    x.train <- model.matrix(y~., train)[,-1]
    x.test <- model.matrix(y~., newdata)[,-1]
    pred <- knn(x.train, x.test, train$y, k=k)
    pred
  }
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

# Try several ks
optimize.knn.k <- function (dataset, ks = seq(1,15,2))
{
  results = list()
  for(i in 1:(length(ks))){
    print(paste("k = ", ks[i]))
    results[[i]] <- run.knn(dataset, ks[i])
  }
  z <- list(ks=ks,results=results)
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  z$accuracy.sd <- unlist(lapply(results,function(t) t$accuracy.sd))
  
  max.id <- which.max(z$F1)[1]
  z$max.k=ks[max.id]
  z$max.F1 = z$F1[max.id]
  z$max.F1.sd = z$F1.sd[max.id]
  z
}
(d1.knn <- optimize.knn.k(dataset.train))
#(d2.knn <- optimize.knn.k(dataset.cat.train))
(d3.knn <- optimize.knn.k(d3.pcamca.train))
(d4.knn <- optimize.knn.k(d4.mca.train))

save(d1.knn,d3.knn,d4.knn, file = "tmp/knn-results.Rdata")
load("tmp/knn-results.Rdata")
#Plot results
df <- data.frame(k=c(d3.knn$ks,d4.knn$ks, d1.knn$ks),
                 F1=c(d3.knn$F1,    d4.knn$F1, d1.knn$F1), 
                 acc=c(d3.knn$accuracy, d4.knn$accuracy, d1.knn$accuracy),
                 group=c(rep('D3',length(d3.knn$ks)),rep('D4',length(d3.knn$ks)),rep('D1',length(d1.knn$ks))))

ggplot(df, aes(x=k, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1")+ geom_line() + theme_minimal()

ggplot(df, aes(x=k, y=acc, group=group, color=group)) + 
  scale_y_continuous(name = "Accuracy") + geom_line() + theme_minimal()

####################################################################
# Naïve Bayes
####################################################################

run.NaiveBayes <- function (dataset, laplace=0)
{
  createModelAndPredict <- function(train, newdata){
    model <- naiveBayes(y ~ ., data = train, laplace=laplace)
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(d1.naive = run.NaiveBayes(dataset.train))
(d1.naive.lapl = run.NaiveBayes(dataset.train,laplace=1))

(d2.naive = run.NaiveBayes(dataset.cat.train))
(d2.naive.lapl = run.NaiveBayes(dataset.cat.train,laplace=1))

(d3.naive = run.NaiveBayes(d3.pcamca.train))
(d3.naive.lapl = run.NaiveBayes(d3.pcamca.train,laplace=1))

(d4.naive = run.NaiveBayes(d4.mca.train))
(d4.naive.lapl = run.NaiveBayes(d4.mca.train,laplace=1))

ls = c(0:10)
result = numeric(length(ls))
for(i in 1:length(ls)){
  result[i] = run.NaiveBayes(dataset.train,laplace=ls[i])$F1.mean
}
plot(ls,result,type='l')

####################################################################
# Multilayer Perceptrons
####################################################################

run.MLP <- function (dataset, nneurons, decay=0)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- nnet(y ~., data = train, weights = weights, size=nneurons, 
                  maxit=100, decay=decay, MaxNWts=10000, trace=FALSE)
    print( c("Weights",length(model$wts)))
    test.pred <- predict (model, newdata)
    return(probabilityToFactor(test.pred))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

optimize.decay <- function(dataset, nneurons, decays=c(0,10^seq(-3,0,by=0.1))){
  results <- list()
  for (i in 1:(length(decays)))
  { 
    print(paste("Decay: ",decays[i]))
    results[[i]] <- run.MLP(dataset,nneurons, decays[i])
  }
  z = list(decays = decays, results=results)
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  max.idx <- which.max(z$F1)[1]
  z$max.decay <- z$decays[max.idx]
  z$max.F1 <- z$F1[max.idx]
  z
}

# We fix a large number of hidden units in one hidden layer, and explore different regularization values
nneurons <- 20
decays <- c(0,10^seq(-3,0,by=1))

(d1.mlp <- optimize.decay(dataset.train,    nneurons, decays))
(d2.mlp <- optimize.decay(dataset.cat.train,nneurons, decays))
(d3.mlp <- optimize.decay(d3.pcamca.train,  nneurons, decays))
(d4.mlp <- optimize.decay(d4.mca.train,     nneurons, decays))

####################################################################
# SVM
####################################################################


run.SVM <- function (dataset, C=1, which.kernel="linear", gamma=0.5)
{
  createModelAndPredict <- function(train, newdata){
    class.weights <- 1-table(train$y)/nrow(train) #Give more weights to YES
    switch(which.kernel,
           linear={model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF=   {model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="radial", gamma=gamma, scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  run.k.fold.CV(createModelAndPredict, dataset, k=10, performance.metric=c("accuracy","F1"))
}

#Run several C values
optimize.C <- function (dataset, Cs = 10^seq(-2,3), which.kernel="linear", gamma=0.5)
{
  results <- list()
  
  for(i in 1:(length(Cs))){
    print(paste("C ", Cs[i]))
    results[[i]] <- run.SVM(dataset,C=Cs[i], which.kernel=which.kernel, gamma=gamma)
  }
  
  z = list(Cs = Cs, results=results)
  
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  
  max.C.idx <- which.max(z$F1)[1]
  z$max.C <-Cs[max.C.idx]
  z$max.F1 <- z$F1[max.C.idx]
  z$max.F1.sd <- z$F1.sd[max.C.idx]
  z
}

#Linear kernel
Cs <- 10^seq(-3,1)
d1.svm.lin.tmp <- optimize.C(dataset.train,     c(1e2), which.kernel="linear")
d2.svm.lin <- optimize.C(dataset.cat.train, Cs, which.kernel="linear")
d3.svm.lin <- optimize.C(d3.pcamca.train  , Cs, which.kernel="linear")
d4.svm.lin <- optimize.C(d4.mca.train     , Cs, which.kernel="linear")

d4.svm.lin$Cs <-       c(d4.svm.lin.tmp$Cs,d4.svm.lin$Cs)
d4.svm.lin$F1 <-       c(d4.svm.lin.tmp$F1,d4.svm.lin$F1)
d4.svm.lin$F1.sd <-    c(d4.svm.lin.tmp$F1.sd,d4.svm.lin$F1.sd)
d4.svm.lin$accuracy <- c(d4.svm.lin.tmp$accuracy,d4.svm.lin$accuracy)
for( i in 5:1){
  d4.svm.lin$results[[i+1]] <- d4.svm.lin$results[[i]]
}
d4.svm.lin$results[[1]] <- d4.svm.lin.tmp$results[[1]]


save(Cs,d1.svm.lin,d2.svm.lin,d3.svm.lin,d4.svm.lin, file = "tmp/smv-lin-results-v2.Rdata")
load("tmp/smv-lin-results-v2.Rdata")
df.res.lin <- data.frame(k=c(d1.svm.lin$Cs,d2.svm.lin$Cs,d3.svm.lin$Cs,d4.svm.lin$Cs),
                 F1=c(d1.svm.lin$F1,  d2.svm.lin$F1, d3.svm.lin$F1, d4.svm.lin$F1), 
                 accuracy=c(d1.svm.lin$accuracy, d2.svm.lin$accuracy, d3.svm.lin$accuracy,d4.svm.lin$accuracy),
                 group=c(rep('D1 (lineal)',length(d1.svm.lin$Cs)),rep('D2 (lineal)',length(d2.svm.lin$Cs)),rep('D3 (lineal)',length(d3.svm.lin$Cs)),rep('D4 (lineal)',length(d4.svm.lin$Cs))))

ggplot(df.res.lin, aes(x=k, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1") + scale_x_continuous(trans='log10') + geom_line() + theme_minimal()


#Polinomial 2
d1.svm.poly2 <- optimize.C(dataset.train,     Cs, which.kernel="poly.2")
d2.svm.poly2 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.2")
d3.svm.poly2 <- optimize.C(d3.pcamca.train,   Cs, which.kernel="poly.2")
d4.svm.poly2 <- optimize.C(d4.mca.train,      Cs, which.kernel="poly.2")



save(Cs,d1.svm.poly2,d2.svm.poly2,d3.svm.poly2,d4.svm.poly2, file = "tmp/svm-poly2-results-v2.Rdata")
load("tmp/svm-poly2-results-v2.Rdata")
df.res.poly2 <- data.frame(k=c(d1.svm.poly2$Cs, d2.svm.poly2$Cs, d3.svm.poly2$Cs, d4.svm.poly2$Cs),
                         F1=c(d1.svm.poly2$F1,  d2.svm.poly2$F1, d3.svm.poly2$F1, d4.svm.poly2$F1), 
                         accuracy=c(d1.svm.poly2$accuracy, d2.svm.poly2$accuracy, d3.svm.poly2$accuracy,d4.svm.poly2$accuracy),
                         group=c(rep('D1 (poly2)',length(d1.svm.poly2$Cs)),rep('D2 (poly2)',length(d2.svm.poly2$Cs)),rep('D3 (poly2)',length(d3.svm.poly2$Cs)),rep('D4 (poly2)',length(d4.svm.poly2$Cs))))

ggplot(df.res.poly2, aes(x=k, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1") + scale_x_continuous(trans='log10') + geom_line() + theme_minimal()

#Polinomial 3
d1.svm.poly3 <- optimize.C(dataset.train,     Cs, which.kernel="poly.3")
d2.svm.poly3 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.3")
d3.svm.poly3 <- optimize.C(d3.pcamca.train,   Cs, which.kernel="poly.3")
d4.svm.poly3 <- optimize.C(d4.mca.train,      Cs, which.kernel="poly.3")

save(Cs,d1.svm.poly3,d2.svm.poly3,d3.svm.poly3,d4.svm.poly3, file = "tmp/svm-poly3-results-v2.Rdata")
load("tmp/svm-poly3-results-v2.Rdata")
df.res.poly3 <- data.frame(k=c(d1.svm.poly3$Cs,d2.svm.poly3$Cs,d3.svm.poly3$Cs,d4.svm.poly3$Cs),
                           F1=c(d1.svm.poly3$F1,  d2.svm.poly3$F1, d3.svm.poly3$F1, d4.svm.poly3$F1), 
                           accuracy=c(d1.svm.poly3$accuracy, d2.svm.poly3$accuracy, d3.svm.poly3$accuracy,d4.svm.poly3$accuracy),
                           group=c(rep('D1 (poly3)',length(d1.svm.poly3$Cs)),rep('D2 (poly3)',length(d2.svm.poly3$Cs)),rep('D3 (poly3)',length(d3.svm.poly3$Cs)),rep('D4 (poly3)',length(d4.svm.poly3$Cs))))

ggplot(df.res.poly3, aes(x=k, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1") + scale_x_continuous(trans='log10') + geom_line() + theme_minimal()


#RBF
d1.svm.RBF.g05 <- optimize.C(dataset.train,     Cs, which.kernel="RBF", gamma=0.5)
d2.svm.RBF.g05 <- optimize.C(dataset.cat.train, Cs, which.kernel="RBF",gamma=0.5)
d3.svm.RBF.g05 <- optimize.C(d3.pcamca.train,   Cs, which.kernel="RBF",gamma=0.5)
d4.svm.RBF.g05 <- optimize.C(d4.mca.train,      Cs, which.kernel="RBF",gamma=0.5)

save(Cs,d1.svm.RBF.g05,d2.svm.RBF.g05,d2.svm.RBF.g05,d4.svm.RBF.g05, file = "tmp/svm-RFB-results-05.Rdata")
load("tmp/svm-RFB-results-05.Rdata")

gammas <- 2^seq(-3,4)
d1.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
d2.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
d3.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
d4.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
for (i in c(2,4))#Gamma
{
  print(paste("gamma ", gammas[i]))
  d1.svm.RBF.F1[,i] <- optimize.C(dataset.train,    Cs, which.kernel="RBF", gamma= gammas[i])
  d2.svm.RBF.F1[,i] <- optimize.C(dataset.cat.train,Cs, which.kernel="RBF", gamma=gammas[i])
  d3.svm.RBF.F1[,i] <- optimize.C(d3.pcamca.train,  Cs, which.kernel="RBF", gamma=gammas[i])
  d4.svm.RBF.F1[,i] <- optimize.C(d4.mca.train,     Cs, which.kernel="RBF", gamma=gammas[i])
}

d1.svm.RBF.F1[,3] <-d1.svm.RBF.g05$F1
d2.svm.RBF.F1[,3] <-d2.svm.RBF.g05$F1
d3.svm.RBF.F1[,3] <-d3.svm.RBF.g05$F1
d4.svm.RBF.F1[,3] <-d4.svm.RBF.g05$F1
# Plot
df.res.svm = data.frame()
for(i in 1:(length(Cs))){
  for(j in 1:(length(gammas))){
    df.tmp <- data.frame(C=rep(Cs[i],4),
                         gamma=rep(gammas[j],4),
                         F1=c(d1.svm.RBF.F1[i,j],  d2.svm.RBF.F1[i,j], d3.svm.RBF.F1[i,j], d4.svm.RBF.F1[i,j]), 
                         group=c('D1 (RBF)','D2 (RBF)','D3 (RBF)','D4 (RBF)'))
    df.res.svm <- rbind(df.res.svm,df.tmp)
  }
}
df.res.svm$C <- as.factor(df.res.svm$C)
df.res.svm$gamma <- as.factor(df.res.svm$gamma)

ggplot(data = df.res.svm[df.res.svm$group=='D1 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  geom_tile()
ggplot(data = df.res.svm[df.res.svm$group=='D2 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  geom_tile()
ggplot(data = df.res.svm[df.res.svm$group=='D3 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  geom_tile()
ggplot(data = df.res.svm[df.res.svm$group=='D4 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  geom_tile()
#ggplot(df.res.svm, aes(x=C, y=F1, group=group, color=group)) + 
#  scale_y_continuous(name = "F1") + scale_x_continuous(trans='log10') + geom_line() + theme_minimal()

####################################################################
# Tree
####################################################################

run.tree <- function (dataset)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- tree(y ~ ., weights= weights, data=train)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}
(d1.tree = run.tree(dataset.train))
(d2.tree = run.tree(dataset.cat.train))
(d3.tree = run.tree(d3.pcamca.train))
(d4.tree = run.tree(d4.mca.train))

####################################################################
# Random forest
####################################################################

run.randomForest <- function (dataset, ntree=100)
{
  createModelAndPredict <- function(train, newdata){
    class.sampsize <- min(table(train$y))
    model <- randomForest(y ~ ., data=train, ntree=ntree, proximity=FALSE, 
                          sampsize=c(yes=class.sampsize, no=class.sampsize), strata=train$y)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

#Run severl ntrees
optimize.ntrees <- function(dataset, ntrees=round(10^seq(1,2,by=0.2))){
  results <- list()
  for (i in 1:(length(ntrees)))
  { 
    print(paste("ntrees: ",ntrees[i]))
    results[[i]] <- run.randomForest(dataset,ntrees[i])
  }
  z = list(ntrees = ntrees)
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  max.idx <- which.max(z$F1)[1]
  z$max.ntrees <- z$ntrees[max.idx]
  z$max.F1 <- z$F1[max.idx]
  z
}
ntrees= round(10^seq(1,2,by=0.2))
(d1.randomForest = optimize.ntrees(dataset.train, ntrees))
(d2.randomForest = optimize.ntrees(dataset.cat.train, ntrees))
(d3.randomForest = optimize.ntrees(d3.pcamca.train, ntrees))
(d4.randomForest = optimize.ntrees(d4.mca.train, ntrees))


df <- data.frame(ntree=rep(ntrees,4),
                 F1=c(d1.randomForest$F1, d2.randomForest$F1, d3.randomForest$F1, d4.randomForest$F1), 
                 accuracy=c(d1.randomForest$accuracy, d2.randomForest$accuracy, d3.randomForest$accuracy, d4.randomForest$accuracy), 
                 group=c(rep('D1',length(ntrees)),rep('D2',length(ntrees)), rep('D3',length(ntrees)),rep('D4',length(ntrees))))

ggplot(df, aes(x=ntree, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1", limits = c(0.65, 0.75)) +
  scale_x_continuous(trans='log2') +
  geom_line() + geom_point()+theme_minimal()

