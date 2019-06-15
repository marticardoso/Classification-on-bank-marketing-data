#Use default lambdas (glmnet)
get.default.lambdas <- function (dataset, alpha = 1, nlambda=25)
{
  x <- model.matrix(y~., dataset)[,-1]
  y <- ifelse(dataset$y == "yes", 1, 0)
  model <- glmnet(x, y, alpha = alpha,nlambda=nlambda, weights = compute.weights(dataset$y), family = "binomial")
  return(model$lambda)
}

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


## OPTIMIZE P
optimize.P <- function(dataset, Ps=seq(0.3,0.7,0.05)){
  results <- list()
  for (i in 1:(length(Ps)))
  { 
    print(paste("P: ",Ps[i]))
    results[[i]] <- run.logisticRegression(dataset,Ps[i])
  }
  z = list(Ps = Ps)
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  max.P.idx <- which.max(z$F1)[1]
  z$max.P <- z$ntrees[max.P.idx]
  z$max.F1 <- z$F1[max.P.idx]
  z
}
Ps=seq(0.35,0.65,0.05)
(logReg.d1 = optimize.P(dataset.train,Ps))
(logReg.d2 = optimize.P(dataset.cat.train,Ps))
(logReg.d3 = optimize.P(d3.pcamca.train,Ps))
(logReg.d4 = optimize.P(d4.mca.train,Ps))

df <- data.frame(Ps=rep(Ps,4),
                 F1=c(logReg.d1$F1, logReg.d2$F1, logReg.d3$F1, logReg.d4$F1), 
                 accuracy=c(logReg.d1$accuracy, logReg.d2$accuracy, logReg.d3$accuracy, logReg.d4$accuracy), 
                 group=c(rep('D1',length(Ps)),rep('D2',length(Ps)), rep('D3',length(Ps)),rep('D4',length(Ps))))

ggplot(df, aes(x=Ps, y=F1, group=group, color=group)) + 
  scale_y_continuous(name = "F1", limits = c(0.6, 0.75)) +
  scale_x_continuous(trans='log2') +
  geom_line() + geom_point()+theme_minimal()
