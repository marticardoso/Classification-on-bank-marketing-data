

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

