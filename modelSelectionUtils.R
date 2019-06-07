#############################################
# Some functions usefull for the prediction #
#############################################

# Function that compute the accuracy given prediction and real values
compute.accuracy <- function (pred, real)
{
  ct <- table(Truth=real, Pred=pred)
  round(100*(1-sum(diag(ct))/sum(ct)),2)
}

# harmonic mean
harm <- function (a,b) { 2/(1/a+1/b) }

# Function that computes the F1 Score (performance mesure): the harmonic mean of precision and recall:
compute.F1 <- function (pred, real)
{
  ct <- table(Truth=real, Pred=pred)
  harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2])
}

# Function that runs a k-fold-CV using:
# - The generateModelAndPredict function creates the model and predict in each fold, 
# - The loss.func computes the goodness of the current fold 
run.k.fold.CV <- function(generateModelAndPredict, dataset, loss.func= compute.accuracy, k = 10){
  set.seed(1234)
  CV.folds <- generateCVRuns (dataset$y, ntimes=1, nfold=k, stratified=TRUE)
  acc = numeric(k)
  for (j in 1:k)
  {
    print(paste(c('Fold ',j)))
    va <- unlist(CV.folds[[1]][[j]])
    pred.va <- generateModelAndPredict(dataset[-va,], dataset[va,])
    acc[j]<-loss.func(pred.va, dataset[va,]$y)
  }
  return(acc)
}

# Function that computes the weights of each class
compute.weights = function(y){
  priors = table(y)/length(y)
  weights = numeric(length(y))
  weights[y=="yes"] = 1/priors["yes"]
  weights[y=="no"] = 1/priors["no"]
  weights
}

# Function that given a vector of probabilities, returns a vector of factors "no"/"yes"
probabilityToFactor <- function(v, P=.5){
  result = factor(levels = c("no","yes")) 
  result[v<P] <- "no"
  result[v>=P] <- "yes"
  result
}
