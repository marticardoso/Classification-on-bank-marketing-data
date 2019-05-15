####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 2: Feature selection
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

#Load preprocessed data
load("bank-processed.Rdata")
load("bank-processed-cat.Rdata")

require(FSelector)
require(mlbench)
require(MASS)
require(CORElearn)


#### Feature selection analysis
dataset.is.numeric <- unlist(lapply(names(dataset), function(col) is.numeric(dataset[,col])))
cont.vars <- which(dataset.is.numeric==TRUE)

# Feature selection for continuous variables using Fisher's F

pvalcon = NULL
for (i in 1:length(cont.vars)) 
  pvalcon[i] <- (oneway.test (dataset[,cont.vars[i]]~dataset$y))$p.value

pvalcon <- matrix(pvalcon)
row.names(pvalcon) <- names(dataset)[cont.vars]

as.matrix(sort(pvalcon[,1]))
#All p-values <0.5, so we should keep all continuous variables.

# Graphical representation of outpurt
ncon = nrow(pvalcon)
par (mfrow=c(2,4))
for (i in 1:nrow(pvalcon)) 
{
  barplot (tapply(dataset[,cont.vars[i]], dataset$y, function(x) mean(x, na.rm=TRUE)),main=paste("Means by",row.names(pvalcon)[i]), las=2, cex.names=1.25)
  abline (h=mean(dataset[,cont.vars[i]]))
  legend (0,mean(dataset[,cont.vars[i]]),"Global mean",bty="n") 
}

## Categorical variables

apply.chisq.test <- function(data){
  dataset.is.factor <- unlist(lapply(names(data), function(col) is.factor(data[,col])))
  cat.vars <- which(dataset.is.factor[-20]==TRUE)
  cat.pvalcon = NULL
  for (i in 1:(length(cat.vars)))
    cat.pvalcon[i] <- (chisq.test(table(data[,cat.vars[i]],data$y)))$p.value
  
  cat.pvalcon <- matrix(cat.pvalcon)
  row.names(cat.pvalcon) <- names(data)[cat.vars]
  
  as.matrix(sort(cat.pvalcon[,1]))
}
apply.chisq.test(dataset)
apply.chisq.test(dataset.cat)



## TO continue...


#############################################################
## Filters

# 1. Correlation filter
subset.CFS <- cfs (y~., dataset)
(subset.CFS.formula <- as.simple.formula(subset.CFS,"Target"))

# 2. Consistency filter

subset.Consistency <- consistency (y~., dataset)
(subset.Consistency.formula <- as.simple.formula(subset.Consistency,"Target"))

# 3. ReliefF
estReliefF <- attrEval(Target ~ ., dataset, estimator="RReliefFequalK", ReliefIterations=1000)

print(sort(estReliefF,decreasing=TRUE))

subset.relief <- sort(estReliefF,decreasing=TRUE)[1:5] # this is tricky, we should not know we must select the first 5!
subset.relief.formula <- as.simple.formula(names(subset.relief),"Target")

# remove unnecessary information
attr(subset.relief.formula,".Environment") <- NULL

subset.relief.formula


#############################################################
# Wrappers

# 4. Linear Regression (i.e., using a linear model)

# Using the coefficients for feature importance is meaningful only when the variables are standardized
# so we do it (we standardize all predictors and add Target back)

ind <- c(rep(TRUE,13+p),FALSE)
dataforFSS.std <- cbind(data.frame(lapply(dataforFSS[ind], scale)), Target=dataforFSS$Target)


LinearRegression <- glm (Target~., family = gaussian, data = dataforFSS.std)
subset.LinearRegression.formula <- step(LinearRegression)$formula

subset.LinearRegression.formula

# 5. Random Forest (i.e., using a non-linear model)

# We will cover Random Forests in the last lecture; for the moment we use them for FSS
weights.randomForest <- random.forest.importance (Target~., dataforFSS, importance.type = 1)
print(weights.randomForest)

subset.randomForest.formula <- as.simple.formula(cutoff.k(weights.randomForest, 5), "Target") # this is again tricky!

# remove unnecessary information
attr(subset.randomForest.formula,".Environment") <- NULL

subset.randomForest.formula

###############################
## Now for the final comparison

print(subset.randomForest.formula)
print(subset.LinearRegression.formula)
print(subset.relief.formula)
print(subset.Consistency.formula)
print(subset.CFS.formula)

## A good exercise is found in comparing the previous results in terms of the selected relevant, irrelevant 
## and redundant features; another consideration is training time. What do you think is the best method overall?


####################################################################
## Example 2: Feature extraction and selection for classification
####################################################################

## We recover from a previous Lab the problem of classifying wines

## We are given results of a chemical analysis of wines grown in a region in Italy but derived from three different cultivars.
## The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
## The goal is to separate the three types of wines

wine <- read.table("wine.data", sep=",", dec=".", header=FALSE)

dim(wine)

colnames(wine) <- c('Wine.type','Alcohol','Malic.acid','Ash','Alcalinity.of.ash','Magnesium','Total.phenols',
                    'Flavanoids','Nonflavanoid.phenols','Proanthocyanins','Color.intensity','Hue','OD280.OD315','Proline')

wine$Wine.type <- as.factor(wine$Wine.type)

summary(wine)

## The first thing we should do is to perform a basic (univariate) feature selection
## using a statistical test of association for continuous variables: Fisher's F

pvalcon <- NULL

varc <- names(wine)[-1] # all predictors, no target

for (i in 1:length(varc)) { pvalcon[i] <- oneway.test(as.simple.formula("Wine.type", varc[i]), data=wine)$p.value }

pvalcon <- matrix(pvalcon)
row.names(pvalcon) <- varc

## Ordered list according to their dependence to Wine.type
## (smaller p-value indicates stronger association)

sort(pvalcon[,1])

## Flavanoids is the best variable when taken in isolation
with(wine,plot(Wine.type,Flavanoids))

## Whereas Ash is the worst
with(wine,plot(Wine.type,Ash))

## Since all p-values are way lower than 0.05, all features are relevant
## this typically happens with statistical test of association

## We already reduced dimension with LDA, used for feature extraction (FDA):

lda.model <- lda (Wine.type ~ ., data = wine)

lda.model

## Plot the projected data in the first two LDs (the extracted features)
## We can see that the discrimination is very good

plot(lda.model, col=as.numeric(wine$Wine.type))

## We could easily create a classifier on this reduced 2D space

## For illustration purposes, we will do a different thing: reduce the original 13 variables
## (that would be feature selection) in wrapper mode, and then re-do LDA (feature extraction)

library(ipred) # yet another nice package for machine learning (kind of alternative to {CORElearn})

# Resampling control 1: TIMESxK-cross validation (stratified)

## note: 10x10CV with stratification is one of the most reliable resampling methods
K <- 10
TIMES <- 10

mycontrol <- control.errorest (k = K, strat = TRUE, random = FALSE, predictions = TRUE)

# Resampling control 2 : LOO-cross validation (cannot be stratified or iterated)

K <- nrow(wine)
TIMES <- 1

mycontrol <- control.errorest (k = K, strat = FALSE, random = FALSE, predictions = TRUE)

## here we go, we start with LDA
## for some classifiers (as LDA and QDA), we must force predict() to return class labels only
mypredict <- function(object, newdata)
  predict(object, newdata = newdata)$class


## {ipred} provides with the method errorest(), which we conveniently wrap to iterate it TIMES times

evaluator.accuracy <- function (subset) 
{
  cat(length(subset), subset)
  print(1 - mean(replicate(TIMES,errorest (as.simple.formula(subset, "Wine.type"), 
                                           data=wine, 
                                           model=mymethod, 
                                           estimator = "cv", 
                                           predict = mypredict, 
                                           est.para=mycontrol)$error)))
}

## Now we use some built-in FSelector search algorithms: 
# forward.search()
# backward.search() and 
# best.first.search()

## Making two packages work together is not easy, since they were not designed for this

## We are going to print the current subset size, the subset itself and its estimated resampled error;
## the search ends when no further improvement is possible

mymethod <- lda

# BACKWARD FSS

subset <- backward.search(names(wine)[-1], evaluator.accuracy)
f.lda1 <- as.simple.formula(subset, "Wine.type")
print(f.lda1)

# FORWARD FSS

subset <- forward.search(names(wine)[-1], evaluator.accuracy)
f.lda2 <- as.simple.formula(subset, "Wine.type")
print(f.lda2)

# As we can see, we obtain a very good performance with only 7 predictors (which include Ash!)

# BEST FIRST FSS

subset <- best.first.search(names(wine)[-1], evaluator.accuracy)
f.lda3 <- as.simple.formula(subset, "Wine.type")
print(f.lda3)

## Same solution as before, note that the best solution is not unique!!!
## In these cases, I recommend to deliver all of them to a domain expert and let her/him decide
## Since it is quite likely that other considerations apply: measurement cost, for example

## Given that SFS delivered better results, we stick to it and change now to QDA:

mymethod <- qda

# FORWARD FSS (SFS)

subset <- forward.search(names(wine)[-1], evaluator.accuracy)
f.qda <- as.simple.formula(subset, "Wine.type")
print(f.qda)

## We get a wonderful solution with only 6 predictors
## Alcohol Magnesium Flavanoids Color.intensity Hue Proline
## reaching 99.4% of accuracy with LOOCV

## We get similar results switching resampling to 10x10CV (but the process is slower)
## Note that for LDA, QDA efficient methods exist for getting the LOOCV
## (just set the CV flag to TRUE)

## It seems that, if we add Ash to it, we can reach 100% accuracy
## Obviously this is not true, but it is true that this model makes no mistakes
## in the training set using a safe resampling

## We should refit our model using the reduced set of variables and make it predict a test set

## We can also use the reduced set of variables to perform feature extraction on it
## but we should use the LDA model for this, not the QDA
## just do:

lda.model.reduced <- lda (f.lda2, data = wine)

par(mfrow=c(1,2))

plot(lda.model, col=as.numeric(wine$Wine.type), main="Original set")
plot(lda.model.reduced, col=as.numeric(wine$Wine.type), main="Reduced set")

## we get a very similar result (a little bit worse if you want), but with half the predictors


####################################################################
## Example 3: Feature extraction in action: PCA vs. FDA
####################################################################

## We are going to study the difference between PCA and FDA (LDA for feature extraction)

## The idea is best illustrated by designing two datasets: one for which PCA should perform better
## and another for which FDA should perform better

par(mfrow=c(1,1))

## First dataset should perform better with PCA, since the classes
## have the same mean and the discriminatory information is entirely due to the different covariances

N <- 1000

x1 <- rnorm(N)
y1 <- 4*rnorm(N)

x2 <- rnorm(N)
y2 <- rnorm(N)

a <- c(-10, 10)
b <- c(-10, 10)

plot(c(a, x1, x2), c(b, y1,y2), col=c(rep('white', 2),rep('green',N),rep('red',N)))

d1 <- data.frame(c(rep(1,N),rep(2,N)),c(x1,x2),c(y1,y2))

## Second dataset should perform better with FDA, since the classes
## are separable using the dimension exhibiting less variability

x3 <- 0.25*rnorm(N)+0.75
y3 <- 5*rnorm(N)

x4 <- 0.25*rnorm(N)-0.75
y4 <- 5*rnorm(N)

plot(c(a, x3, x4), c(b, y3,y4), col=c(rep('white', 2),rep('green',N),rep('red',N)))

d2 <- data.frame(c(rep(1,N),rep(2,N)),c(x3,x4),c(y3,y4))

## Let's check it! Compute PCA and FDA for both datasets:

PCA1 <- prcomp(d1[-1])
PCA2 <- prcomp(d2[-1])

LDA1 <- lda(d1[c(2,3)],d1[,1])
LDA2 <- lda(d2[c(2,3)],d2[,1])


## Now we need to project the datasets in the new coordinates
## We just preserve the first PC because otherwise the 
## representation would be equivalent to the original 2D data

d1PCA <- PCA1$x[,1]
d2PCA <- PCA2$x[,1]

## for FDA, this amounts to y = w^T x, as seen in the slides

d1LDA <- d1[,2] * LDA1$scaling[1] + d1[,3] * LDA1$scaling[2]
d2LDA <- d2[,2] * LDA2$scaling[1] + d2[,3] * LDA2$scaling[2]

## Now we project the new axes over the original
## representations. For that, we first need the slopes:

PCAslope1 <- PCA1$rotation[2,1]/PCA1$rotation[1,1]
PCAslope2 <- PCA2$rotation[2,1]/PCA2$rotation[1,1]
LDAslope1 <- LDA1$scaling[2]/LDA1$scaling[1]
LDAslope2 <- LDA2$scaling[2]/LDA2$scaling[1]


## And now we can perform the visualization:
## Note that we solid lines represent the projection directions (not the decision boundaries)

par(mfrow=c(2,2))

plot(c(a, x1, x2), c(b, y1,y2), col=c(rep('white', 2),rep('green',N),rep('red',N)), main="Dataset 1 using PCA")
abline(0,PCAslope1,col='black',lwd=2)

plot(c(a, x1, x2), c(b, y1,y2), col=c(rep('white', 2),rep('green',N),rep('red',N)), main="Dataset 1 using LDA")
abline(0,LDAslope1,col='black',lwd=2)

plot(c(a, x3, x4), c(b, y3,y4), col=c(rep('white', 2),rep('green',N),rep('red',N)), main="Dataset 2 using PCA")
abline(0,PCAslope2,col='black',lwd=2)

plot(c(a, x3, x4), c(b, y3,y4), col=c(rep('white', 2),rep('green',N),rep('red',N)), main="Dataset 2 using LDA")
abline(0,LDAslope2,col='black',lwd=2)

par(mfrow=c(1,1))







