####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 1: Preprocessing
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

library(MASS)

####################################################################
# READING CSV 
####################################################################

dataset <- read.csv("./dataset/bank-additional/bank-additional-full.csv", header = TRUE, sep= ";", dec = ".", check.names=TRUE)

# dataset dimensions 
dim(dataset)

# Column names
names(dataset)


####################################################################
# BASIC INSPECTION OF THE DATASET
####################################################################

#Summary
summary(dataset)

# First of all, we remove the duration variable because the duration is not known before a call is performed
# so, it cannot be used for the predictive model
dataset$duration <- NULL

# Transform y to logical
tmp <- as.logical(dataset$y)
tmp[dataset$y == "yes"] <- TRUE
tmp[dataset$y == "no"] <- FALSE
dataset$y <- as.factor(tmp);

# Set NA
# pdays has 999 values, the documentation says that 999 means client was not previously contacted.
# We decided to set this values to NA
dataset$pdays[dataset$pdays==999] <- NA

# Check factors
dataset.is.factor <- unlist(lapply(names(dataset), function(col) is.factor(dataset[,col])))
names(dataset.is.factor) <- names(dataset)
dataset.is.factor

# Summary:
# 9 continuous variables
# 10 categorical variables
# 1 output variable


# inspect the first 4 instances
dataset[1:4,]


####################################################################
# MISSING VALUES
####################################################################

# When we have missing values in the categorical variables, 
# we have an implicit modality for them (called 'unknown'), so all missing will belong to this modality.
# (As there are a lot of instances with missings, e.g. default has more than 8000 instances, 
# we cannot remove them, and beeing unknown could be important for the classification)
# So, we keep the missings in the categorical variables.

# Also there is one continuous variable with missings: pdays
# But more than 30000 instances are NA, so we cannot remove them.
summary(dataset$pdays)

attach(dataset)
####################################################################
# GAUSSIANITY AND TRANSFORMATIONS
####################################################################

# Graphical summary for each varaible
# Continuous data:

par(mfrow=c(1,2))
plotHistAndBoxplot <- function(col){
  hist(dataset[,col],main=paste("Histogram of ", col))
  boxplot(dataset[,col], main=paste("Boxplot of ", col))
}

plotHistAndBoxplot("age")
plotHistAndBoxplot("campaign")
plotHistAndBoxplot("pdays")
plotHistAndBoxplot("previous")
plotHistAndBoxplot("emp.var.rate")
plotHistAndBoxplot("cons.price.idx")
plotHistAndBoxplot("cons.conf.idx")
plotHistAndBoxplot("euribor3m")
plotHistAndBoxplot("nr.employed")

#Plot density
par(mfrow=c(3,3))
for (i in which(dataset.is.factor==FALSE)) 
{ plot(density(na.omit(dataset[, i])), xlab="", main=names(dataset)[i]) }

par(mfrow=c(1,2))
#Could be interesting to take logs on some variables: campaign, pdays, previous
dataset$log.campaign = log(dataset$campaign+1)
plotHistAndBoxplot("log.campaign")

# Pdays
dataset$log.pdays = log(dataset$pdays+1)
plotHistAndBoxplot("log.pdays")
# usefull transformation

# Pdays
dataset$log.previous = log(dataset$previous+1)
plotHistAndBoxplot("log.previous")
dataset$log.previous <- NULL # Not works as expected


# Categorical variables:
par(mfrow=c(1,1))
show.barplot <- function(col){
  t <- table(dataset[,col])
  barplot(prop.table(t), main=paste('Barplot of ', col), ylab="proportion", xlab='levels')
  t
}
show.barplot("job")
show.barplot("marital")
show.barplot("education")
show.barplot("default")
show.barplot("housing")
show.barplot("loan")
show.barplot("contact")
show.barplot("month")
show.barplot("day_of_week")
show.barplot("poutcome")
show.barplot("y")


# Plot pairs of variables
plot(dataset$age, dataset$campaign)
pairs(~ dataset$age + dataset$campaign)


##############################################
# STATISTICAL ANALYSIS
##############################################

summary(dataset) 

dataset.is.factor <- unlist(lapply(names(dataset), function(col) is.factor(dataset[,col])))

# basic summary statistics by the output variable

library(psych)
describeBy (dataset[,dataset.is.factor==FALSE], dataset$y)


#### Feature selection analysis

cont.vars = which(dataset.is.factor==FALSE)

par (mfrow=c(3,4))
for (i in 1:length(cont.vars)){
  qqnorm(dataset[,cont.vars[i]], main=names(cont.vars)[i])
}

# Feature selection for continuous variables using Fisher's F

pvalcon = NULL
for (i in 1:length(cont.vars)) 
  pvalcon[i] = (oneway.test (dataset[,cont.vars[i]]~dataset$y))$p.value

pvalcon = matrix(pvalcon)
row.names(pvalcon) = names(dataset)[cont.vars]

as.matrix(sort(pvalcon[,1]))

# Graphical representation of outpurt

ncon = nrow(pvalcon)

par (mfrow=c(3,4))

for (i in 1:nrow(pvalcon)) 
{
  barplot (tapply(dataset[,cont.vars[i]], dataset$y, function(x) mean(x, na.rm=TRUE)),main=paste("Means by",row.names(pvalcon)[i]), las=2, cex.names=1.25)
  abline (h=mean(dataset[,cont.vars[i]]))
  legend (0,mean(dataset[,cont.vars[i]]),"Global mean",bty="n") 
}


####################################################################
# SAVE THE DATA SET
####################################################################

# Shuffle the data before saving
shuffle <- function(data){
  data[sample(nrow(data)),]
}
set.seed (104)
dataset <- shuffle(dataset)

# Save data set

save(dataset, file = "bank-processed.Rdata")

