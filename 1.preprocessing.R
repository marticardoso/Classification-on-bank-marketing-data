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
dataset$y <- tmp;


# Check factors
dataset.is.numeric <- unlist(lapply(names(dataset), function(col) is.numeric(dataset[,col])))
names(dataset.is.numeric) <- names(dataset)
dataset.is.numeric

# Summary:
# 9 continuous variables
# 10 categorical variables
# 1 output variable


# inspect the first 4 instances
dataset[1:4,]


####################################################################
# Missing values
####################################################################

# When we have missing values in the categorical variables, 
# we have an implicit modality for them (called 'unknown'), so all missing will belong to this modality.
# (As there are a lot of instances with missings, e.g. 'default' has more than 8000 observations, 
# we cannot remove them, and beeing unknown could be important for the classification)
# So, we keep the missings in the categorical variables.

# Secondly, the continuous varaibles does not have missing.

# Set NA to pdays
# pdays has 999 values, the documentation says that 999 means client was not previously contacted.
# We decided to set this values to NA
hist(dataset$pdays)
dataset$pdays[dataset$pdays==999] <- NA
hist(dataset$pdays)
# But more than 30000 instances are NA, so we cannot remove them. 
# We decided to discretize this variable into an ordered varaible (this way we can keep the 999 meaning)

pdays.CAT <- cut(dataset$pdays, c(0,3,5,8, 15,30), right = FALSE, ordered_result = TRUE)
levels(pdays.CAT) <- c(levels(pdays.CAT), "Never contacted")
pdays.CAT[is.na(pdays.CAT)] <- "Never contacted"

# Append result to dataset
dataset$pdays.CAT <- pdays.CAT
summary(dataset$pdays.CAT)
 
####################################################################
# Graphical summary of variables
####################################################################

# Graphical summary for each varaible
# Continuous data:

par(mfrow=c(2,2))
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

#Could be interesting to take logs on some variables: campaign, previous

#Campaign
par(mfrow=c(1,2))
dataset$log.campaign <- log(dataset$campaign+1)
plotHistAndBoxplot("log.campaign")
# There is a little improvement, but not much, the distribution does not seems to be gaussian.

# previous
dataset$log.previous = log(dataset$previous+1)
plotHistAndBoxplot("log.previous")
dataset$log.previous <- NULL 

# For the 'previous' as it only has 7 different values, 
# we decided to discritize it into an ordered factor (and do not use the log)
dataset$previous.CAT = (cut(dataset$previous, c(0,1,2,4,10), labels=c("0","1","2-3",">4"),right = FALSE, ordered_result=TRUE))
plot(dataset$previous.CAT, main="previous.cat")

#Plot density
par(mfrow=c(3,3))
for (i in which(dataset.is.numeric)) 
{ plot(density(na.omit(dataset[, i])), xlab="", main=names(dataset)[i]) }

par(mfrow=c(2,3))
#Looking at the density functions, we don't have variables that looks like gaussian, they are irregular.
# The only variable that seems gaussian is 'age'.
# For the other variables that seems to be very irregular, we decided to discretize them:
#   campaign/log.campaign, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

#emp.var.rate
dataset$emp.var.rate.CAT <- cut(dataset$emp.var.rate, c(-4,-2.5,-1,0.5,2), labels=c("<-2.5","-2.5->-1","-1->0.5",">0.5"),right = FALSE, ordered_result=TRUE)
plot(dataset$emp.var.rate.CAT, main="emp.var.rate")

#cons.price.idx
dataset$cons.price.idx.CAT <- cut(dataset$cons.price.idx, c(92,92.75,93.25,93.75,94.25,95), labels=c("<92.75","92.75-93.25","93.25-93.75","93.75-94.25", ">94.25"),right = FALSE, ordered_result=TRUE)
plot(dataset$cons.price.idx.CAT, main="cons.price.idx")

#cons.conf.idx
dataset$cons.conf.idx.CAT <- cut(dataset$cons.conf.idx, c(-55,-48,-44,-39,-35,-20), labels=c("<-48","-48<-44","-44<-39","-39<-35", "-35<"),right = FALSE, ordered_result=TRUE)
plot(dataset$cons.conf.idx.CAT, main="cons.conf.idx")

#euribor3m
dataset$euribor3m.CAT <- cut(dataset$euribor3m, c(0, 1,2.5,4.5,6), labels=c("very low","low","high","very high"),right = FALSE, ordered_result=TRUE)
plot(dataset$euribor3m.CAT, main="euribor3m")

#nr.employed
dataset$nr.employed.CAT <- cut(dataset$nr.employed, c(4000, 5050,5150,5200,5300), labels=c("<5050","5050<5150","5150<5200",">5200"),right = FALSE, ordered_result=TRUE)
plot(dataset$nr.employed.CAT, main="nr.employed")

#campaign/log.campaign
dataset$campaign.CAT <- cut(dataset$campaign, c(1, 2, 4,11,100), labels=c("1","2-3", "4-10", ">11"), right = FALSE, ordered_result=TRUE)
plot(dataset$campaign.CAT, main="campaign")

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
plot(dataset$campaign, dataset$pdays)


##############################################
# STATISTICAL ANALYSIS
##############################################

summary(dataset) 

dataset.is.numeric <- unlist(lapply(names(dataset), function(col) is.numeric(dataset[,col])))

# basic summary statistics by the output variable

library(psych)
describeBy (dataset[,dataset.is.numeric], dataset$y)

####################################################################
# SAVE THE DATA SET
####################################################################

# Shuffle the data before saving
shuffle <- function(data){
  data[sample(nrow(data)),]
}
set.seed (104)

dataset <- shuffle(dataset)
dataset.source <- dataset

d1.cols = c("age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
             "log.campaign", "pdays.CAT", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
             "euribor3m", "nr.employed", "y")
# We are going to save two datasets:
# 1. Using all continues variables
# 2. Using all categorical variables

dataset <- dataset.source[,d1.cols]

# Save data set
save(dataset, file = "bank-processed.Rdata")

#DATASET 2
d2.cols = c("age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
             "campaign.CAT", "pdays.CAT", "previous.CAT", "poutcome", "emp.var.rate.CAT", "cons.price.idx.CAT", "cons.conf.idx.CAT",
             "euribor3m.CAT", "nr.employed.CAT", "y")
dataset.cat <- dataset.source[,d2.cols]

# Save data set
save(dataset.cat, file = "bank-processed-cat.Rdata")
