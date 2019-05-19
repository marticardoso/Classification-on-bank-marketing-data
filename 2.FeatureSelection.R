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


# Feature selection analysis -------------------------------------------------------

# First we apply a Fisher's F test to the continuous variables
# (we are going to use the dataset with continuous vars)

dataset.is.numeric <- unlist(lapply(names(dataset), function(col) is.numeric(dataset[,col])))
cont.vars <- which(dataset.is.numeric==TRUE)

pvalcon = NULL
for (i in 1:length(cont.vars)) 
  pvalcon[i] <- (oneway.test (dataset[,cont.vars[i]]~dataset$y))$p.value

pvalcon <- matrix(pvalcon)
row.names(pvalcon) <- names(dataset)[cont.vars]

as.matrix(sort(pvalcon[,1]))
#All p-values < 0.5, so we should keep all continuous variables.

# Graphical representation of outpurt
ncon = nrow(pvalcon)
par (mfrow=c(2,4))
for (i in 1:nrow(pvalcon)) 
{
  barplot (tapply(dataset[,cont.vars[i]], dataset$y, function(x) mean(x, na.rm=TRUE)),main=paste("Means by",row.names(pvalcon)[i]), las=2, cex.names=1.25)
  abline (h=mean(dataset[,cont.vars[i]]))
  legend (0,mean(dataset[,cont.vars[i]]),"Global mean",bty="n") 
}

# Secondly, we apply a chisq test to the categorical variables

apply.chisq.test <- function(data, target){
  dataset.is.factor <- unlist(lapply(names(data), function(col) is.factor(data[,col])))
  cat.vars <- which(dataset.is.factor==TRUE)
  pval = NULL
  for (i in 1:(length(cat.vars)))
    pval[i] <- (chisq.test(table(data[,cat.vars[i]],target)))$p.value
  
  pval <- matrix(pval)
  row.names(pval) <- names(data)[cat.vars]
  
  as.matrix(sort(pval[,1]))
}
apply.chisq.test(dataset[,-20], dataset$y)
apply.chisq.test(dataset.v2[,-20], dataset$y)

# All variables except 'loan' and 'housing' have p-value < 0.05, so, we should keep them
# But for loan and housing, we get p-value > 0.05, 
# we cannot reject that the output and these variables are independents (no relation), 
# so we remove them from the dataset

dataset$loan <- NULL
dataset$housing <- NULL
dataset.v2$loan <- NULL
dataset.v2$housing <- NULL

# Save datasets -----------------------------------------------------------

save(dataset, file = "bank-processed-2.Rdata")
save(dataset.v2, file = "bank-processed-cat-2.Rdata")








