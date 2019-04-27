####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 2: Split data into train and test
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

#Load preprocessed data
load("bank-processed.Rdata")

set.seed (104)

# Shuffle the data
shuffle <- function(data){
  data[sample(nrow(data)),]
}
dataset <- shuffle(dataset)

# Split data into train and test: stratified split (2/3 train, 1/3 test)
dataset.y.True <- dataset[dataset$y == TRUE,]
dataset.y.False <- dataset[dataset$y == FALSE,]

dataset.y.True.train.idx <- 1:floor(2/3*nrow(dataset.y.True)) # Take the first rows, no need to sample because the data has been shuffled before
dataset.y.False.train.idx <- 1:floor(2/3*nrow(dataset.y.False))

#Join both cases
dataset.train <- rbind(dataset.y.True[dataset.y.True.train.idx,],
                       dataset.y.False[dataset.y.False.train.idx,])
dataset.test <- rbind(dataset.y.True[-dataset.y.True.train.idx,],
                      dataset.y.False[-dataset.y.False.train.idx,])

#Save data
save(dataset.train, file = "bank-processed-train.Rdata")
save(dataset.test, file = "bank-processed-test.Rdata")
