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
load("bank-processed-cat.Rdata")

# First, we split the dataset, and finally we apply the same split to dataset.cat

set.seed (104)
# Shuffle the data
shuffle <- function(data){
  data[sample(nrow(data)),]
}
dataset <- shuffle(dataset)

# Split data into train and test: stratified split (2/3 train, 1/3 test)
dataset.y.Yes <- dataset[dataset$y == 'yes',]
dataset.y.No <- dataset[dataset$y == 'no',]

dataset.y.Yes.train.idx <- 1:floor(2/3*nrow(dataset.y.Yes)) # Take the first rows, no need to sample because the data has been shuffled before
dataset.y.No.train.idx <- 1:floor(2/3*nrow(dataset.y.No))

#Join both cases
dataset.train <- rbind(dataset.y.Yes[dataset.y.Yes.train.idx,],
                       dataset.y.No[dataset.y.No.train.idx,])
dataset.test <- rbind(dataset.y.Yes[-dataset.y.Yes.train.idx,],
                      dataset.y.No[-dataset.y.No.train.idx,])

#Save data
save(dataset.train, dataset.test, file = "bank-processed-train-test.Rdata")

# Apply same split (exact) to dataset.cat and save it
dataset.cat.train <- dataset.cat[row.names(dataset.train),]
dataset.cat.test <- dataset.cat[row.names(dataset.test),]
save(dataset.cat.train, dataset.cat.test, file = "bank-processed-cat-train-test.Rdata")
