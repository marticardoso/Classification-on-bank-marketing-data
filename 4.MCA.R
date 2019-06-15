####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 4: MCA (Just for Categorical)
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

#Load preprocessed data
load("bank-processed-cat-train-test.Rdata")

## Since the data is categorical, we perform a Multiple Correspondence Analysis
# All variables are categorical, except age, so we discretize this variable
dataset.cat.train$age <- cut(dataset.cat.train$age, c(20,30,40,50,60,80), labels=c("<30","31<40","41<50","51<60",">61"),right = FALSE, ordered_result=TRUE)
plot(dataset.cat.train$age, main="nr.age")
dataset.cat.test$age <- cut(dataset.cat.test$age, c(20,30,40,50,60,80), labels=c("<30","31<40","41<50","51<60",">61"),right = FALSE, ordered_result=TRUE)


# Find number of dimension
ncols <- ncol(dataset.cat.train)
nmod = 0
for (j in 1:ncols) {nmod = nmod + length(levels(dataset.cat.train[,j]))}
mca.max.nd = nmod - ncols

#Perform mca
mca.res = MCA(dataset.cat.train[,-18],ncp=mca.max.nd)

#Dimensions to keep
fviz_screeplot(mca.res,ncp=mca.max.nd)
i <- 1
while (mca.res$eig[i,1] > mean(mca.res$eig[,1])) i <- i+1
(mca.nd <- i-1)

#Plot MCA
fviz_mca_var(mca.res, choice="mca.cor")
fviz_mca_var(mca.res)
ggplot(data.frame(mca.res$ind$coord[,1:2],y=dataset.cat.train$y), aes(x=Dim.1, y=Dim.2, group=y, color=y)) +  
  geom_point(size = 0.1) +theme_minimal()

# Create new dataframe (train)
d4.mca.train = data.frame(mca.res$ind$coord[,1:mca.nd])
d4.mca.train$y = dataset.cat.train$y

# To create the dataframe for test, we should apply the same projection than before
#Function needed for computing projections (MCA), it fixes a bug
fixMCADatasetForPrediction = function(data){
  niveau <- unlist(lapply(data,levels))
  for (i in 1:ncol(data)){
    if (sum(niveau %in% levels(data[, i])) != nlevels(data[, i]))
      levels(data[, i]) = paste(attributes(data)$names[i], levels(data[, i]), sep = "_")
  }
  data
}

# Create new dataframe (test)
d4.mca.test = predict(mca.res, fixMCADatasetForPrediction(dataset.cat.test[,-18]))$coord[,1:mca.nd]
d4.mca.test = data.frame(d4.mca.test)
d4.mca.test$y = dataset.cat.test$y

# Save data
save(d4.mca.train, d4.mca.test, file = "D4.MCA.dataset.Rdata")