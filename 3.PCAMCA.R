####################################################################
# Course project - MIRI
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 3: 
# June 2019
####################################################################

rm(list = ls())

library(FactoMineR)
library(factoextra)

# Set environment
setwd(".")
par(mfrow=c(1,1))
###############
## Dataset 1 ##
###############

# Some explanation

#Load preprocessed data
load("bank-processed-train-test.Rdata")

#PCA #
dataset.train.numerical = dataset.train[,-c(2:8,10,12,18)]

pca1.result = prcomp(dataset.train.numerical)

fviz_eig(pca1.result, main="Scree plot - First PCA - D1")
fviz_pca_var(pca1.result, col.var = "contrib",  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE )

plot(pca1.result$x[,1],pca1.result$x[,2], col=dataset.train$y)
ggplot(data.frame(pca1.result$x[,1:2],y=dataset.train$y), aes(x=PC1, y=PC2, group=y, color=y)) +  
  geom_point(size = 0.1) +theme_minimal()

pca1.nd = 6

##### MCA ####

dataset.train.cat = dataset.train[,c(2:8,10,12)]

#Number of dimension
ncols <- ncol(dataset.train.cat)
nmod = 0
for (j in 1:ncols) {nmod = nmod + length(levels(dataset.train.cat[,j]))}
mca.max.nd = nmod - ncols

mca.res = MCA(dataset.train.cat,ncp=mca.max.nd)
fviz_screeplot(mca.res,ncp=mca.max.nd)

i <- 1
while (mca.res$eig[i,1] > mean(mca.res$eig[,1])) i <- i+1
(mca1.nd <- i-1)

fviz_mca_var(mca.res, choice="mca.cor")
fviz_mca_var(mca.res)
ggplot(data.frame(mca.res$ind$coord[,1:2],y=dataset.train$y), aes(x=Dim.1, y=Dim.2, group=y, color=y)) +  
  geom_point(size = 0.1) +theme_minimal()

#PCA+MCA - PCA again

join.dataset = data.frame(pca1.result$x[,1:pca1.nd], mca.res$ind$coord[,1:mca1.nd])

join.pca = prcomp(join.dataset)

fviz_eig(join.pca,ncp=ncol(join.dataset),main="Scree plot - PCA over (PCA+MCA) - D1")

plot(join.pca$x[,1],join.pca$x[,2], col=dataset.train$y)

join.pca.nd= 6

d3.pcamca.train = data.frame(join.pca$x[,1:join.pca.nd])
d3.pcamca.train$y = dataset.train$y

ggplot(d3.pcamca.train, aes(x=PC1, y=PC2, group=y, color=y)) +  geom_point(size = 0.1) +theme_minimal()

### Apply projections to Test dataset ###

#First PCA (to numerical)
test.pca1.coord = predict(pca1.result, dataset.test[,-c(2:8,10,12,18)])

# Apply MCA to categorical
#Function needed for computing projections (MCA)
fixMCADatasetForPrediction = function(data){
  niveau <- unlist(lapply(data,levels))
  for (i in 1:ncol(data)){
    if (sum(niveau %in% levels(data[, i])) != nlevels(data[, i]))
      levels(data[, i]) = paste(attributes(data)$names[i], levels(data[, i]), sep = "_")
  }
  data
}

test.mca1.coord = predict(mca.res, fixMCADatasetForPrediction(dataset.test[,c(2:8,10,12)]))$coord

# Join PCA+MCA and apply PCA projection
test.join.dataframe = data.frame(test.pca1.coord[,1:pca1.nd],test.mca1.coord[,1:mca1.nd])
tmp = predict(join.pca, test.join.dataframe)
d3.pcamca.test = data.frame(tmp[,1:join.pca.nd])
d3.pcamca.test$y = dataset.test$y

save(d3.pcamca.train, d3.pcamca.test, file = "D3.PCAMCA.dataset.Rdata")

################
## Dataset 2 ###
################

