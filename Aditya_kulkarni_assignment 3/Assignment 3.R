library(caret)
library(ROCR)
library(e1071)
library(arules)
library(factoextra)
library(cluster)

# Read the ILPD dataset..
setwd("C:\\Users\\Pravin Kulkarni\\Desktop")
ilpd = read.csv("ILPD.csv")
set.seed(100)
#Split into 60-40 (train-test).  The dataset has 22 observations; 12 are 
# used for training and 8 for testing.
index <- sample(1:nrow(ilpd), size=0.4*nrow(ilpd))
test <- ilpd[index, ]
train <- ilpd[-index, ]
##################################################################################
# Problem 2.1
# Create the Naive bayes model.
model <- naiveBayes(as.factor(label) ~ tb + db + aap + alb + age , method="class", data=train)
pred <- predict(model, test, type="class")
table(pred)
#ROC curve and AUC
pred.rocrred <- predict(model, newdata=test, type="raw")[,2]
f.pred <- prediction(pred.rocrred, test$label)
f.perf <- performance(f.pred, "tpr", "fpr")
plot(f.perf, colorize=T, lwd=3)
abline(0,1)
auc <- performance(f.pred, measure = "auc")
auc@y.values[[1]]
# a) ROC curve plotted.
# b) The area under the curve is: 0.7354952
#   > The area under the curve for the best Decision tree model was 0.71.(The decision tree model in last lecture was calculated in wrong way. 
#       I reprogrammed it and got the pruned model to be better than normal model.)
#   > The accuracy of the naive has decreased but the AUC has increased slightly. As we measure the performance on auc, 
#       the naive bayes is better than decision tree
####################################################################################
####################################################################################
#Problem 2.2

setwd("C:\\Users\\Pravin Kulkarni\\Desktop")
groc <- read.transactions("groceries.csv", format = c("basket", "single"), sep = ",", rm.duplicates = F)
#trans <- as.matrix(read.csv("groceries.csv", header=F, sep=" ", comment.char = '#'))

rules <- apriori(groc, parameter = list(supp = 0.001))
groc <- as(groc, "matrix")
#rules <- apriori(groc, parameter = list(supp = 0.1, conf = 0.1, maxlen=3, originalSupport = F))

summary(rules)
# Prints 410 rules hence commented.
#inspect(rules)
#image(groc)

#inspect top and bottom rules sorted by support
top.support <- sort(rules, decreasing = T, by = "support", na.last = NA)
inspect(head(top.support,5))
top.support <- sort(rules, decreasing = F, by = "support", na.last = NA)
inspect(head(top.support,5))

#inspect top and bottom rules sorted by confidence
top.conf <- sort(rules, decreasing = T, by = "confidence", na.last = NA)
inspect(head(top.conf,5))
top.conf <- sort(rules, decreasing = F, by = "confidence", na.last = NA)
inspect(head(top.conf,5))

#inspect top and bottom rules sorted by lift
top.lift <- sort(rules, decreasing = T, by = "lift", na.last = NA)
inspect(head(top.lift,5))
top.lift <- sort(rules, decreasing = F, by = "lift", na.last = NA)
inspect(head(top.lift,5))

#abc <- as.matrix((itemFrequency(groc,type="absolute")))
#itemFrequencyPlot(groc,type="absolute")

# a) Most frequent Item - whole milk - Frequency - 2513
# b) Least Frequent Item - Baby Food - Frequency - 1
# c) Rules can be generated at various values of support & confidence maximum being 0.003 support at which 1 rule gets generated.
#     The support of 0.001 generates 410 rules, while support of 0.0001 makes me restart the pc.
#     Considering the support of 0.001

# Below answers will be different for different minsup and minconf. For below ans minsupp = 0.001 & minconf is default which is 0.8. This generates 410 rules.
# d) Top 5 rules sorted by support are:
#      i.  {citrus fruit, root vegetables,tropical fruit,whole milk} => {other vegetables} support = 0.0031
#     ii.  {curd, domestic eggs,other vegetables} => {whole milk}                          support = 0.0028
#    iii.  {curd, hamburger meat} => {whole milk}                                          support = 0.0025
#     iv.  {herbs, rolls/buns} => {whole milk}                                             support = 0.0024
#      v.  {herbs, tropical fruit} => {whole milk}                                         support = 0.0023

# e) Top 5 rules sorted by confidence 
#      i.  {rice,sugar}                                          => {whole milk}       confidence = 1
#     ii.  {canned fish,hygiene articles}                        => {whole milk}       confidence = 1         
#    iii.  {butter,rice,root vegetables}                         => {whole milk}       confidence = 1         
#     iv.  {flour,root vegetables,whipped/sour cream}            => {whole milk}       confidence = 1         
#      v.  {butter,domestic eggs,soft cheese}                    => {whole milk}       confidence = 1
#
# f) Top 5 rules sorted by lift
#      i.  {liquor,red/blush wine}        => {bottled beer}                                 lift = 11.235269
#     ii.  {citrus fruit,fruit/vegetable juice,other vegetables,soda} => {root vegetables}  lift = 8.340400
#    iii.  {oil,other vegetables,tropical fruit, whole milk,yogurt} => {root vegetables}    lift = 8.340400
#     iv.  {citrus fruit,fruit/vegetable juice,grapes} => {tropical fruit}                  lift = 8.063879
#      v.  {other vegetables,rice, whole milk,yogurt} => {root vegetables}                  lift = 7.951182


# g) Bottom 5 rules sorted by support
#     i. {cereals,curd}                                 => {whole milk}       support = 0.001016777
#    ii. {butter,jam}                                   => {whole milk}       support = 0.001016777
#   iii. {pastry,sweet spreads}                         => {whole milk}       support = 0.001016777
#    iv. {butter,rice,root vegetables}                  => {whole milk}       support = 0.001016777 
#     v. {other vegetables,rice,tropical fruit}         => {whole milk}       support = 0.001016777
#
#
# h) Bottom 5 rules sorted by confidence
#     i. {curd,turkey}                               => {other vegetables}    confidence = 0.8           
#    ii. {fruit/vegetable juice,herbs}               => {other vegetables}    confidence = 0.8           
#   iii. {herbs,rolls/buns}                          => {whole milk}          confidence = 0.8           
#    iv. {onions,waffles}                            => {other vegetables}    confidence = 0.8 
#     v. {root vegetables,tropical fruit,turkey}     => {other vegetables}    confidence = 0.8

# i) Bottom 5 rules sorted by list
#     i. {herbs,rolls/buns}                          => {whole milk} lift = 3.130919 
#    ii. {butter,soft cheese,yogurt}                 => {whole milk} lift = 3.130919    
#   iii. {frankfurter,frozen meals,other vegetables} => {whole milk} lift = 3.130919    
#    iv. {frozen meals,tropical fruit,yogurt}        => {whole milk} lift = 3.130919  
#     v. {curd,hamburger meat,other vegetables}      => {whole milk} lift = 3.130919

################################################################################
# Problem 3:
# a) DATA CLEAN UP:
#      i. For clustering first attribute needs to be removed which is the name of the species as it is not numeric.
#     ii. The data needs to be standardize as some attributes are from 0-1 range while some are from 0-8. 
#         Inorder to avoid the attribute with larger value be the only driving force.
#    iii. The cleaned file converted to csv is attached named - clustCleaned.csv.

# JUST FOR REFERENCE BOTH SCALED AND UNSCALED DATASETS ARE USED. ONLY CONSIDER THE OUTPUT OF SCALED.
setwd("C:\\Users\\Pravin Kulkarni\\Desktop")
clust = read.table("file19.txt", header=T, skip = 20)
# Write in csv format to remove extra spaces and make comma seprated.
write.csv(clust, "clustCleaned.csv", row.names = F)
rm(clust)
clust <- read.csv("clustCleaned.csv")
clustScaled <- scale(clust[2:9])

#Identifying optimal cluster.
fviz_nbclust(clustScaled, kmeans, method="silhouette", print.summary = T)
fviz_nbclust(clust[2:9], kmeans, method="silhouette", print.summary = T)

#View(clust)
k_clust_Scaled <- kmeans(clustScaled, centers=8, nstart=25)
k_clust <- kmeans(clust[2:9], centers=10, nstart=25)
#Creating the cluster analysis.
fviz_cluster(k_clust_Scaled, clustScaled)
fviz_cluster(k_clust, clust[2:9])

print(k_clust_Scaled)
print(k_clust_Scaled$withinss)
print(k_clust_Scaled$tot.withinss)

for(i in 1:8){
  print(which(k_clust_Scaled$cluster == i))
}
for(i in 1:10){
  print(which(k_clust$cluster == i))
}



# b. CLUSTERING
#   i. Via silhouette graph the cluster for scaled data, optimal number of clusters is 8
#  ii. Plotted.
# iii. Observations in each cluster 1-8 respectively : 9, 17, 2, 17, 9, 3, 1, 8
#  iv. Total sse of the clusters : 55.03614
#   v. SSE of each cluster is 13.174224 14.918373  2.223560  9.591579  6.337449  4.546254  0.000000  4.244696
#
