######## Support Vector Machines ##################
# Date: 02-12-2019                                #
# Author: Trygvi Laksafoss TRLK                   #
# Description: This is an example script          #
# for support vector mashines using the built     #
# in Iris dataset                                 #
###################################################

library(dplyr)
library(caTools) 
library(e1071) #This is the oddly named package with the SVM funciton
library(plotrix)
library(scatterplot3d)

#Load Iris data
data("iris")
str(iris)

#  Plot data
my_cols <- c("#00AFBB", "#E7B800", "#FC4E07")  
pairs(iris[,1:4], pch = 22,  cex = 0.5,
      col = my_cols[iris$Species],
      lower.panel=NULL,main=c("blue=Setosa, Yellow=Versicolor, red=Virginica"))

#  We want only to know if it is a virginica or not, so we pool the other two species
iris$species2<-iris$Species=="virginica"
#iris$species2<-iris$Species=="virginica"
iris$species2<-as.factor(as.numeric(iris$species2))

#  We extract only species, sepal length and width in this example
iris_sub<-select(iris,species2, Sepal.Length, Petal.Width)

#  Split dataset into training and test sets
set.seed(1243) 
split = sample.split(iris_sub$species2, SplitRatio = 0.65) 

training_set = subset(iris_sub, split == TRUE) 
test_set = subset(iris_sub, split == FALSE) 

#  Feature scaling
#  obs: negative 1 indexing chooses all columns except the first
training_set[-1] = scale(training_set[-1]) 
test_set[-1] = scale(test_set[-1]) 

#  Plot training and test data (Expand to see full code)
{
par(mfrow=c(1,2))
plot(training_set$Sepal.Length,training_set$Petal.Width,col=training_set$species2,
     main = 'Training data',
     xlab = 'Sepal Length', ylab = 'Sepal Width',
     cex=1.5,pch = 19)
legend(1, -1, legend=c("Virginica", "Other"),
       col=c("red", "black"), pch=c(1,1), cex=0.8)

plot(test_set$Sepal.Length,test_set$Petal.Width,col=test_set$species2,
     main = 'Test data',
     xlab = 'Sepal Length', ylab = 'Sepal Width',
     cex=1.5,pch = 19)
legend(1, -1, legend=c("Virginica", "Other"),
       col=c("red", "black"), pch=c(1,1), cex=0.8)
}

#  Fitting linear SVM to training set. 
#  Try changing kernel between "linear", "polynomial" and "radial".
#  Try also changing the cost to see how the hyperplane changes.
classifier = svm(formula = species2 ~ ., 
                 data = training_set, 
                 kernel = 'linear',cost=10) 

classifier

#  Predicting the Test set results and create confusion matrix
y_pred = predict(classifier, newdata = test_set[-1]) 
y_pred_training = predict(classifier, newdata = training_set[-1]) 
cm = table(test_set[, 1], y_pred)
cm_training = table(training_set[, 1], y_pred_training)
rownames(cm) <- c("Actual 0","Actual 1")
colnames(cm) <- c("Predicted 0", "Predicted 1")
rownames(cm_training) <- c("Actual 0","Actual 1")
colnames(cm_training) <- c("Predicted 0", "Predicted 1")

#  Costum plot (Expand to see full code)
{
par(mfrow=c(1,2),mar = c(11, 4, 4, 4))
grid<-function(x, n = 75) {
   grange = apply(x, 2, range)
   x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
   x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
   expand.grid(X1 = x1, X2 = x2)
}
xgrid<-grid(training_set[-1])
colnames(xgrid)<-c("Sepal.Length","Petal.Width")
ygrid<-predict(classifier, xgrid)
plot(xgrid, col = c("darksalmon","darkslategray3")[as.numeric(ygrid)], pch = 19, cex = .6,main="SVM Training Data",
     xlab="Sepal Length (Scaled)",ylab="Petal Width (Scaled)")
points(training_set[-1],col=as.numeric(training_set[,1]), pch = 19,cex=1.5)
addtable2plot(-2 ,-3,cm_training,title="CM (Ttraining data)",display.colnames=TRUE,display.rownames=TRUE)

plot(xgrid, col = c("darksalmon","darkslategray3")[as.numeric(ygrid)], pch = 19, cex = .6,main="SVM Test Data",
     xlab="Sepal Length (Scaled)",ylab="Petal Width (Scaled)")
points(test_set[-1],col=as.numeric(test_set[,1]), pch = 19,cex=1.5)
addtable2plot(-2 ,-3,cm,title="CM (Test data)",display.colnames=TRUE,display.rownames=TRUE)
}





#Illustrative kernel using phi(a,b)=(a,b,a^2+b^2)
x<-training_set$Sepal.Length
y<-training_set$Petal.Width
z<-abs(training_set$Sepal.Length)^2+abs(training_set$Petal.Width)^2
colors <- c("#000000","#FF0000")
colors <- colors[as.numeric(training_set$species2)]

scatterplot3d(x, y, z,main="Kernel trick",
              xlab = "Sepal Length (cm)",
              ylab = "petal Width (cm)",
              zlab = "Sepal Length^2+petal Width^2",
              pch = 16, color=colors,
              angle = 210)
