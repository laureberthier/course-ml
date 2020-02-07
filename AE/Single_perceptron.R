######## Support Vector Machines ##################
# Date: 30-01-2019                                #
# Author: Trygvi Laksafoss TRLK                   #
# Description: This is an example script          #
# for a single neuron in a neuron networg based   #
# on the Iris dataset                             #
###################################################

library(dplyr)
library(caTools) 
library(e1071) #This is the oddly named package with the SVM funciton
library(plotrix)


mySigmoid <- function(x){
  y=1/(1+exp(-x))
  return(y)
}

mySigmoid_derivative<-function(x){
  y= x * ( 1 - x )
  return(y)
}

#Load Iris data
data("iris")
str(iris)

#  Plot data
my_cols <- c("#00AFBB", "#E7B800", "#FC4E07")  
pairs(iris[,1:4], pch = 22,  cex = 0.5,
      col = my_cols[iris$Species],
      lower.panel=NULL,main=c("blue=Setosa, Yellow=Versicolor, red=Virginica"))

#  We want only to know if it is a virginica or not, so we pool the other two species
#iris$species2<-iris$Species=="virginica"
iris$species2<-iris$Species=="versicolor"
iris$species2<-as.factor(as.numeric(iris$species2))
iris$Sepal.Length[iris$species2==0]=iris$Sepal.Length[iris$species2==0]+100

#  We extract only species, sepal length and width in this example
iris_sub<-select(iris,species2, Sepal.Length,Sepal.Width, Petal.Length,Petal.Width)

#  Split dataset into training and test sets
set.seed(1243) 
split = sample.split(iris_sub$species2, SplitRatio = 0.65) 

training_set = subset(iris_sub, split == TRUE) 
training_set<- training_set[sample(nrow(training_set)),]
test_set = subset(iris_sub, split == FALSE) 

#  Feature scaling
#  obs: negative 1 indexing chooses all columns except the first
training_set[-1] = scale(training_set[-1]) 
test_set[-1] = scale(test_set[-1]) 


#Define training input and output layers (We extract part of the data as example)
training_inputs<-as.matrix(training_set[30:45,2:5])
training_outputs<-as.numeric(training_set[30:45,1])-1



#Define initial synaptic weights
set.seed(1)
synaptic_weights<-runif(4, min=-1, max=1)

for (iteration in 1:10000){
  
  #Define input layer
  input_layer<- training_inputs
  
  #Take the dot product of synaptic weights and the input layer
  dop_product<-input_layer%*%synaptic_weights
  
  #Put the data through activation function (sigmoid)
  outputs<-mySigmoid(dop_product)
  
  #Calculate the error
  error<- training_outputs-outputs
  
  #Define the adjustments that should be made to the weights and update them
  adjustments<- error*mySigmoid_derivative(outputs)
  synaptic_weights<-synaptic_weights+(t(input_layer)%*%adjustments)
}


#Chech the predictions on training data compared to labels
(result<-data.frame(outputs,training_outputs))

