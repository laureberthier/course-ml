## Principal Component Analysis (PCA)
## LAEB


## Load library 
library(ggplot2) # for being able to load factoextra
library(factoextra) # for visualizing the PCA's reuslts
library(dplyr)
library(krb5)
library(odbc)
library(DBI)

## Clean environment 
rm(list=ls())

## Import dataset 
cars04  <- readRDS(file = '~/birwe_data/Data/playground/prepared-zone/methods-and-libraries/ml-reading-course/cars04.RData')

con <- DBI::dbConnect(odbc::odbc(), "RWE")
krb5::kinit(user="laeb", realm="HLU.CORP.LUNDBECK.COM", keytab='~/.keytab/.t.keytab', cache=NULL)
con_rwe <- odbc::dbConnect(odbc::odbc(), "SQLServer RWE;database=RWE")



## Look at dataset
# Documentation about the dataset here: http://jse.amstat.org/datasets/04cars.txt
# Var 1:8 -- Name of the car, Var 8:18 -- Initial Features 
str(cars04)
summary(cars04)
head(cars04)

## Perform a pca
# Two ways of doing a pca
# 1. Spectral decompostion which examines the covariance between features  (princomp)
# 2. Singular Value decomposition which examines covariance between individuals (prcomp) 
# Singular Value Decomposition is supposed to have a better numerical accuracy compared to Spectral decomposition

## Using prcomp 
cars04.pca = prcomp(cars04[,8:18], center = TRUE, scale.=TRUE) # scale is True to standardize the features before pca

## PCA object 
str(cars04.pca)

## PC info 
# Center point, scaling of initial features  
cars04.pca$center
cars04.pca$scale
# Standard deviation of each PC
cars04.pca$sdev # i.e. square root of the eigenvalue of the covariance matrix 
# Eigenvalues of each PC i.e. variance retained 
eig <- (cars04.pca$sdev)^2
eig
# Variance in percentage i.e. proportion of variance (also given in summary)
variance <- eig*100/sum(eig)
variance
# Cumulative variances in percentage (also given in the summary)
cumvar <- cumsum(variance)
cumvar
# Data frame with eigenvalues, variance, and cumulative variance
eig.cars04 <- data.frame(eig = eig, variance = variance,
                                    cumvariance = cumvar)
eig.cars04
# 11 PC PC1-11 each of which explains a certain percentage of the total variance in the dataset
# PC1 explains nearly 65% of it, PC2 explains 17% of it etc. By knowing PC1 and PC2 
# you can explain 82% of the variance of the dataset 
# Note that standard deviation, variance and cumulative variance can also be accessed from the summary 
summary(cars04.pca)
# or can be accessed using the factoextra library 
eig.val <- get_eigenvalue(cars04.pca)
head(eig.val)

## Scree Plot 
# The importance of PC can be visualized using a screeplot  i.e. bar plot of the variance for each dimension/PC with added connecting lines
# Using Base R 
barplot(eig.cars04[, 2], names.arg=1:nrow(eig.cars04), 
        main = "Variances",
        xlab = "Principal Components",
        ylab = "Percentage of variances",
        col ="steelblue")
# Add connected line segments to the plot
lines(x = 1:nrow(eig.cars04), 
      eig.cars04[, 2], 
      type="b", pch=19, col = "red")
# Using factoextra library 
fviz_eig(cars04.pca)
# Also possible to visualize the eigenvalues instead of the explained variance using factoextra 
fviz_screeplot(cars04.pca, ncp=10, choice="eigenvalue")


## Determine the number of dimensions/PC to retain 
# Kaiser Criterion: Keep PC with eigenvalue > 1 as this indicates that PCs account for more variance than accounted by one of the original variables in standardized data. 
# This is commonly used as a cutoff point for which PCs are retained.
# Other Criterion: elbow shape 
# Criterion on the cumulative variance: limit the number of component to that number that accounts for a certain fraction of the total variance e.g. 80%
# In our case using Kaiser Criterion: we keep the first 2 PCs


## Loading or weight matrix
# Matrix of the eigenvectors of the covariance matrix expressed in terms of initial features
# Correlation/anti correlation between initial features and PC
round(cars04.pca$rotation[,],2)
# PC1 : all initial features have a negative projection on PC1 except gas mileage
# The first principal component tells us about whether we are getting a big, expensive gas-guzzling car with a powerful
# engine, or whether we are getting a small, cheap, fuel-efficient car with a wimpy engine.
# Engine size and gas. size vs. fuel efficiency 
# PC2: mileage hardly project on to it at all. Instead we have a contrast between the
# physical size of the car (positive projection) and the price and horsepower. Basi-
# cally, this axis separates mini-vans, trucks and SUVs (big, not so expensive, not
# so much horse-power) from sports-cars (small, expensive, lots of horse-power). sporty vs boxy 


## Graph of variables: the correlation Circle
# The correlation between variables and principal components is used as coordinates. It can be calculated as follow :
# Variable correlations with PCs = loadings * the component standard deviations. 
# Helper function : 
# Correlation between variables and principal components
var_cor_func <- function(var.loadings, comp.sdev){
  var.loadings*comp.sdev
}
# Variable correlation/coordinates
loadings <- cars04.pca$rotation
sdev <- cars04.pca$sdev
var.coord <- var.cor <- t(apply(loadings, 1, var_cor_func, sdev))
head(var.coord[, 1:4])
# Graph of variables using Base R 
# Plot the correlation circle
a <- seq(0, 2*pi, length = 100)
plot( cos(a), sin(a), type = 'l', col="gray",
      xlab = "PC1",  ylab = "PC2")
# Plot axis
abline(h = 0, v = 0, lty = 2)
# Add active variables
arrows(0, 0, var.coord[, 1], var.coord[, 2], 
       length = 0.1, angle = 15, code = 2)
# Add labels
text(var.coord, labels=rownames(var.coord), cex = 1, adj=1)
# Using factoextra package 
fviz_pca_var(cars04.pca)
# Interpretation of the Correlation Circle 
# The graph of variables shows the relationships between all variables :
# Positively correlated variables are grouped together.
# Negatively correlated variables are positioned on opposite sides of the plot origin (opposed quadrants). 
# The distance between variables and the origine measures the quality of the variables on the factor map. Variables that are away from the origin are well represented on the factor map.
# Plots confirms our intuition about PC1 and PC2 i.e. PC = size vs. fuel efficiency, PC2 =  sporty vs boxy 


## Quality of representation of the variables on the factor map 
# Cos2 : quality of representation for variables on the factor map
# The cos2 of variables are calculated as the squared coordinates : var.cos2 = var.coord * var.coord 
var.cos2 <- var.coord^2
head(var.cos2[, 1:4])
# Using factoextra package
fviz_pca_var(cars04.pca, col.var="contrib")+
scale_color_gradient2(low="white", mid="blue", 
                      high="red", midpoint=8) + theme_minimal()

## Contribution of variables to PC 
# The contribution of a variable to a given principal component is (in percentage) : (var.cos2 * 100) / (total cos2 of the component)
# For PC2 
comp.cos2 <- apply(var.cos2, 2, sum)
comp.cos2
contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}

var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))
head(var.contrib[, 1:4])
# Highlight the most contributing variables to a PC  
fviz_pca_var(cars04.pca, col.var="contrib") +
scale_color_gradient2(low="white", mid="blue", 
                      high="red", midpoint=9) + theme_minimal()


## Graph of individuals 
# Coordinates of individuals on the PC 
ind.coord <- cars04.pca$x
head(ind.coord[, 1:4])

## Quality of representation for individuals on the principal components 
# To calculate the cos2 of individuals, 2 simple steps are required :
# 1. Calculate the square distance between each individual and the PCA center of gravity
# d2 = [(var1_ind_i - mean_var1)/sd_var1]^2 + …+ [(var10_ind_i - mean_var10)/sd_var10]^2 + …+..
# Calculate the cos2 = ind.coord^2/d2
# Compute the square of the distance between an individual and the
# center of gravity
center <- cars04.pca$center
scale<- cars04.pca$scale
getdistance <- function(ind_row, center, scale){
  return(sum(((ind_row-center)/scale)^2))
}
d2 <- apply(eig.cars04,1,getdistance, center, scale)
# Compute the cos2
cos2 <- function(ind.coord, d2){return(ind.coord^2/d2)}
ind.cos2 <- apply(ind.coord, 2, cos2, d2)
head(ind.cos2[1:5, ])


## Contribution of individuals to the princial components
# The contribution of individuals (in percentage) to the principal components can be computed as follow :
#  100 * (1 / number_of_individuals)*(ind.coord^2 / comp_sdev^2)
# Contributions of individuals
contrib <- function(ind.coord, comp.sdev, n.ind){
  100*(1/n.ind)*ind.coord^2/comp.sdev^2
}

ind.contrib <- t(apply(ind.coord,1, contrib, 
                       cars04.pca$sdev, nrow(ind.coord)))
head(ind.contrib[, 1:4])

## Individual Graph using Base R 
plot(ind.coord[,1], ind.coord[,2], pch = 19,  
     xlab="PC1",ylab="PC2")
abline(h=0, v=0, lty = 2)
text(ind.coord[,1], ind.coord[,2], labels=rownames(ind.coord),
     cex=0.7, pos = 3)
## Using factoextra package 
fviz_pca_ind(cars04.pca)
fviz_pca_ind(cars04.pca, col.ind="cos2") +
  scale_color_gradient2(low="white", mid="blue", 
                        high="red", midpoint=0.50) + theme_minimal()

## Biplot of individuals and variables using Base R
biplot(cars04.pca,cex=0.4)
## Biplot using the factoextra package 
fviz_pca_biplot(cars04.pca,  geom = "text") +
  theme_minimal()


