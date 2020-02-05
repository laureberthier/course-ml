
################################
# Linear Discriminant Analysis #
################################

library(MASS) # lda
library(ggplot2)
require(scales)
require(gridExtra)
library(datasets)
library(klaR) # partimat
# Loading the iris data:
data(iris)
summary(iris)


############################
#                          #
# Fitting LDA to iris data #
#                          #
############################

#
# LDA model with 2 classes and 2 covariates:
#

# We make an example of fitting a LDA to two groups of species. We start with only including two covariates.
# The two covariates we include are the Sepal Length and Petal length.

iris$type01 <- rep(0, length = dim(iris)[1])
# Species: virginica, versicolor, setosa
iris$type01[iris$Species == "virginica"] <- 1
iris$type01 <- as.factor(iris$type01)

(ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) + geom_point(aes(color = type01)))

mod <- lda(type01 ~ Sepal.Length + Sepal.Width, data = iris, prior = c(100/150,50/150))
mod

# Predictions:
modpr <- predict(mod)
head(modpr$class)
head(modpr$posterior)
head(modpr$x)

# Make density plot!
d <- data.frame(LD1 = modpr$x, class = iris$type01)
(ggplot(d, aes(x = LD1)) + geom_density(aes(fill = class, color = class), alpha = 0.4) + geom_point(y = 0, aes(color = class)))

table(iris$type01, modpr$class)

partimat(type01 ~ Sepal.Width + Sepal.Length, data = iris, method = "lda",
         prec = 150, gs = rep(c(4,5), table(iris$type01)), image.colors =  c("darksalmon","darkslategray3"))

#
# LDA model with 3 classes and 2 covariates:
#

(ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) + geom_point(aes(color = Species)))

mod2 <- lda(Species ~ Sepal.Length + Sepal.Width, data = iris)
mod2

# Predictions:
modpr2 <- predict(mod2)
table(iris$Species, modpr2$class)

###
## Plotting:
#

partimat(Species ~ Sepal.Width + Sepal.Length, data = iris, method = "lda",
         prec = 150, gs = rep(c(4,5,3), table(iris$Species)), image.colors =  c("darksalmon","darkslategray3","darkseagreen4"))


### BACK TO SLIDES ###


#############################################################################################################

#
# LDA model with 3 groups and 4 covariates: (Example dimensionality reduction!)
#

mod1 <- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris)
mod1

# Prediction of the classes: (Confusion matrix)
modpr <- predict(mod1)
table(iris$Species, modpr$class)

###
## Plotting options:
#

# Generic function:
library(klaR)
partimat(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris, method = "lda",
         prec = 150)

# Trygvis plotting function for classifiers:
mod.t <- modpr$x[,1:2]

# Plot of linear discriminant bounds:
modl1 <- lda(mod.t, iris$Species)
{
  par(mfrow=c(1,2),mar = c(11, 4, 4, 4))
  grid<-function(x, n = 75) {
    grange = apply(x, 2, range)
    x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
    x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
    expand.grid(X1 = x1, X2 = x2)
  }
  xgrid<-grid(mod.t)
  colnames(xgrid)<-c("LD1","LD2")
  ygrid <- predict(modl1, xgrid)
  plot(xgrid, col = c("darksalmon","darkslategray3","darkseagreen4")[as.numeric(ygrid$class)], pch = 19, cex = .6,main="LDA Training Data",
       xlab="LD1",ylab="LD2")
  points(mod.t,col = c("goldenrod1","blue","firebrick1")[as.numeric(iris[,5])], pch = 19,cex=1.5)
  
  #plot(xgrid, col = c("darksalmon","darkslategray3")[as.numeric(ygrid$class)], pch = 19, cex = .6,main="SVM Test Data",
  #     xlab="Sepal Length (Scaled)",ylab="Petal Width (Scaled)")
  #points(test_set[-1],col=as.numeric(test_set[,1]), pch = 19,cex=1.5)
  #addtable2plot(-2 ,-3.3,cm,title="CM (Test data)",display.colnames=TRUE,display.rownames=TRUE)
}


#############################################################################################################

#
# Dimensionality reduction compared to PCA:
#

# PCA model:
pca_mod <- prcomp(iris[,c(-5,-6)], center = TRUE, scale. = TRUE) 

# Variance explained
prop.pca <- pca_mod$sdev^2/sum(pca_mod$sdev^2)

# Fitting LDA model:
lda_mod <- lda(Species ~ ., iris[,-6], prior = c(1,1,1)/3)

prop.lda <- lda_mod$svd^2/sum(lda_mod$svd^2)

# Predictions:
pred_lda <- predict(object = lda_mod, newdata = iris)

dataset <- data.frame(species = iris[,"Species"], pca = pca_mod$x, lda = pred_lda$x)

p1 <- (ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = species, shape = species), size = 2.5)
       + xlab(paste("LD1 (", percent(prop.lda[1]), ")", sep="")) + ylab(paste("LD2 (", percent(prop.lda[2]), ")", sep=""))
       + xlim(c(-10,10)) + ylim(c(-5,5)) 
       + theme(legend.position = 'bottom', legend.direction = 'horizontal', legend.text = element_text(size=14),
               text = element_text(size=14)))

p2 <- (ggplot(dataset) + geom_point(aes(pca.PC1, pca.PC2, colour = species, shape = species), size = 2.5)
  + xlab(paste("PC1 (", percent(prop.pca[1]), ")", sep="")) + ylab(paste("PC2 (", percent(prop.pca[2]), ")", sep=""))
  + xlim(c(-3.5,3.5)) + ylim(c(-4,4)) 
  + theme(legend.position = 'bottom', legend.direction = 'horizontal', legend.text = element_text(size=14),
          text = element_text(size=14)))

grid.arrange(p2, p1, ncol = 2)

## BACK TO SLIDES ##

#############################################################################################################

#
# Quadratic discriminant analysis:
#

modq <- qda(Species ~ Sepal.Length + Sepal.Width, data = iris)

# Prediction of the classes: (Confusion matrix)
modqpr <- predict(modq)
table(iris$Species, modqpr$class)

###
## Plotting:
#

partimat(Species ~ Sepal.Width + Sepal.Length, data = iris, method = "qda",
         prec = 150, gs = rep(c(4,5,3), table(iris$Species)), image.colors =  c("darksalmon","darkslategray3","darkseagreen4"))

# Plotting the feature dimensions with decision boundaries:
{
  par(mfrow=c(1,2),mar = c(11, 4, 4, 4))
  grid<-function(x, n = 75) {
    grange = apply(x, 2, range)
    x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
    x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
    expand.grid(X1 = x1, X2 = x2)
  }
  xgrid<-grid(iris[,c("Sepal.Width","Sepal.Length")])
  colnames(xgrid)<-c("Sepal.Width","Sepal.Length")
  ygrid <- predict(modq, xgrid)
  plot(xgrid, col = c("darksalmon","darkslategray3","darkseagreen4")[as.numeric(ygrid$class)], pch = 19, cex = .8,main="LDA Training Data",
       xlab="Sepal.Width",ylab="Sepal.Length")
  points(iris[,c("Sepal.Width","Sepal.Length")], col = c("goldenrod1","blue","firebrick1")[as.numeric(iris[,5])], pch = 19,cex=1.3)
  
  #plot(xgrid, col = c("darksalmon","darkslategray3")[as.numeric(ygrid$class)], pch = 19, cex = .6,main="SVM Test Data",
  #     xlab="Sepal Length (Scaled)",ylab="Petal Width (Scaled)")
  #points(test_set[-1],col=as.numeric(test_set[,1]), pch = 19,cex=1.5)
  #addtable2plot(-2 ,-3.3,cm,title="CM (Test data)",display.colnames=TRUE,display.rownames=TRUE)
}

####
## 4 Covariates:

modq4 <- qda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris)

# Prediction of the classes: (Confusion matrix)
modqpr4 <- predict(modq4)
table(iris$Species, modqpr4$class)
## Not really better than the linear discriminant method in this case..

# The partimat in this case make two-by-two qda:
library(klaR)
partimat(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris, method = "qda",
         prec = 150)

##########################

###
## Harder example of qda:
#

iris$type01 <- rep(0, length = dim(iris)[1])
# Species: virginica, versicolor, setosa
iris$type01[iris$Species == "versicolor"] <- 1
iris$type01 <- as.factor(iris$type01)

(ggplot(iris, aes(x = Petal.Length, y = Sepal.Length)) + geom_point(aes(color = type01)))

modq2 <- qda(type01 ~ Petal.Length + Sepal.Length, data = iris)

# Prediction of the classes: (Confusion matrix)
modqpr2 <- predict(modq2)
table(iris$type01, modqpr2$class)

###
## Plotting:
#

# Generic function:
library(klaR)
partimat(type01 ~ Sepal.Length + Petal.Length, data = iris, method = "qda",
         prec = 150, gs = rep(c(4,5), table(iris$type01)), image.colors =  c("darksalmon","darkslategray3"))


# Plotting the feature dimensions with decision boundaries:
{
  par(mfrow=c(1,2),mar = c(11, 4, 4, 4))
  grid<-function(x, n = 75) {
    grange = apply(x, 2, range)
    x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
    x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
    expand.grid(X1 = x1, X2 = x2)
  }
  xgrid<-grid(iris[,c("Petal.Length","Sepal.Length")])
  colnames(xgrid)<-c("Petal.Length","Sepal.Length")
  ygrid <- predict(modq2, xgrid)
  plot(xgrid, col = c("darksalmon","darkslategray3","darkseagreen4")[as.numeric(ygrid$class)], pch = 19, cex = .8,main="QDA Training Data",
       xlab="Petal.Length",ylab="Sepal.Length")
  points(iris[,c("Petal.Length","Sepal.Length")], col = c("goldenrod1","blue","firebrick1")[as.numeric(iris[,6])], pch = 19,cex=1.3)
  
  #plot(xgrid, col = c("darksalmon","darkslategray3")[as.numeric(ygrid$class)], pch = 19, cex = .6,main="SVM Test Data",
  #     xlab="Sepal Length (Scaled)",ylab="Petal Width (Scaled)")
  #points(test_set[-1],col=as.numeric(test_set[,1]), pch = 19,cex=1.5)
  #addtable2plot(-2 ,-3.3,cm,title="CM (Test data)",display.colnames=TRUE,display.rownames=TRUE)
}


####
## LDA model:

modl2 <- lda(type01 ~ Petal.Length + Sepal.Length, data = iris)

# Prediction of the classes: (Confusion matrix)
modlpr2 <- predict(modl2)
table(iris$type01, modlpr2$class)

# Plotting the feature dimensions with decision boundaries:
{
  par(mfrow=c(1,2),mar = c(11, 4, 4, 4))
  grid<-function(x, n = 75) {
    grange = apply(x, 2, range)
    x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
    x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
    expand.grid(X1 = x1, X2 = x2)
  }
  xgrid<-grid(iris[,c("Petal.Length","Sepal.Length")])
  colnames(xgrid)<-c("Petal.Length","Sepal.Length")
  ygrid <- predict(modl2, xgrid)
  plot(xgrid, col = c("darksalmon","darkslategray3","darkseagreen4")[as.numeric(ygrid$class)], pch = 19, cex = .8,main="QDA Training Data",
       xlab="Petal.Length",ylab="Sepal.Length")
  points(iris[,c("Petal.Length","Sepal.Length")], col = c("goldenrod1","blue","firebrick1")[as.numeric(iris[,6])], pch = 19,cex=1.3)
  
  #plot(xgrid, col = c("darksalmon","darkslategray3")[as.numeric(ygrid$class)], pch = 19, cex = .6,main="SVM Test Data",
  #     xlab="Sepal Length (Scaled)",ylab="Petal Width (Scaled)")
  #points(test_set[-1],col=as.numeric(test_set[,1]), pch = 19,cex=1.5)
  #addtable2plot(-2 ,-3.3,cm,title="CM (Test data)",display.colnames=TRUE,display.rownames=TRUE)
}

####
## 4 Covariates:

modq24 <- qda(type01 ~ Petal.Length + Sepal.Length + Petal.Width + Sepal.Width, data = iris)

# Prediction of the classes: (Confusion matrix)
modqpr24 <- predict(modq24)
table(iris$type01, modqpr24$class)

### Same model for lda:
modl24 <- lda(type01 ~ Petal.Length + Sepal.Length + Petal.Width + Sepal.Width, data = iris)

# Prediction of the classes: (Confusion matrix)
modlpr24 <- predict(modl24)
table(iris$type01, modlpr24$class)
# QDA much better than the linear case still

# Plot of linear discriminant bounds:
mod24.t <- modlpr24$x[,1]

d <- data.frame(x = mod24.t, class = iris$type01)
(ggplot(d, aes(x = x)) + geom_density(aes(fill = class), alpha = 0.4) + geom_point(aes(color = class), y = 0))

# Notice that we cannot plot a similar plot for the quadratic discriminant analysis as it does not
# include a similar dimensionality reduction.


#####################################################################################################

#
# Simulated example:
#

# Simulate two classes/distributions of points:
n <- 200
c1 <- mvrnorm(n, mu = c(0,3), Sigma = matrix(c(4,4.5,4.5,6), ncol = 2))
c2 <- mvrnorm(n, mu = c(0,-3), Sigma = matrix(c(4,4.5,4.5,6), ncol = 2))

# Projected data:
projoptx <- rbind(c1,c2)%*%c(1,-1)/sqrt(2)

d <- data.frame(x1 = c(c1[,1],c2[,1]), x2 = c(c1[,2],c2[,2]), meanx = rep(c(mean(c1[,1]),mean(c2[,1])), each = n),
                meany = rep(c(mean(c1[,2]),mean(c2[,2])), each = n), y = rep(c("0","1"), each = n),
                projmx =  rep(0, n), projoptx = projoptx, projopty = -projoptx, 
                projomx = rep(c(mean(projoptx[1:200]),mean(projoptx[200:400])),each = n),
                projomy = rep(c(-mean(projoptx[1:200]),-mean(projoptx[200:400])),each = n))

ggplot(d, aes(x = x1, y = x2)) + xlab('x') + ylab('y') + theme(legend.position = '', legend.direction = 'vertical') + geom_point(aes(color = y), shape = "+", size = 2) + geom_point(aes(x = meanx, y = meany), size = 3) + xlim(c(-15,15)) + ylim(c(-15,15))
ggplot(d, aes(x = x1, y = x2)) + xlab('x') + ylab('y') + theme(legend.position = '', legend.direction = 'vertical') + geom_point(aes(color = y), shape = "+", size = 2) + geom_vline(xintercept = 0, linetype = 2) + geom_point(aes(x = meanx, y = meany), size = 3) + xlim(c(-15,15)) + ylim(c(-15,15))
ggplot(d, aes(x = projmx, y = x2)) + xlab('x') + ylab('y') + theme(legend.position = '', legend.direction = 'vertical') + geom_point(aes(color = y), shape = "+", size = 2) + geom_vline(xintercept = 0, linetype = 2) + geom_point(aes(x = projmx, y = meany), size = 3) 

ggplot(d, aes(x = x1, y = x2)) + xlab('x') + ylab('y') + theme(legend.position = '', legend.direction = 'vertical') + geom_point(aes(color = y), shape = "+", size = 2) + geom_abline(intercept = 0, slope = -1, linetype = 2) + geom_point(aes(x = meanx, y = meany), size = 3) + xlim(c(-15,15)) + ylim(c(-15,15))
ggplot(d, aes(x = projoptx, y = projopty)) + xlab('x') + ylab('y') + theme(legend.position = '', legend.direction = 'vertical') + geom_point(aes(color = y), shape = "+", size = 2) + geom_abline(intercept = 0, slope = -1, linetype = 2) + geom_point(aes(x = projomx, y = projomy), size = 3) + xlim(c(-15,15)) + ylim(c(-15,15))

d_long <- data.frame(x =c(d$projmx,d$x1), y = c(d$x2,d$x2), class = c(d$y,d$y), type = rep(c("raw","proj"), each = 2*n),
                     mx = c(d$projmx,d$projmx), my = c(d$meany,d$meany))
d_long$class <- as.factor(d_long$class)

library(gganimate)

p1 <- (ggplot(d_long, aes(x = x, y = y)) + geom_point(aes(color = class), size = 2, shape = "+") + geom_point(aes(x = mx, y = my), size = 3)
       + geom_vline(xintercept = 0, linetype = 2) + xlim(c(-15,15)) + ylim(c(-15,15)) + theme(legend.position = '', legend.direction = 'vertical'))
                     
p2 <- p1 + transition_states(type, wrap = FALSE) #+ shadow_trail()
p2

#anim <- animate(p2)
#anim_save("bad_proj.gif", anim, ani.width = 1000, ani.height = 1000)

d_long2 <- data.frame(x = c(d$projoptx,d$x1), y = c(d$projopty,d$x2), class = c(d$y,d$y), type = rep(c("raw","proj"), each = 2*n),
                      mx = c(d$projomx,d$projmx), my = c(d$projomy,d$meany))
d_long2$class <- as.factor(d_long$class)

p21 <- (ggplot(d_long2, aes(x = x, y = y)) + geom_point(aes(color = class), size = 2, shape = "+") + geom_point(aes(x = mx, y = my), size = 3)
       + geom_abline(intercept = 0, slope = -1, linetype = 2) + xlim(c(-15,15)) + ylim(c(-15,15)) + theme(legend.position = '', legend.direction = 'vertical'))

p22 <- p21 + transition_states(type, wrap = FALSE) #+ shadow_trail()
p22

#anim <- animate(p22)
#anim_save("good_proj.gif", anim, ani.width = 1000, ani.height = 1000)

#############################################################################################################
