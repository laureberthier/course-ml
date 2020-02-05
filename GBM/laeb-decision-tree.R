## LAEB decision Tree 

# Good website: http://uc-r.github.io/regression_trees

## Load packages 
library(dplyr)

## Import dataset 
cars04  <- readRDS(file = 
                     '~/birwe_data/Data/playground/prepared-zone/methods-and-libraries/ml-reading-course/cars04.RData')
## Look at dataset
str(cars04)
summary(cars04)
head(cars04)

## Splitting the data into a train and test set 
# Load package
library(caret) # training/test sets
# Set seed
set.seed(123)
# Do the split
cars04.sub <- cars04  %>% 
  select(-c("SUV","Wagon","Minivan","Pickup","AWD","RWD"))
training.sample <- cars04.sub$Sports %>% createDataPartition(p = 0.8, list = FALSE)
cars04.train.data  <- cars04.sub[training.sample, ]
cars04.test.data <- cars04.sub[-training.sample, ]
# Check rows of the train and test datasets
nrow(cars04.train.data)
nrow(cars04.test.data)
# First rows of the train and test datasets 
head(cars04.train.data)
head(cars04.test.data)


## Decision Tree with rpart package  

## Load packages 
library(rpart) # preforming decision trees
library(rpart.plot) # visualization decision trees 
library(rattle) # Visualization decision trees 

## Methodology 
# There are many methodologies for constructing regression trees but one of the oldest is known as the classification and regression tree (CART) 
# approach developed by Breiman et al. (1984).  
# Basic regression trees partition a data set into smaller subgroups and then fit a simple constant for each observation in the subgroup. 
# The partitioning is achieved by successive binary partitions (aka recursive partitioning) based on the different predictors. 
# The constant to predict is based on the average response values for all observations that fall in that subgroup (if regression) or majority vote (clasification).

## Regression tree 
reg.tree.fit <- rpart( 
                  formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
                  method ="anova", 
                  data = cars04.train.data)
# Look at regression tree 
reg.tree.fit
# Look at output 
# Root branch: 310 observations, SSE = 10023, HighwayMPG pred = 27.22
# First split on Horsepower >= 147.5, number of observations in this branch =  260, SSE = 3489, HighwayMPG prediction = 25.56
# etc. 
# It tells us that Horsepower is the var having the most important reduction in SSE, then it's Engine etc. 
# plot tree 
# Directly using base R 
plot(reg.tree.fit, uniform=TRUE, 
     main="Classification Tree for Highway Mileage")
text(tree.fit, use.n=TRUE, all=TRUE, cex=.8)
# Using rpart package
rpart.plot(reg.tree.fit)
# Using the rattle packages 
fancyRpartPlot(reg.tree.fit)
# Pruning 
# Cost Complexity Cirterion 
# Often a balance to be found in the depth and complexity of the tree to optimize predictive performance on unseen data
# rpart is actually behind the scene already aplying a cost complexity alpha values to prune the tree 
# It is performing a 10 fold CV on the training dataset 
# In our example it is finding a minimum CV error for a tree with 6 terminal nodes i.e. leaves
# The dashed line indicates the 1 standard deviation of the minimum CV error. In practice it is common to use 
# a tree with the minimum size within 1 standard deviation of the minimum CV error so a tree with 3 leaves in our case? 
plotcp(reg.tree.fit)
# Force rpart to generate a full tree with no penalty 
reg.tree.fit2 <- rpart(
  formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
  data    = cars04.train.data,
  method  = "anova", 
  control = list(cp = 0)
)
fancyRpartPlot(reg.tree.fit2)
plotcp(reg.tree.fit2)
abline(v = 6, lty = "dashed")
reg.tree.fit2$cptable
# Tuning 
# In addition to the cost complexity parameter 
# One can tune other parameters e.g. minsplit and maxdepth using the control arguments of rpart
# minsplit: Set the minimum number of observations in the node before the algorithm perform a split
# maxdepth: Set the maximum depth of any node of the final tree. The root node is treated a depth 0
# Manually tune these parameters and assess the performance of your model
reg.tree.fit3 <- rpart(
  formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
  data    = cars04.train.data,
  method  = "anova", 
  control = list(minsplit = 10, maxdepth = 6, xval = 10)
)
# Making a grid search is more useful than searching manually 
# Make a grid to search 
hypergrid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(3, 9, 1)
)
head(hypergrid) # head of the grid 
nrow(hypergrid) # total number of combinations 
# One model for each hyperparameter combination 
models <- list()

for (i in 1:nrow(hypergrid)) {
  
  # get minsplit, maxdepth values at row i
  minsplit <- hypergrid$minsplit[i]
  maxdepth <- hypergrid$maxdepth[i]
  # train a model and store in the list
  models[[i]] <- rpart(
    formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
    data    = cars04.train.data,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}
# create a function to extract the minimum error associated with the optimal cost complexity α value for each model. 
# function to get optimal cp
getcp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}
# function to get minimum error
getminerror <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

hypergrid_min <- hypergrid %>%
  mutate(
    cp    = purrr::map_dbl(models, getcp),
    error = purrr::map_dbl(models, getminerror)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)
hypergrid_min
# Optimal Model 
reg.tree.fit.opt <- rpart(
  formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
  data    = cars04.train.data,
  method  = "anova", 
  control = list(minsplit = 6, maxdepth = 8, cp = 0.01)
)
# Predict on test dataset 
# On average, our predicted highway mileages are about 3.25 off from the actual highway mileage.
pred <- predict(reg.tree.fit.opt, newdata = cars04.train.data)
RMSE(pred = pred, obs = cars04.train.data$HighwayMPG)


## Bagging (Boostrap Aggregating) 
# Bagging combines and averages multiple models. 
# Averaging across multiple trees reduces the variability of any one tree and reduces overfitting, which improves predictive performance.
# Steps: 1 create m bootstrapped samples from the training data, 2 for each bootstrapped sample train, make an unprunned tree model, 3
# average the prediction for each tree to create an overall average
## Bagging with ipred package
library(ipred) # bagging 
# make bootstrapping reproducible
set.seed(123)
# train bagged model
bagged.reg.tree.fit <- bagging(
  formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
  data    = cars04.train.data,
  coob    = TRUE
)
bagged.reg.tree.fit
# Out of bag Estimate of  RMSE is 3.64 
# By default the number of trees used is 25 but this might not be enough
# assess 10-50 bagged trees
ntree <- 10:50
# create empty vector to store OOB RMSE values
rmse <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(123)
  
  # perform bagged model
  model <- bagging(
    formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
    data    = cars04.train.data,
    coob    = TRUE,
    nbagg   = ntree[i]
  )
  # get OOB error
  rmse[i] <- model$err
}
plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 45, col = "red", lty = "dashed")
# Use 45 trees instead of 25
opt.bagged.reg.tree.fit <- bagging(
  formula = HighwayMPG ~ Engine + Cylinders + Horsepower,
  data    = cars04.train.data,
  coob    = TRUE,
  nbagg = 45
)
opt.bagged.reg.tree.fit
## Bagging with caret package 
# Bagging with the caret package makes it easier to perform CV and assess variable importance accross bagged trees
# Specify 10-fold cross validation
ctrl <- trainControl(method = "cv",  number = 10) 

# CV bagged model
bagged_cv <- train(
  HighwayMPG ~ .,
  data = cars04.train.data,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)
# assess results
bagged_cv

# Plot 20 most important variables of our model 
# plot most important variables
plot(varImp(bagged_cv), 10)
# Predictions 
pred <- predict(bagged_cv, cars04.test.data)
RMSE(pred, cars04.test.data$HighwayMPG)


## Gradient Boosting Machines 
# GBM builds an ensemble of shallow weak successive trees 
# RF builds an ensemble of independent deep trees 



## Classification Tree 
#¤ To know whether a car is a sports car or not
# Grow tree 
tree.fit <- rpart(Sports ~ Retail + Dealer + Engine + Cylinders + Horsepower + CityMPG + HighwayMPG + Weight + Wheelbase + Length + Width,
             method="class", data = cars04.train.data)
# Look at output
# Root branch: 310 observations, 34 observations i.e. cars that you misclass, prediction is 0 i.e. not a sport car (base model), prediction is 
# correct for 89% of the observations, incorrect for 11% 
# First split: Wheelbase>= 102.5, in this branch: 246 observations, 9 observations misclassed, prediction is 0, prediction is correct for 
# 96% of the observations and incorrect 3% etc. 
# Most important variables to reduce the error (Gini or entropy) are Wheelbase, Horsepower, Retail, etc.
tree.fit
# plot tree 
# Directly using base R 
plot(tree.fit, uniform=TRUE, 
     main="Classification Tree for Sports")
text(tree.fit, use.n=TRUE, all=TRUE, cex=.8)
# Using rpart package
rpart.plot(tree.fit)
# Using the rattle packages 
fancyRpartPlot(tree.fit)
# display the results 
printcp(tree.fit) 
# visualize cross-validation results 
plotcp(tree.fit) 
# detailed summary of splits
summary(tree.fit) 
# make predictions from the tree
tree.pred <- predict(tree.fit, cars04.test.data, type = "class")
cars04.test.data.pred <- cars04.test.data %>% mutate(Sports.pred = tree.pred)
# Performance 
tab <- table(cars04.test.data.pred$Sports, cars04.test.data.pred$Sports.pred)
tab
# Sensitivity TP/(TP + FN)
tab[2,2]/(tab[2,2] + tab[2,1])
# Specificity  TN/(TN + FP)
tab[1,1]/(tab[1,1] + tab[1,2])
# Precision TP/predicted yes.
tab[2,2]/(tab[2,2] + tab[1,2])
# Accuracy (TP + TN)/(All)
(tab[2,2] + tab[1,1])/(tab[2,2] + tab[1,1] + tab[1,2] + tab[2,1])
# Cost Complexity Cirterion 
# Often a balance to be found in the depth and complexity of the tree to optimize predictive performance on unseen data
# Tune the hyperparameters to improve the accuracy of the model 
# This can be done using rpart.control
# Arguments:
# -minsplit: Set the minimum number of observations in the node before the algorithm perform a split
# -minbucket:  Set the minimum number of observations in the final note i.e. the leaf
# -maxdepth: Set the maximum depth of any node of the final tree. The root node is treated a depth 0
# First we construct the accuracy function 
accuracy.tune <- function(fit) {
  predict.unseen <- predict(fit, cars04.test.data, type = 'class')
  table.mat <- table(cars04.test.data$Sports, predict.unseen)
  accuracy.test <- sum(diag(table.mat)) / sum(table.mat)
  accuracy.test
}

control <- rpart.control(minsplit = 50,
                         minbucket = round(50/3),
                         maxdepth = 30,
                         cp = 0)
tune.fit <- rpart(Sports ~Retail + Dealer + Engine + Cylinders + Horsepower + CityMPG + HighwayMPG + Weight + Wheelbase + Length + Width,
                  data = cars04.train.data, method = 'class', control = control)
accuracy.tune(tune.fit)


