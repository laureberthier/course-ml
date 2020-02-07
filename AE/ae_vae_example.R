
library(datasets)
library(ggplot2)
library(keras)
library(tensorflow)
library(caret)

#
# Autoencoders
#

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

###
## Example on the Iris data
#

# Data preperation --------------------------------------------------------------------------------

# Load the data
data(iris)
summary(iris)
target_names <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")

# Split in training and test:
#split_ind <- iris$Species %>% caret::createDataPartition(p = 0.9,list = FALSE)
train <- iris#[split_ind,]
#test <- iris[-split_ind,]

# Normalization:
train_iris <- train[,1:4] %>% as.matrix()
train_iris <- apply(train_iris, 2, function(x) normalize(x))

train_iris_y <- as.numeric(train[,5]) %>% 
  keras::to_categorical()

#test_iris <- test[,1:4] %>% as.matrix()
#test_iris <- apply(test_iris, 2, function(x) normalize(x))



# Training model as PCA ---------------------------------------------------------------------------

set.seed(5)

input_layer <- 
  layer_input(shape = c(4)) 

# Structure of encoder
encoder <- 
  input_layer %>% 
  layer_dense(units = 2, activation = 'linear', use_bias = TRUE) # 2 dimensions for the latent layer

# Structure of decoder:
decoder <- 
  encoder %>% 
  layer_dense(units = 4, activation = 'linear', use_bias = TRUE) # 4 dimensions for the original 4 variables

##
# Defining the autoencoder:
autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

autoencoder_model %>% compile(
  loss='mse',
  optimizer='adam'
)
summary(autoencoder_model)

##
# Training the autoencoder:
history <-
  autoencoder_model %>%
  keras::fit(train_iris,
             train_iris,
             epochs=1000,
             batch_size = 16,
             shuffle=TRUE,
             validation_split = 0.1,
             verbose = 0
             #validation_data= list(test_iris, test_iris)
  )
plot(history)


# Visualizations -------------------------------------------------------------------------------------

##
# Look at the reconstructed points:
reconstructed_points_iris <- 
  autoencoder_model %>% 
  keras::predict_on_batch(x = train_iris)

Viz_data <- 
  dplyr::bind_rows(
    as.matrix(reconstructed_points_iris) %>% 
      tibble::as_tibble() %>% 
      setNames(names(train_iris %>% tibble::as_tibble())) %>% 
      dplyr::mutate(data_origin = "reconstructed"),
    train_iris %>% 
      tibble::as_tibble() %>% 
      dplyr::mutate(data_origin = "original")
  )

Viz_data %>%
  ggplot(aes(Petal.Length,Sepal.Width, color = data_origin))+
  geom_point()


# Defining the encoder model with the autoencoder weights:
input_a <- layer_input(shape = 4)
enco <- autoencoder_model$layers[[2]](input_a)
encoder_model <- keras_model(inputs = input_a, outputs = enco)

encoder_model %>% compile(
  loss='mse',
  optimizer='adam'
)


# Predict bottleneck points
embeded_points_linear <- 
  encoder_model %>% 
  predict(x = train_iris)
#embeded_points %>% head


# PCA
pre_process <- caret::preProcess(train_iris,method = "pca",pcaComp = 2)
pca <- predict(pre_process,train_iris)
#pca %>% head

##
# Creating data frame and plotting dimensionality reduction:
Viz_data_encoded <- 
  dplyr::bind_rows(
    pca %>% 
      tibble::as_tibble() %>% 
      setNames(c("dim_1","dim_2")) %>% 
      dplyr::mutate(data_origin = "pca",
                    Species = train$Species),
    as.matrix(embeded_points_linear) %>% 
      tibble::as_tibble() %>% 
      setNames(c("dim_1","dim_2")) %>% 
      dplyr::mutate(data_origin = "embeded_points",
                    Species = train$Species)
  )

Viz_data_encoded %>% 
  ggplot(aes(dim_1,dim_2, color = Species)) +
  geom_point() +
  facet_wrap(~data_origin, scale = "free", ncol = 1)



# Deeper and nonlinear autoencoder -----------------------------------------------------------------

#
# Defining the model
#

input_layer <- 
  layer_input(shape = c(4)) 

# Structure of encoder
encoder <- 
  input_layer %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 2) # 2 dimensions for the output layer

# Structure of decoder:
decoder <- 
  encoder %>% 
  layer_dense(units = 25, activation = "relu") %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 50, activation = "relu") %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4) # 4 dimensions for the original 4 variables

##
# Defining the autoencoder:
autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

autoencoder_model %>% compile(
  loss='mse',
  optimizer='adam'
)
summary(autoencoder_model)

##
# Training the autoencoder:
history <-
  autoencoder_model %>%
  keras::fit(train_iris,
             train_iris,
             epochs=1000,
             batch_size = 16,
             shuffle=TRUE,
             validation_split = 0.1
             #validation_data= list(test_iris, test_iris)
  )
plot(history)


# Visualization ---------------------------------------------------------------------------

##
# Look at the reconstructed points:
reconstructed_points <- 
  autoencoder_model %>% 
  keras::predict_on_batch(x = train_iris)

Viz_data <- 
  dplyr::bind_rows(
    as.matrix(reconstructed_points) %>% 
      tibble::as_tibble() %>% 
      setNames(names(train_iris %>% tibble::as_tibble())) %>% 
      dplyr::mutate(data_origin = "reconstructed"),
    train_iris %>% 
      tibble::as_tibble() %>% 
      dplyr::mutate(data_origin = "original")
  )

Viz_data %>%
  ggplot(aes(Petal.Length,Sepal.Width, color = data_origin))+
  geom_point()


# Comparison of dimensionality reduction with PCA ----------------------------------------------------------

# Extracting weights from autoencoder:
autoencoder_weights <- 
  autoencoder_model %>%
  keras::get_weights()
#autoencoder_weights

# Saving autoencoder weights:
keras::save_model_weights_hdf5(object = autoencoder_model,filepath = 'autoencoder_weights.hdf5',overwrite = TRUE)

# Defining the encoder model with the autoencoder weights:
encoder_model <- keras_model(inputs = input_layer, outputs = encoder)

encoder_model %>% keras::load_model_weights_hdf5(filepath = "autoencoder_weights.hdf5",skip_mismatch = TRUE,by_name = TRUE)

encoder_model %>% compile(
  loss='mse',
  optimizer='adam'
)

# Predict bottleneck points
embeded_points <- 
  encoder_model %>% 
  keras::predict_on_batch(x = train_iris)
#embeded_points %>% head

##
# Creating data frame and plotting dimensionality reduction:
Viz_data_encoded <- 
  dplyr::bind_rows(
    pca %>% 
      tibble::as_tibble() %>% 
      setNames(c("dim_1","dim_2")) %>% 
      dplyr::mutate(data_origin = "pca",
                    Species = train$Species),
    as.matrix(embeded_points_linear) %>% 
      tibble::as_tibble() %>% 
      setNames(c("dim_1","dim_2")) %>% 
      dplyr::mutate(data_origin = "embeded_points_linear",
                    Species = train$Species),
    as.matrix(embeded_points) %>% 
      tibble::as_tibble() %>% 
      setNames(c("dim_1","dim_2")) %>% 
      dplyr::mutate(data_origin = "embeded_points",
                    Species = train$Species)
  )

Viz_data_encoded %>% 
  ggplot(aes(dim_1,dim_2, color = Species))+
  geom_point()+
  facet_wrap(~data_origin, scale = "free")


############################################################################################################

#
# Example on MNIST data:
#

library(dplyr)
library(ggplot2)
library(dslabs)
# Modeling packages
library(stringr)  # for fitting autoencoders
library(tibble)
library(tidyr)

# Data preparation -------------------------------------------------------------------

mnist <- dslabs::read_mnist()
names(mnist)

rand_s <- sample(c(1:2000),2000)

train_X <- mnist$train$images[rand_s,] %>% as.matrix()
train_X <- train_X/max(train_X)

train_y <- mnist$train$labels[rand_s] %>% 
  keras::to_categorical()

test_X <- mnist$test$images %>% as.matrix()
test_X <- test_X/max(test_X)

train_Xa <- array_reshape(train_X, c(2000,28,28), c("C","F"))
train_Xa[sample(1:100,1), 1:28, 1:28] %>%
  as_data_frame() %>% 
  rownames_to_column(var = 'y') %>% 
  gather(x, val, V1:V28) %>%
  mutate(x = str_replace(x, 'V', '')) %>% 
  mutate(x = as.numeric(x),
         y = as.numeric(y)) %>% 
  mutate(y = 28-y) %>%
  ggplot(aes(x, y))+
  geom_tile(aes(fill = val+1))+
  coord_fixed()+
  theme_void()+
  theme(legend.position="none")


# Defining the model -------------------------------------------------------------------

input_layer <- layer_input(shape = 784) 

encoder <- input_layer %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 32)

decoder <- encoder %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 128, activation = 'sigmoid') %>%
  layer_dense(units = 784)

# Defining the autoencoder:
autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

autoencoder_model %>% compile(
  loss='mean_squared_error',
  optimizer='adam')
summary(autoencoder_model)

# Training the autoencoder --------------------------------------------------------------
history <-
  autoencoder_model %>%
  keras::fit(train_X,
             train_X,
             epochs=400,
             batch_size = 100,
             shuffle=TRUE,
             validation_data= list(test_X, test_X)
  )
plot(history)


reconstructed_points <- 
  autoencoder_model %>% 
  keras::predict_on_batch(x = train_X)

non_reg_points <- as.matrix(reconstructed_points)
non_reg_points <- array_reshape(non_reg_points, c(2000,28,28), c("C","F"))

#-----------------------------------------------------------------------------------------

###                         #
## Regularized Autoencoder ##
#                         ###

# Defining the model ------------------------------------------------------------

input_layer <- layer_input(shape = 784) 

encoder <- input_layer %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu', activity_regularizer = regularizer_l2(10e-3)) %>%
  layer_dense(units = 32)

decoder <- encoder %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 128, activation = 'sigmoid') %>%
  layer_dense(units = 784)

##
# Defining the autoencoder:
autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

autoencoder_model %>% compile(
  loss='mean_squared_error',
  optimizer='adam')
summary(autoencoder_model)


# Training the autoencoder --------------------------------------------------

history <-
  autoencoder_model %>%
  keras::fit(train_X,
             train_X,
             epochs=400,
             shuffle=TRUE,
             batch_size=100,
             validation_data= list(test_X, test_X)
  )
plot(history)

# Visualizations ------------------------------------------------------------

reconstructed_points <- 
  autoencoder_model %>% 
  keras::predict_on_batch(x = train_X)

reg_points <- as.matrix(reconstructed_points)
reg_points <- array_reshape(reg_points, c(2000,28,28), c("C","F"))


rand_s <- sample(1:2000,6)
p_list <- list()
for(i in 1:6){
  dd <- train_Xa[rand_s[i], 1:28, 1:28] %>%
    as_data_frame() %>% 
    rownames_to_column(var = 'y') %>% 
    gather(x, val, V1:V28) %>%
    mutate(x = str_replace(x, 'V', '')) %>% 
    mutate(x = as.numeric(x),
           y = as.numeric(y)) %>% 
    mutate(y = 28-y)
  
  p_list[[i]] <- ggplot(dd,aes(x, y))+
    geom_tile(aes(fill = val + 1))+
    coord_fixed()+
    theme_void()+
    scale_fill_gradient2(high = "white", low = "black", mid = "gray", midpoint = 2) +
    theme(legend.position="none")
}
for(i in 7:12){
  dd <- non_reg_points[rand_s[i-6], 1:28, 1:28] %>%
    as_data_frame() %>% 
    rownames_to_column(var = 'y') %>% 
    gather(x, val, V1:V28) %>%
    mutate(x = str_replace(x, 'V', '')) %>% 
    mutate(x = as.numeric(x),
           y = as.numeric(y)) %>% 
    mutate(y = 28-y)
  
  p_list[[i]] <- ggplot(dd,aes(x, y))+
    geom_tile(aes(fill = val + 1))+
    coord_fixed()+
    theme_void()+
    scale_fill_gradient2(high = "white", low = "black", mid = "gray", midpoint = 2) +
    theme(legend.position="none")
}
for(i in 13:18){
  dd <- reg_points[rand_s[i-12], 1:28, 1:28] %>%
    as_data_frame() %>% 
    rownames_to_column(var = 'y') %>% 
    gather(x, val, V1:V28) %>%
    mutate(x = str_replace(x, 'V', '')) %>% 
    mutate(x = as.numeric(x),
           y = as.numeric(y)) %>% 
    mutate(y = 28-y)
  
  p_list[[i]] <- ggplot(dd,aes(x, y))+
      geom_tile(aes(fill = val + 1))+
      coord_fixed()+
      theme_void()+
      scale_fill_gradient2(high = "white", low = "black", mid = "gray", midpoint = 2) +
      theme(legend.position="none")
}


library(gridExtra)
do.call("grid.arrange", c(p_list, nrow=3))




#########################################################################################################################################

#
# Example of Variational Autoencoder:
#

###     ###
## MNIST ##
#         #

# Data preparation --------------------------------------------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")


# Parameters --------------------------------------------------------------

batch_size <- 100
original_dim <- 784
latent_dim <- 2
intermediate_dim <- 256
epochs <- 100
epsilon_std <- 1.0

library(keras)
K <- keras::backend()

# Model definition --------------------------------------------------------

library(gnn)
x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)


vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = "rmsprop", loss = vae_loss, experimental_run_tf_function=FALSE)



# Model training ----------------------------------------------------------

history <- vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)
plot(history)

# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-2.5, 2.5, length.out = n)
grid_y <- seq(-2.5, 2.5, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()

# Generated example
z_sample <- matrix(c(-1, 2.5), ncol = 2)
img1 <- predict(generator, z_sample) %>% matrix(ncol = 28)
plot(as.raster(img1))

#mod_vae <- VAE_model(c(784,256,2), activation = c("relu", "sigmoid"), sd = epsilon_std, loss.type = "binary.cross",
#                     dropout.rate = 0.1)
#
#mod_vae$model %>% compile(optimizer = 'sgd', loss = 'binary_crossentropy')


################################################################################################################

#
# Latent space representation of AE vs VAE
#

###             #
## Autoencoder ##
#             ###

# Model definition --------------------------------------------------------

library(gnn)
x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z <- layer_dense(h, latent_dim)

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_x <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_ae <- decoder_x(h_decoded)

# end-to-end autoencoder
ae <- keras_model(x, x_decoded_ae)

# encoder, from inputs to latent space
encoder_ae <- keras_model(x, z)

# generator, from latent space to reconstructed inputs
decoder_input_ae <- layer_input(shape = c(latent_dim))
h_decoded_2 <- decoder_h(decoder_input_ae)
x_decoded_mean_2 <- decoder_x(h_decoded_2)
generator_ae <- keras_model(decoder_input_ae, x_decoded_mean_2)

ae %>% compile(optimizer = "rmsprop", loss = 'binary_crossentropy', experimental_run_tf_function=FALSE)

# Training the autoencoder --------------------------------------------------------------
history <-
  ae %>%
  keras::fit(x_train,
             x_train,
             epochs = epochs,
             batch_size = batch_size,
             shuffle=TRUE,
             validation_data= list(x_test, x_test)
  )
plot(history)


# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded_ae <- predict(encoder_ae, x_test, batch_size = batch_size)

x_test_encoded_ae %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(0, 75, length.out = n)
grid_y <- seq(10, -30, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator_ae, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()


###################################################################################################################

#
# Deeper VAE
#

# Parameters --------------------------------------------------------------

batch_size <- 100
original_dim <- 784
latent_dim <- 2
intermediate_dim1 <- 256
intermediate_dim2 <- 128
epochs <- 100
epsilon_std <- 1.0

library(keras)
K <- keras::backend()

# Model definition --------------------------------------------------------

library(gnn)
x <- layer_input(shape = c(original_dim))
h1 <- layer_dense(x, intermediate_dim1, activation = "relu")
h2 <- layer_dense(h1, intermediate_dim2, activation = "relu")
z_mean <- layer_dense(h2, latent_dim)
z_log_var <- layer_dense(h2, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
decoder_h2 <- layer_dense(units = intermediate_dim2, activation = "relu")
decoder_h1 <- layer_dense(units = intermediate_dim1, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h2_decoded <- decoder_h2(z)
h1_decoded <- decoder_h1(h2_decoded)
x_decoded_mean <- decoder_mean(h1_decoded)

# end-to-end autoencoder
vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h2_decoded_2 <- decoder_h2(decoder_input)
h1_decoded_2 <- decoder_h1(h2_decoded_2)
x_decoded_mean_2 <- decoder_mean(h1_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)


vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = "rmsprop", loss = vae_loss, experimental_run_tf_function=FALSE)



# Model training ----------------------------------------------------------

history <- vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)
plot(history)

# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-2.5, 2.5, length.out = n)
grid_y <- seq(-2.5, 2.5, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()





