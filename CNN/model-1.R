## Fashion MNIST
library(keras)
library(tidyr)
library(ggplot2)

# download the data.
# dataset_fashion_mnist(): part of the keras package
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

# Make some tables about the data and maybe show some in the viewer
# Input image dimensions
img_rows <- 28
img_cols <- 28
num_channels <- 1
input_shape <- c(img_rows, img_cols, num_channels)

test_image <- train_images[1,,]
train_images <- train_images / 255
train_images <- array_reshape(train_images, c(nrow(train_images), img_rows, img_cols, 1))

test_images <- test_images / 255
test_images <- array_reshape(test_images, c(nrow(test_images), img_rows, img_cols, 1))


# Dataset specifications
batch_size <- 128
num_classes <- 10
epochs <- 10




##### Simple feed forward neural net with 1 hidden layer and 128 units 
model_1 <- keras_model_sequential()
model_1 %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')


model_1 %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model_1 %>% fit(train_images, train_labels, epochs = 5)

scores_1 <- model_1 %>% evaluate(test_images, test_labels)


##### CONV net from scratch
model_2 <- keras_model_sequential()
model_2 %>% layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                          input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

model_2 %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model_2 %>% fit(
  x = x_train, y = train_labels, 
  batch_size = batch_size, epochs = epochs,
  validation_split = 0.2
)

scores_2 <- model_2 %>% evaluate(
  x_test, test_labels, verbose = 0
)

# Output metrics
cat('Test loss:', scores_1[[1]], '\n')
cat('Test accuracy:', scores_1[[2]], '\n')
# Output metrics
cat('Test loss:', score_2[[1]], '\n')
cat('Test accuracy:', score_2[[2]], '\n')


##### Transfer learning using Resnet



# ensure we have a 4d tensor with single element in the batch dimension,
# the preprocess the input for prediction using resnet50



reshape_to_df <- function(img){
  image <- as.data.frame(img)
  colnames(image) <- seq_len(ncol(image))
  image$y <- seq_len(nrow(image))
  image <- gather(image, "x", "value", -y)
  image$x <- as.integer(image$x)
  image
}
ggplot_img <- function(df){
  ggplot(df, aes(x = x, y = y, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black", na.value = NA) +
    scale_y_reverse() +
    theme_minimal() +
    theme(panel.grid = element_blank())   +
    theme(aspect.ratio = 1) +
    xlab("") +
    ylab("")
}

train_images2 <- train_images[1, , , , drop = F]
train_images2_re <- array_reshape(rep(train_images[1,,,,drop = F],3), 1, 28, 28, 3))
# make predictions then decode and print them

# instantiate the model
model_3 <- application_resnet50(weights = 'imagenet'
                                )


x <- imagenet_preprocess_input(train_images2[1, ,,,drop = F])

preds <- model_3 %>% predict(x)
imagenet_decode_predictions(preds, top = 3)[[1]]
# }

