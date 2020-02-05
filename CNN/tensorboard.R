# Dataset specifications
batch_size <- 128
num_classes <- 10
epochs <- 10

# keras simple feed-forward NN with 1 hidden layer and 128 units 
model_1 <- keras_model_sequential()
model_1 %>%
  layer_flatten(input_shape = c(28, 28, 1)) %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')


model_1 %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

history_1 <- model_1 %>% fit(
  train_images, train_labels, 
  batch_size = batch_size, epochs = epochs,
  verbose = 1,
  callbacks = list(callback_tensorboard("logs/run-ff-nn1",
                                        histogram_freq = 1,
                                        write_graph = T,
                                        write_images = T),
                   callback_early_stopping(monitor = "val_loss",
                                           patience = 1,
                                           restore_best_weights = T)),
  validation_split = 0.2)

scores_1 <- model_1 %>% evaluate(test_images, test_labels)

######## CONV net
model_2 <- keras_model_sequential()
model_2 %>% layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                          input_shape = c(28, 28, 1)) %>% 
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

history_2 <- model_2 %>% fit(
  train_images, train_labels, 
  batch_size = batch_size, epochs = epochs,
  callbacks = list(callback_tensorboard("logs/run-conv1",
                                        histogram_freq = 1,
                                        write_graph = T,
                                        write_images = T),
                   callback_early_stopping(monitor = "val_loss",
                                           patience = 1,
                                           restore_best_weights = T)),
  validation_split = 0.2
)

scores <- model_2 %>% evaluate(
  x_test, test_labels, verbose = 0
)