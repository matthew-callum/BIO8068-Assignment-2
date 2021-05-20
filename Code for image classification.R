#This code is setup to ensure that all potentially required packages
#are loaded into the R environment for use
library(rinat)
library(sf)
library(keras)
library(caret)
library(tensorflow)
source("download_images.R")

#The four species that have been selected for this model are:
#Gray Wolf (Canis lupis)
#Coyote (Canis latrans)
#African Wild Dog (Lycaon pictus)
#Black-Backed Jackal (Lupulella mesomelas)
#These four species are part of the same taxonomic family (Canidae)
#Subfamily (Caninae), Tribe (Canini), and subtribe (Canina)

#Next we search for records of each of these species to ensure that enough are present
#for this analysis

Gray_Wolf_recs <-  get_inat_obs(taxon_name  = "Canis lupis",
                                year=2016, month=5, 
                                quality = "research",
                                maxresults = 1000)

Coyote_recs <-  get_inat_obs(taxon_name  = "Canis latrans",
                             quality = "research",
                             maxresults = 1000)

African_WildDog_recs <-  get_inat_obs(taxon_name  = "Lycaon pictus",
                                      quality = "research",
                                      maxresults = 1000)

BlackBacked_Jackal_recs <-  get_inat_obs(taxon_name  = "Lupulella mesomelas",
                                         quality = "research",
                                         maxresults = 1000)
#All of these species have enough results for use in the automatic classficiation training
#So now we download the results for use in the training and model

#download_images(spp_recs = Gray_Wolf_recs, spp_folder = "Gray Wolves")
#download_images(spp_recs = Coyote_recs, spp_folder = "Coyote")
#download_images(spp_recs = African_WildDog_recs, spp_folder = "African Wild Dogs")
#download_images(spp_recs = BlackBacked_Jackal_recs, spp_folder = "Black-Backed Jackals")
#DOWNLOADED DO NOT REPEAT CODE

#Now we need to seperate the images that are going to be used for testing into a seperate folder
image_files_path <- "images"
spp_list <- dir(image_files_path)

#Check the number of spp classes (species)
output_n <- length(spp_list)

#Make the test, and species sub-folders
for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

for(folder in 1:output_n){
  for(image in 101:200){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}


#Copying over randomly found images in two loops to the testing folder
#Also scaling down images to speed up analysis and give each image a standard size
img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

#Set the three channels (RGB) that are used in the photos
channels <- 3

#Rescaling the images from 255 to between 0 and 1 and define the ratio of
#Images used for validation (20% for this model) and Training (80% for this model)
train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

#Now that the images have been set up we can define the images used for the 
#Training and Validation within the model

#This code is for the training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

#This code is for the validation images
valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)

#Now we define the number of training and validation samples, batch size, and how
#many epochs the model is trained for
train_samples <- train_image_array_gen$n
valid_samples <- valid_image_array_gen$n
batch_size <- 32
epochs <- 10

#Code to initialise the model
model <- keras_model_sequential()

#Add the varying layers of the model
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(output_n, activation = "softmax") 

#Then we check the structure of the Convolutional Neural Network
print (model)

#Finally we compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

#Now with the compiled model we can set the model off and have it start working
history <- model %>% fit_generator(
  train_image_array_gen,
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  verbose = 2
)

#Lets have a look at the results of the training
plot(history)

#With the model finished (for ease of transport from Remote Desktop) let's save the model
#First detach the imageR package to ensure that the correct command is used
detach("package:imager", unload = TRUE)
#Now save the model
save.image("CanidModel.RData")

#Now we can test the model on the independent dataset that we seperated earlier
#in the "test" folder
path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   classes = spp_list,
                                                   shuffle = FALSE,
                                                   batch_size =1,
                                                   seed = 123)

#Now we can test the training images model
model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)


#With these findings we can assume that the model is working correctly
#However it can still be helpful to create a confusion matrix to get more data
#These use the test images which we store the predicted results in a dataframe
predictions <- model %>% 
  predict_generator(
    generator = test_image_array_gen,
    steps = test_image_array_gen$n
  ) %>% as.data.frame
colnames(predictions) <- spp_list

#And then we create a 4x4 table to store the data produced by the confusion matrix
confusion <- data.frame(matrix(0, nrow=4, ncol=4), row.names=spp_list)
colnames(confusion) <- spp_list

obs_values <- factor(c(rep(spp_list[1], 100),
                       rep(spp_list[2], 100),
                       rep(spp_list[3], 100),
                       rep(spp_list[4], 100)))

pred_values <- factor(colnames(predictions)[apply(predictions, 1, which.max)])

conf_mat <- confusionMatrix(data = pred_values, reference = obs_values)
conf_mat

predictions
pred_values
obs_values
