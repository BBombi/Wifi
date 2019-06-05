# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, png, caret,
               lubridate, cluster, caTools)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)

# Loadding the data ----
train <- read.csv("datasets/trainingData.csv")
valid <- read.csv("datasets/validationData.csv")

#Summary some atributes 
summary(train[,521:529])

# Preprocessing ----
# Rescale WAPs units:
train <- cbind(apply(train[1:520],2, function(x) 10^(x/10)*100), 
               train[521:529])
train <- cbind(apply(train[1:520], c(1,2), function(y) 
  ifelse(y == 10^12, y <- 0, y <- y)), train[521:529])

# Converting Floor and Building ID into character variables:
train$BUILDINGID <- as.factor(train$BUILDINGID)
train$FLOOR <- as.factor(train$FLOOR)
valid$BUILDINGID <- as.factor(valid$BUILDINGID)
valid$FLOOR <- as.factor(valid$FLOOR)

# Plotting with plotly ----
plot_ly(train, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~BUILDINGID, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                      yaxis = list(title = 'Longitude'),
                      zaxis = list(title = 'Floor')),
                      title = "Training Data")

plot_ly(valid, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~BUILDINGID, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Validation Data")

# UJI in real life:
grid::grid.raster(readPNG("pictures/UJI_map.png"))

# Looking for duplicates -----
 ## For Training set:
train <- train[order(train$TIMESTAMP), ]
summary(count(paste0("LAT", train$LATITUDE, "LON", train$LONGITUDE, "F", 
                     train$FLOOR, "P", train$PHONEID)))

 ## For Validation:
valid <- valid[order(valid$TIMESTAMP), ]
summary(count(paste0("LAT", valid$LATITUDE, "LON", valid$LONGITUDE, "F", 
                     valid$FLOOR, "P", valid$PHONEID)))
# valid$concat <- paste0("LAT", valid$LATITUDE, "LON", valid$LONGITUDE, "F", 
#                        valid$FLOOR, "P", valid$PHONEID)

  # Removing duplicats at the Trainset:
train$concat <- paste0("LAT", train$LATITUDE, "LON", train$LONGITUDE, "F",
                       train$FLOOR, "P", train$PHONEID)
y <- train[!duplicated(train$concat), ]
train <- train[order(-train$TIMESTAMP), ]
train <- rbind(y, train[!duplicated(train$concat), ])
train <- train[!duplicated(y), ]
train$concat <- NULL
rm(y)

# Join both Datasets, and split them again ----
set.seed(123)
sample <- sample.split(rbind(train, valid), SplitRatio = .80)
newtrain <- subset(rbind(train, valid), sample == TRUE)
newvalid <- subset(rbind(train, valid), sample == FALSE)

# Predicting Building ----
build <- list(c())
build$train <- data.frame(newtrain$BUILDINGID, 
                newtrain[,1:520])

build$valid <- data.frame(newvalid$BUILDINGID, 
                          newvalid[,1:520])
 # RF:
build$rf <- train(newtrain.BUILDINGID ~ ., 
                    data = build$train,
                    method = "rf",
                    tuneGrid=data.frame(mtry=44),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    preProcess = "zv")

build$pred_rf <- predict(build$rf, newdata = build$valid)
build$conf_mat_rf <- table(build$pred_rf, build$valid$newvalid.BUILDINGID)
build$accuracy_rf <- ((sum(diag(build$conf_mat_rf)))/
                          (sum(build$conf_mat_rf)))*100

  # k-NN:  
build$knn <- train(newtrain.BUILDINGID ~ ., 
                   data = build$train, 
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   metric = "Accuracy",
                   preProcess = "zv")

build$pred_knn <- predict(build$knn, newdata = build$valid)
build$conf_mat_knn <- table(build$pred_knn,build$valid$newvalid.BUILDINGID)
build$accuracy_knn <- ((sum(diag(build$conf_mat_knn)))/
                           (sum(build$conf_mat_knn)))*100

# Separate data by building in train and valid----
trainset <- c()
for (i in 0:2) {
  trainset[[paste0("build_",i)]] <- newtrain %>% filter(BUILDINGID == i)
}

validset <- c()
for (i in 0:2) {
  validset[[paste0("build_",i)]] <- newtrain %>% filter(BUILDINGID == i)
}
rm(i, sample, newtrain, newvalid)

# Create data frames per each feature ----
trainset$build_0_lat <- data.frame(trainset$build_0$LATITUDE, 
                                trainset$build_0[,1:520])
trainset$build_0_lon <- data.frame(trainset$build_0$LONGITUDE, 
                                trainset$build_0[,1:520])
trainset$build_0_floor <- data.frame(trainset$build_0$FLOOR, 
                                trainset$build_0[,1:520])

trainset$build_1_lat <- data.frame(trainset$build_1$LATITUDE, 
                                trainset$build_1[,1:520])
trainset$build_1_lon <- data.frame(trainset$build_1$LONGITUDE, 
                                trainset$build_1[,1:520])
trainset$build_1_floor <- data.frame(trainset$build_1$FLOOR, 
                                  trainset$build_1[,1:520])

trainset$build_2_lat <- data.frame(trainset$build_2$LATITUDE, 
                                trainset$build_2[,1:520])
trainset$build_2_lon <- data.frame(trainset$build_2$LONGITUDE, 
                                trainset$build_2[,1:520])
trainset$build_2_floor <- data.frame(trainset$build_2$FLOOR, 
                                  trainset$build_2[,1:520])

validset$build_0_lat <- data.frame(validset$build_0$LATITUDE, 
                                validset$build_0[,1:520])
validset$build_0_lon <- data.frame(validset$build_0$LONGITUDE, 
                                validset$build_0[,1:520])
validset$build_0_floor <- data.frame(validset$build_0$FLOOR, 
                                  validset$build_0[,1:520])

validset$build_1_lat <- data.frame(validset$build_1$LATITUDE, 
                                validset$build_1[,1:520])
validset$build_1_lon <- data.frame(validset$build_1$LONGITUDE, 
                                validset$build_1[,1:520])
validset$build_1_floor <- data.frame(validset$build_1$FLOOR, 
                                  validset$build_1[,1:520])

validset$build_2_lat <- data.frame(validset$build_2$LATITUDE, 
                                validset$build_2[,1:520])
validset$build_2_lon <- data.frame(validset$build_2$LONGITUDE, 
                                validset$build_2[,1:520])
validset$build_2_floor <- data.frame(validset$build_2$FLOOR, 
                                  validset$build_2[,1:520])

# k-NN for Latitude ----
knn <- list(c())
# method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
  ## Build 0:
knn$lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                   data = trainset$build_0_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")
plot(knn$lat_0)

knn$pred_lat_0 <- predict(knn$lat_0, newdata = validset$build_0_lat)
knn$error_lat_0 <- knn$pred_lat_0 - validset$build_0_lat$validset.build_0.LATITUDE
knn$rmse_lat_0 <- sqrt(mean(knn$error_lat_0^2))
knn$rsquared_lat_0 <- (1 - (sum(knn$error_lat_0^2) 
                           / sum((validset$build_0_lat$validset.build_0.LATITUDE - 
                                    mean(validset$build_0_lat$validset.build_0.LATITUDE)
                                  )^2)))*100

## Build 1:
knn$lat_1 <- train(trainset.build_1.LATITUDE ~ ., 
                   data = trainset$build_1_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn$lat_1)

knn$pred_lat_1 <- predict(knn$lat_1, newdata = validset$build_1_lat)
knn$error_lat_1 <- knn$pred_lat_1 - validset$build_1_lat$validset.build_1.LATITUDE
knn$rmse_lat_1 <- sqrt(mean(knn$error_lat_1^2))
knn$rsquared_lat_1 <- (1 - (sum(knn$error_lat_1^2) 
                            / sum((validset$build_1_lat$validset.build_1.LATITUDE - 
                                     mean(validset$build_1_lat$validset.build_1.LATITUDE)
                            )^2)))*100

## Build 2:
knn$lat_2 <- train(trainset.build_2.LATITUDE ~ ., 
                   data = trainset$build_2_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn$lat_2)

knn$pred_lat_2 <- predict(knn$lat_2, newdata = validset$build_2_lat)
knn$error_lat_2 <- knn$pred_lat_2 - validset$build_2_lat$validset.build_2.LATITUDE
knn$rmse_lat_2 <- sqrt(mean(knn$error_lat_2^2))
knn$rsquared_lat_2 <- (1 - (sum(knn$error_lat_2^2) 
                            / sum((validset$build_2_lat$validset.build_2.LATITUDE - 
                                     mean(validset$build_2_lat$validset.build_2.LATITUDE)
                            )^2)))*100

# k-NN for Longitud ----
## Build 0:
knn_lon_0 <- train(trainset.build_0.LONGITUDE ~ ., 
                   data = trainset$build_0_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lon_0)

pred_knn_lon_0 <- predict(knn_lon_0, newdata = validset$build_0_lon)
error_knn_lon_0 <- pred_knn_lon_0 - validset$build_0_lon$validset.build_0.LONGITUDE
rmse_knn_lon_0 <- sqrt(mean(error_knn_lon_0^2))
rsquared_knn_lon_0 <- (1 - (sum(error_knn_lon_0^2) 
                            / sum((validset$build_0_lon$validset.build_0.LONGITUDE - 
                                     mean(validset$build_0_lon$validset.build_0.LONGITUDE)
                            )^2)))*100

## Build 1:
knn_lon_1 <- train(trainset.build_1.LONGITUDE ~ ., 
                   data = trainset$build_1_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lon_1)

pred_knn_lon_1 <- predict(knn_lon_1, newdata = validset$build_1_lon)
error_knn_lon_1 <- pred_knn_lon_1 - validset$build_1_lon$validset.build_1.LONGITUDE
rmse_knn_lon_1 <- sqrt(mean(error_knn_lon_1^2))
rsquared_knn_lon_1 <- (1 - (sum(error_knn_lon_1^2) 
                            / sum((validset$build_1_lon$validset.build_1.LONGITUDE - 
                                     mean(validset$build_1_lon$validset.build_1.LONGITUDE)
                            )^2)))*100

## Build 2:
knn_lon_2 <- train(trainset.build_2.LONGITUDE ~ ., 
                   data = trainset$build_2_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lon_2)

pred_knn_lon_2 <- predict(knn_lon_2, newdata = validset$build_2_lon)
error_knn_lon_2 <- pred_knn_lon_2 - validset$build_2_lon$validset.build_2.LONGITUDE
rmse_knn_lon_2 <- sqrt(mean(error_knn_lon_2^2))
rsquared_knn_lon_2 <- (1 - (sum(error_knn_lon_2^2) 
                            / sum((validset$build_2_lon$validset.build_2.LONGITUDE - 
                                     mean(validset$build_2_lon$validset.build_2.LONGITUDE)
                            )^2)))*100

# k-NN for FLOOR ----
## Build 0:
knn_floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                   data = trainset$build_0_floor,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   metric = "Accuracy",
                   preProcess = "zv")

plot(knn_floor_0)

pred_knn_floor_0 <- predict(knn_floor_0, newdata = validset$build_0_floor)
conf_mat_knn_floor_0 <- table(pred_knn_floor_0, 
                              validset$build_0_floor$validset.build_0.FLOOR)
accuracy_knn_floor_0 <- ((sum(diag(conf_mat_knn_floor_0)))/
                           (sum(conf_mat_knn_floor_0)))*100

## Build 1:
knn_floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                     data = trainset$build_1_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(knn_floor_1)

pred_knn_floor_1 <- predict(knn_floor_1, newdata = validset$build_1_floor)
conf_mat_knn_floor_1 <- table(pred_knn_floor_1, 
                              validset$build_1_floor$validset.build_1.FLOOR)
accuracy_knn_floor_1 <- ((sum(diag(conf_mat_knn_floor_1)))/
                           (sum(conf_mat_knn_floor_1)))*100

## Build 2:
knn_floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                     data = trainset$build_2_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(knn_floor_2)

pred_knn_floor_2 <- predict(knn_floor_2, newdata = validset$build_2_floor)
conf_mat_knn_floor_2 <- table(pred_knn_floor_2, 
                              validset$build_2_floor$validset.build_2.FLOOR)
accuracy_knn_floor_2 <- ((sum(diag(conf_mat_knn_floor_2)))/
                           (sum(conf_mat_knn_floor_2)))*100

# RF for Latitude ----
# method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
rf <- list(c())
## Build 0:
rf$lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                   data = trainset$build_0_lat,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

rf$pred_lat_0 <- predict(rf$lat_0, newdata = validset$build_0_lat)
rf$error_lat_0 <- rf$pred_lat_0 - validset$build_0_lat$validset.build_0.LATITUDE
rf$rmse_lat_0 <- sqrt(mean(rf$error_lat_0^2))
rf$rsquared_lat_0 <- (1 - (sum(rf$error_lat_0^2) 
                            / sum((validset$build_0_lat$validset.build_0.LATITUDE - 
                                     mean(validset$build_0_lat$validset.build_0.LATITUDE)
                            )^2)))*100

## Build 1:
rf_lat_1 <- train(trainset.build_1.LATITUDE ~ ., 
                   data = trainset$build_1_lat,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lat_1)

pred_rf_lat_1 <- predict(rf_lat_1, newdata = validset$build_1_lat)
error_rf_lat_1 <- pred_rf_lat_1 - validset$build_1_lat$validset.build_1.LATITUDE
rmse_rf_lat_1 <- sqrt(mean(error_rf_lat_1^2))
rsquared_rf_lat_1 <- (1 - (sum(error_rf_lat_1^2) 
                            / sum((validset$build_1_lat$validset.build_1.LATITUDE - 
                                     mean(validset$build_1_lat$validset.build_1.LATITUDE)
                            )^2)))*100

## Build 2:
rf_lat_2 <- train(trainset.build_2.LATITUDE ~ ., 
                   data = trainset$build_2_lat,
                  tuneGrid=data.frame(mtry=32),
                   method = "rf",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lat_2)

pred_rf_lat_2 <- predict(rf_lat_2, newdata = validset$build_2_lat)
error_rf_lat_2 <- pred_rf_lat_2 - validset$build_2_lat$validset.build_2.LATITUDE
rmse_rf_lat_2 <- sqrt(mean(error_rf_lat_2^2))
rsquared_rf_lat_2 <- (1 - (sum(error_rf_lat_2^2) 
                            / sum((validset$build_2_lat$validset.build_2.LATITUDE - 
                                     mean(validset$build_2_lat$validset.build_2.LATITUDE)
                            )^2)))*100

# RF for Longitud ----
## Build 0:
rf_lon_0 <- train(trainset.build_0.LONGITUDE ~ ., 
                   data = trainset$build_0_lon,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lon_0)

pred_rf_lon_0 <- predict(rf_lon_0, newdata = validset$build_0_lon)
error_rf_lon_0 <- pred_rf_lon_0 - validset$build_0_lon$validset.build_0.LONGITUDE
rmse_rf_lon_0 <- sqrt(mean(error_rf_lon_0^2))
rsquared_rf_lon_0 <- (1 - (sum(error_rf_lon_0^2) 
                            / sum((validset$build_0_lon$validset.build_0.LONGITUDE - 
                                     mean(validset$build_0_lon$validset.build_0.LONGITUDE)
                            )^2)))*100

## Build 1:
rf_lon_1 <- train(trainset.build_1.LONGITUDE ~ ., 
                   data = trainset$build_1_lon,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lon_1)

pred_rf_lon_1 <- predict(rf_lon_1, newdata = validset$build_1_lon)
error_rf_lon_1 <- pred_rf_lon_1 - validset$build_1_lon$validset.build_1.LONGITUDE
rmse_rf_lon_1 <- sqrt(mean(error_rf_lon_1^2))
rsquared_rf_lon_1 <- (1 - (sum(error_rf_lon_1^2) 
                            / sum((validset$build_1_lon$validset.build_1.LONGITUDE - 
                                     mean(validset$build_1_lon$validset.build_1.LONGITUDE)
                            )^2)))*100

## Build 2:
rf_lon_2 <- train(trainset.build_2.LONGITUDE ~ ., 
                   data = trainset$build_2_lon,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lon_2)

pred_rf_lon_2 <- predict(rf_lon_2, newdata = validset$build_2_lon)
error_rf_lon_2 <- pred_rf_lon_2 - validset$build_2_lon$validset.build_2.LONGITUDE
rmse_rf_lon_2 <- sqrt(mean(error_rf_lon_2^2))
rsquared_rf_lon_2 <- (1 - (sum(error_rf_lon_2^2) 
                            / sum((validset$build_2_lon$validset.build_2.LONGITUDE - 
                                     mean(validset$build_2_lon$validset.build_2.LONGITUDE)
                            )^2)))*100

# RF for FLOOR ----
## Build 0:
rf_floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                     data = trainset$build_0_floor,
                     method = "rf",
                    tuneGrid=data.frame(mtry=32),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(rf_floor_0)

pred_rf_floor_0 <- predict(rf_floor_0, newdata = validset$build_0_floor)
conf_mat_rf_floor_0 <- table(pred_rf_floor_0, 
                              validset$build_0_floor$validset.build_0.FLOOR)
accuracy_rf_floor_0 <- ((sum(diag(conf_mat_rf_floor_0)))/
                           (sum(conf_mat_rf_floor_0)))*100

## Build 1:
rf_floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                     data = trainset$build_1_floor,
                     method = "rf",
                    tuneGrid=data.frame(mtry=32),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(rf_floor_1)

pred_rf_floor_1 <- predict(rf_floor_1, newdata = validset$build_1_floor)
conf_mat_rf_floor_1 <- table(pred_rf_floor_1, 
                              validset$build_1_floor$validset.build_1.FLOOR)
accuracy_rf_floor_1 <- ((sum(diag(conf_mat_rf_floor_1)))/
                           (sum(conf_mat_rf_floor_1)))*100

## Build 2:
rf_floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                     data = trainset$build_2_floor,
                     method = "rf",
                    tuneGrid=data.frame(mtry=32),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(rf_floor_2)

pred_rf_floor_2 <- predict(rf_floor_2, newdata = validset$build_2_floor)
conf_mat_rf_floor_2 <- table(pred_rf_floor_2, 
                              validset$build_2_floor$validset.build_2.FLOOR)
accuracy_rf_floor_2 <- ((sum(diag(conf_mat_rf_floor_2)))/
                           (sum(conf_mat_rf_floor_2)))*100
