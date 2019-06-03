# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, png, caret,
               lubridate)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)

# Loadding the data ----

train <- read.csv("datasets/trainingData.csv")
valid <- read.csv("datasets/validationData.csv")

#Summary some atributes 
summary(train[,521:529])

# train$BFS <- paste0(train$BUILDINGID, "F", 
#                            train$FLOOR, "S", train$SPACEID)
# 
# x <- unique(train$BFS)
# TrainingList <- c()
# for (i in x) {
#   TrainingList[[i]] <- train %>% dplyr::filter(BFS == i)
# }

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

# Preprocessing ----
 # Looking for duplicates -----
train <- train[order(train$TIMESTAMP), ]
train$DateTime <- as_datetime(train$TIMESTAMP)
train$gap <- c(NA, with(train, TIMESTAMP[-1] - TIMESTAMP[-nrow(train)]))
x <- train %>% filter(PHONEID == 1, gap > 0) %>% dplyr::select(DateTime, LONGITUDE, LATITUDE)

 ## For Validation
valid <- valid[order(valid$TIMESTAMP), ]
valid$DateTime <- as_datetime(valid$TIMESTAMP)
valid$gap <- c(NA, with(valid, TIMESTAMP[-1] - TIMESTAMP[-nrow(valid)]))
y <- valid %>% filter(PHONEID == 4, gap = 3) %>% dplyr::select(DateTime, LONGITUDE, LATITUDE)

# Separate data by building in train and valid----
trainset <- c()
for (i in 0:2) {
  trainset[[paste0("build_",i)]] <- train %>% filter(BUILDINGID == i)
}

validset <- c()
for (i in 0:2) {
  validset[[paste0("build_",i)]] <- train %>% filter(BUILDINGID == i)
}
rm(i)

# Converting FLOOR as factor:
trainset$build_0$FLOOR <- as.factor(trainset$build_0$FLOOR)
trainset$build_1$FLOOR <- as.factor(trainset$build_1$FLOOR)
trainset$build_2$FLOOR <- as.factor(trainset$build_2$FLOOR)

validset$build_0$FLOOR <- as.factor(validset$build_0$FLOOR)
validset$build_1$FLOOR <- as.factor(validset$build_1$FLOOR)
validset$build_2$FLOOR <- as.factor(validset$build_2$FLOOR)

# Create data frames per each feature ----
train_build_0_lat <- data.frame(trainset$build_0$LATITUDE, 
                                trainset$build_0[,1:520])
train_build_0_lon <- data.frame(trainset$build_0$LONGITUDE, 
                                trainset$build_0[,1:520])
train_build_0_floor <- data.frame(trainset$build_0$FLOOR, 
                                trainset$build_0[,1:520])

train_build_1_lat <- data.frame(trainset$build_1$LATITUDE, 
                                trainset$build_1[,1:520])
train_build_1_lon <- data.frame(trainset$build_1$LONGITUDE, 
                                trainset$build_1[,1:520])
train_build_1_floor <- data.frame(trainset$build_1$FLOOR, 
                                  trainset$build_1[,1:520])

train_build_2_lat <- data.frame(trainset$build_2$LATITUDE, 
                                trainset$build_2[,1:520])
train_build_2_lon <- data.frame(trainset$build_2$LONGITUDE, 
                                trainset$build_2[,1:520])
train_build_2_floor <- data.frame(trainset$build_2$FLOOR, 
                                  trainset$build_2[,1:520])

valid_build_0_lat <- data.frame(validset$build_0$LATITUDE, 
                                validset$build_0[,1:520])
valid_build_0_lon <- data.frame(validset$build_0$LONGITUDE, 
                                validset$build_0[,1:520])
valid_build_0_floor <- data.frame(validset$build_0$FLOOR, 
                                  validset$build_0[,1:520])

valid_build_1_lat <- data.frame(validset$build_1$LATITUDE, 
                                validset$build_1[,1:520])
valid_build_1_lon <- data.frame(validset$build_1$LONGITUDE, 
                                validset$build_1[,1:520])
valid_build_1_floor <- data.frame(validset$build_1$FLOOR, 
                                  validset$build_1[,1:520])

valid_build_2_lat <- data.frame(validset$build_2$LATITUDE, 
                                validset$build_2[,1:520])
valid_build_2_lon <- data.frame(validset$build_2$LONGITUDE, 
                                validset$build_2[,1:520])
valid_build_2_floor <- data.frame(validset$build_2$FLOOR, 
                                  validset$build_2[,1:520])

# k-NN for Latitude ----
# method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
  ## Build 0:
knn_lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                   data = train_build_0_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lat_0)

pred_knn_lat_0 <- predict(knn_lat_0, newdata = valid_build_0_lat)
error_knn_lat_0 <- pred_knn_lat_0 - valid_build_0_lat$validset.build_0.LATITUDE
rmse_knn_lat_0 <- sqrt(mean(error_knn_lat_0^2))
rsquared_knn_lat_0 <- (1 - (sum(error_knn_lat_0^2) 
                           / sum((valid_build_0_lat$validset.build_0.LATITUDE - 
                                    mean(valid_build_0_lat$validset.build_0.LATITUDE)
                                  )^2)))*100

## Build 1:
knn_lat_1 <- train(trainset.build_1.LATITUDE ~ ., 
                   data = train_build_1_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lat_1)

pred_knn_lat_1 <- predict(knn_lat_1, newdata = valid_build_1_lat)
error_knn_lat_1 <- pred_knn_lat_1 - valid_build_1_lat$validset.build_1.LATITUDE
rmse_knn_lat_1 <- sqrt(mean(error_knn_lat_1^2))
rsquared_knn_lat_1 <- (1 - (sum(error_knn_lat_1^2) 
                            / sum((valid_build_1_lat$validset.build_1.LATITUDE - 
                                     mean(valid_build_1_lat$validset.build_1.LATITUDE)
                            )^2)))*100

## Build 2:
knn_lat_2 <- train(trainset.build_2.LATITUDE ~ ., 
                   data = train_build_2_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lat_2)

pred_knn_lat_2 <- predict(knn_lat_2, newdata = valid_build_2_lat)
error_knn_lat_2 <- pred_knn_lat_2 - valid_build_2_lat$validset.build_2.LATITUDE
rmse_knn_lat_2 <- sqrt(mean(error_knn_lat_2^2))
rsquared_knn_lat_2 <- (1 - (sum(error_knn_lat_2^2) 
                            / sum((valid_build_2_lat$validset.build_2.LATITUDE - 
                                     mean(valid_build_2_lat$validset.build_2.LATITUDE)
                            )^2)))*100

# k-NN for Longitud ----
## Build 0:
knn_lon_0 <- train(trainset.build_0.LONGITUDE ~ ., 
                   data = train_build_0_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lon_0)

pred_knn_lon_0 <- predict(knn_lon_0, newdata = valid_build_0_lon)
error_knn_lon_0 <- pred_knn_lon_0 - valid_build_0_lon$validset.build_0.LONGITUDE
rmse_knn_lon_0 <- sqrt(mean(error_knn_lon_0^2))
rsquared_knn_lon_0 <- (1 - (sum(error_knn_lon_0^2) 
                            / sum((valid_build_0_lon$validset.build_0.LONGITUDE - 
                                     mean(valid_build_0_lon$validset.build_0.LONGITUDE)
                            )^2)))*100

## Build 1:
knn_lon_1 <- train(trainset.build_1.LONGITUDE ~ ., 
                   data = train_build_1_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lon_1)

pred_knn_lon_1 <- predict(knn_lon_1, newdata = valid_build_1_lon)
error_knn_lon_1 <- pred_knn_lon_1 - valid_build_1_lon$validset.build_1.LONGITUDE
rmse_knn_lon_1 <- sqrt(mean(error_knn_lon_1^2))
rsquared_knn_lon_1 <- (1 - (sum(error_knn_lon_1^2) 
                            / sum((valid_build_1_lon$validset.build_1.LONGITUDE - 
                                     mean(valid_build_1_lon$validset.build_1.LONGITUDE)
                            )^2)))*100

## Build 2:
knn_lon_2 <- train(trainset.build_2.LONGITUDE ~ ., 
                   data = train_build_2_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(knn_lon_2)

pred_knn_lon_2 <- predict(knn_lon_2, newdata = valid_build_2_lon)
error_knn_lon_2 <- pred_knn_lon_2 - valid_build_2_lon$validset.build_2.LONGITUDE
rmse_knn_lon_2 <- sqrt(mean(error_knn_lon_2^2))
rsquared_knn_lon_2 <- (1 - (sum(error_knn_lon_2^2) 
                            / sum((valid_build_2_lon$validset.build_2.LONGITUDE - 
                                     mean(valid_build_2_lon$validset.build_2.LONGITUDE)
                            )^2)))*100

# k-NN for FLOOR ----
## Build 0:
knn_floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                   data = train_build_0_floor,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   metric = "Accuracy",
                   preProcess = "zv")

plot(knn_floor_0)

pred_knn_floor_0 <- predict(knn_floor_0, newdata = valid_build_0_floor)
conf_mat_knn_floor_0 <- table(pred_knn_floor_0, 
                              valid_build_0_floor$validset.build_0.FLOOR)
accuracy_knn_floor_0 <- ((sum(diag(conf_mat_knn_floor_0)))/
                           (sum(conf_mat_knn_floor_0)))*100

## Build 1:
knn_floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                     data = train_build_1_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(knn_floor_1)

pred_knn_floor_1 <- predict(knn_floor_1, newdata = valid_build_1_floor)
conf_mat_knn_floor_1 <- table(pred_knn_floor_1, 
                              valid_build_1_floor$validset.build_1.FLOOR)
accuracy_knn_floor_1 <- ((sum(diag(conf_mat_knn_floor_1)))/
                           (sum(conf_mat_knn_floor_1)))*100

## Build 2:
knn_floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                     data = train_build_2_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(knn_floor_2)

pred_knn_floor_2 <- predict(knn_floor_2, newdata = valid_build_2_floor)
conf_mat_knn_floor_2 <- table(pred_knn_floor_2, 
                              valid_build_2_floor$validset.build_2.FLOOR)
accuracy_knn_floor_2 <- ((sum(diag(conf_mat_knn_floor_2)))/
                           (sum(conf_mat_knn_floor_2)))*100

# RF for Latitude ----
# method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
## Build 0:
rf_lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                   data = train_build_0_lat,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lat_0)

pred_rf_lat_0 <- predict(rf_lat_0, newdata = valid_build_0_lat)
error_rf_lat_0 <- pred_rf_lat_0 - valid_build_0_lat$validset.build_0.LATITUDE
rmse_rf_lat_0 <- sqrt(mean(error_rf_lat_0^2))
rsquared_rf_lat_0 <- (1 - (sum(error_rf_lat_0^2) 
                            / sum((valid_build_0_lat$validset.build_0.LATITUDE - 
                                     mean(valid_build_0_lat$validset.build_0.LATITUDE)
                            )^2)))*100

## Build 1:
rf_lat_1 <- train(trainset.build_1.LATITUDE ~ ., 
                   data = train_build_1_lat,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lat_1)

pred_rf_lat_1 <- predict(rf_lat_1, newdata = valid_build_1_lat)
error_rf_lat_1 <- pred_rf_lat_1 - valid_build_1_lat$validset.build_1.LATITUDE
rmse_rf_lat_1 <- sqrt(mean(error_rf_lat_1^2))
rsquared_rf_lat_1 <- (1 - (sum(error_rf_lat_1^2) 
                            / sum((valid_build_1_lat$validset.build_1.LATITUDE - 
                                     mean(valid_build_1_lat$validset.build_1.LATITUDE)
                            )^2)))*100

## Build 2:
rf_lat_2 <- train(trainset.build_2.LATITUDE ~ ., 
                   data = train_build_2_lat,
                  tuneGrid=data.frame(mtry=32),
                   method = "rf",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lat_2)

pred_rf_lat_2 <- predict(rf_lat_2, newdata = valid_build_2_lat)
error_rf_lat_2 <- pred_rf_lat_2 - valid_build_2_lat$validset.build_2.LATITUDE
rmse_rf_lat_2 <- sqrt(mean(error_rf_lat_2^2))
rsquared_rf_lat_2 <- (1 - (sum(error_rf_lat_2^2) 
                            / sum((valid_build_2_lat$validset.build_2.LATITUDE - 
                                     mean(valid_build_2_lat$validset.build_2.LATITUDE)
                            )^2)))*100

# RFfor Longitud ----
## Build 0:
rf_lon_0 <- train(trainset.build_0.LONGITUDE ~ ., 
                   data = train_build_0_lon,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lon_0)

pred_rf_lon_0 <- predict(rf_lon_0, newdata = valid_build_0_lon)
error_rf_lon_0 <- pred_rf_lon_0 - valid_build_0_lon$validset.build_0.LONGITUDE
rmse_rf_lon_0 <- sqrt(mean(error_rf_lon_0^2))
rsquared_rf_lon_0 <- (1 - (sum(error_rf_lon_0^2) 
                            / sum((valid_build_0_lon$validset.build_0.LONGITUDE - 
                                     mean(valid_build_0_lon$validset.build_0.LONGITUDE)
                            )^2)))*100

## Build 1:
rf_lon_1 <- train(trainset.build_1.LONGITUDE ~ ., 
                   data = train_build_1_lon,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lon_1)

pred_rf_lon_1 <- predict(rf_lon_1, newdata = valid_build_1_lon)
error_rf_lon_1 <- pred_rf_lon_1 - valid_build_1_lon$validset.build_1.LONGITUDE
rmse_rf_lon_1 <- sqrt(mean(error_rf_lon_1^2))
rsquared_rf_lon_1 <- (1 - (sum(error_rf_lon_1^2) 
                            / sum((valid_build_1_lon$validset.build_1.LONGITUDE - 
                                     mean(valid_build_1_lon$validset.build_1.LONGITUDE)
                            )^2)))*100

## Build 2:
rf_lon_2 <- train(trainset.build_2.LONGITUDE ~ ., 
                   data = train_build_2_lon,
                   method = "rf",
                  tuneGrid=data.frame(mtry=32),
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

plot(rf_lon_2)

pred_rf_lon_2 <- predict(rf_lon_2, newdata = valid_build_2_lon)
error_rf_lon_2 <- pred_rf_lon_2 - valid_build_2_lon$validset.build_2.LONGITUDE
rmse_rf_lon_2 <- sqrt(mean(error_rf_lon_2^2))
rsquared_rf_lon_2 <- (1 - (sum(error_rf_lon_2^2) 
                            / sum((valid_build_2_lon$validset.build_2.LONGITUDE - 
                                     mean(valid_build_2_lon$validset.build_2.LONGITUDE)
                            )^2)))*100

# RF for FLOOR ----
## Build 0:
rf_floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                     data = train_build_0_floor,
                     method = "rf",
                    tuneGrid=data.frame(mtry=32),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(rf_floor_0)

pred_rf_floor_0 <- predict(rf_floor_0, newdata = valid_build_0_floor)
conf_mat_rf_floor_0 <- table(pred_rf_floor_0, 
                              valid_build_0_floor$validset.build_0.FLOOR)
accuracy_rf_floor_0 <- ((sum(diag(conf_mat_rf_floor_0)))/
                           (sum(conf_mat_rf_floor_0)))*100

## Build 1:
rf_floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                     data = train_build_1_floor,
                     method = "rf",
                    tuneGrid=data.frame(mtry=32),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(rf_floor_1)

pred_rf_floor_1 <- predict(rf_floor_1, newdata = valid_build_1_floor)
conf_mat_rf_floor_1 <- table(pred_rf_floor_1, 
                              valid_build_1_floor$validset.build_1.FLOOR)
accuracy_rf_floor_1 <- ((sum(diag(conf_mat_rf_floor_1)))/
                           (sum(conf_mat_rf_floor_1)))*100

## Build 2:
rf_floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                     data = train_build_2_floor,
                     method = "rf",
                    tuneGrid=data.frame(mtry=32),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

plot(rf_floor_2)

pred_rf_floor_2 <- predict(rf_floor_2, newdata = valid_build_2_floor)
conf_mat_rf_floor_2 <- table(pred_rf_floor_2, 
                              valid_build_2_floor$validset.build_2.FLOOR)
accuracy_rf_floor_2 <- ((sum(diag(conf_mat_rf_floor_2)))/
                           (sum(conf_mat_rf_floor_2)))*100
