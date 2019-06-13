# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, png, caret,
               lubridate, cluster, caTools, RColorBrewer, gridExtra, ISLR,
               gbm)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)

# Loadding the data ----
train <- read.csv("datasets/trainingData.csv")
valid <- read.csv("datasets/validationData.csv")

#Summary some atributes 
summary(train[,521:529])

# Converting Floor and Phone ID into character variables ----
train$BUILDINGID <- as.factor(train$BUILDINGID)
valid$BUILDINGID <- as.factor(valid$BUILDINGID)

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

# Removing duplicates: -----
train <- train[!duplicated(train), ]
# Preprocessing ----
# Rescale WAPs units:
train <- cbind(apply(train[1:520],2, function(x) 10^(x/10)*100), 
               train[521:529])
train <- cbind(apply(train[1:520], c(1,2), function(y) 
  ifelse(y == 10^12, y <- 0, y <- y)), train[521:529])

valid <- cbind(apply(valid[1:520],2, function(x) 10^(x/10)*100), 
               valid[521:529])
valid <- cbind(apply(valid[1:520], c(1,2), function(y) 
  ifelse(y == 10^12, y <- 0, y <- y)), valid[521:529])

# Removing Near Zero Variance WAPs:
x <- nearZeroVar(train[ ,1:520], saveMetrics = TRUE)
# New datasets
newtrain <- train[ ,c(which(x$percentUnique > 0.010362695), 521:529)] %>% 
  dplyr::group_by(BUILDINGID, FLOOR, LATITUDE, LONGITUDE, PHONEID) %>%
  dplyr::sample_frac(0.4)
newvalid <- valid[ ,c(which(x$percentUnique > 0.010362695), 521:529)]

# Predicting Building ----
build <- list(c())
build$train <- data.frame(newtrain$BUILDINGID, newtrain[,c(1:(ncol(newtrain)-9))])

build$valid <- data.frame(newvalid$BUILDINGID, newvalid[,c(1:(ncol(newvalid)-9))])

methods <- c("rf", "knn", "gmb")
control <- c()

for (i in methods) {
  ifelse(i = "rf", control <- trainControl(method = "cv",
                                           number = 2,
                                           verboseIter = TRUE),
         control <- trainControl(method = "cv",
                                 number = 5,
                                 verboseIter = TRUE))
  
  build[[rf]] <- train(newtrain.BUILDINGID ~ ., 
                       data = build$train, 
                       method = "rf",
                       tuneGrid = ifelse(i = "rf", data.frame(mtry=c(22,29,30)),
                                         NULL),
                       trControl = control,
                       metric = "Accuracy",
                       preProcess = "zv")
  
  build[[paste0("pred_",i)]] <- predict(build[[i]], newdata = build$valid)
  build[[paste0("conf_mat_",i)]] <- table(build[[paste0("pred_",i)]], 
                                          build$valid$newvalid.BUILDINGID)
  build[[paste0("accuracy_",i)]] <- ((sum(diag(build[[paste0("conf_mat_",i)]])))/
                                       (sum(build[[paste0("conf_mat_",i)]])))*100
}

# The majority vote
build$pred_majority <- as.factor(
  ifelse(build$pred_knn=='0' & build$pred_rf=='0' | 
           build$pred_knn=='0' & build$pred_gbm=='0' | 
           build$pred_rf=='0' & build$pred_gbm=='0','0',
         ifelse(build$pred_knn=='1' & build$pred_rf=='1' | 
                  build$pred_knn=='1' & build$pred_gbm=='1' | 
                  build$pred_rf=='1' & build$pred_gbm=='1','1',
                ifelse(build$pred_knn=='2' & build$pred_rf=='2' |
                         build$pred_knn=='2' & build$pred_gbm=='2' | 
                         build$pred_rf=='2' & build$pred_gbm=='2',
                       '2',build$pred_gbm))))

build$conf_mat_majority <- table(build$pred_majority, build$valid$newvalid.BUILDINGID)
build$accuracy_majority <- ((sum(diag(build$conf_mat_majority)))/
                              (sum(build$conf_mat_majority)))*100

# Separate data by building in train and valid----
trainset <- c()
for (i in 0:2) {
  trainset[[paste0("build_",i)]] <- newtrain %>% filter(BUILDINGID == i)
}

validset <- c()
for (i in 0:2) {
  validset[[paste0("build_",i)]] <- newvalid %>% filter(BUILDINGID == i)
}

rm(i, newtrain, newvalid)

# Create data frames per each feature ----
trainset$build_0_lat <- data.frame(trainset$build_0$LATITUDE, 
                                   trainset$build_0[,c(1:428)])
trainset$build_0_lon <- data.frame(trainset$build_0$LONGITUDE, 
                                   trainset$build_0[,c(1:428)])
trainset$build_0_floor <- data.frame(trainset$build_0$FLOOR, 
                                     trainset$build_0[,c(1:428)])
trainset$build_0_floor$trainset.build_0.FLOOR <- as.factor(
  trainset$build_0_floor$trainset.build_0.FLOOR)

trainset$build_1_lat <- data.frame(trainset$build_1$LATITUDE, 
                                   trainset$build_1[,c(1:428)])
trainset$build_1_lon <- data.frame(trainset$build_1$LONGITUDE, 
                                   trainset$build_1[,c(1:428)])
trainset$build_1_floor <- data.frame(trainset$build_1$FLOOR, 
                                     trainset$build_1[,c(1:428)])
trainset$build_1_floor$trainset.build_1.FLOOR <- as.factor(
  trainset$build_1_floor$trainset.build_1.FLOOR)

trainset$build_2_lat <- data.frame(trainset$build_2$LATITUDE, 
                                   trainset$build_2[,c(1:428)])
trainset$build_2_lon <- data.frame(trainset$build_2$LONGITUDE, 
                                   trainset$build_2[,c(1:428)])
trainset$build_2_floor <- data.frame(trainset$build_2$FLOOR, 
                                     trainset$build_2[,c(1:428)])
trainset$build_2_floor$trainset.build_2.FLOOR <- as.factor(
  trainset$build_2_floor$trainset.build_2.FLOOR)

validset$build_0_lat <- data.frame(validset$build_0$LATITUDE, 
                                   validset$build_0[,c(1:428)])
validset$build_0_lon <- data.frame(validset$build_0$LONGITUDE, 
                                   validset$build_0[,c(1:428)])
validset$build_0_floor <- data.frame(validset$build_0$FLOOR, 
                                     validset$build_0[,c(1:428)])
validset$build_0_floor$validset.build_0.FLOOR <- as.factor(
  validset$build_0_floor$validset.build_0.FLOOR)

validset$build_1_lat <- data.frame(validset$build_1$LATITUDE, 
                                   validset$build_1[,c(1:428)])
validset$build_1_lon <- data.frame(validset$build_1$LONGITUDE, 
                                   validset$build_1[,c(1:428)])
validset$build_1_floor <- data.frame(validset$build_1$FLOOR, 
                                     validset$build_1[,c(1:428)])
validset$build_1_floor$validset.build_1.FLOOR <- as.factor(
  validset$build_1_floor$validset.build_1.FLOOR)

validset$build_2_lat <- data.frame(validset$build_2$LATITUDE, 
                                   validset$build_2[,c(1:428)])
validset$build_2_lon <- data.frame(validset$build_2$LONGITUDE, 
                                   validset$build_2[,c(1:428)])
validset$build_2_floor <- data.frame(validset$build_2$FLOOR, 
                                     validset$build_2[,c(1:428)])
validset$build_2_floor$validset.build_2.FLOOR <- as.factor(
  validset$build_2_floor$validset.build_2.FLOOR)

# k-NN for Latitude ----
knn <- list(c())
# method = "zv" remove attributes with a near zero variance (close to the same value)
## Build 0:
knn$lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                   data = trainset$build_0_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

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

knn$pred_lat_2 <- predict(knn$lat_2, newdata = validset$build_2_lat)
knn$error_lat_2 <- knn$pred_lat_2 - validset$build_2_lat$validset.build_2.LATITUDE
knn$rmse_lat_2 <- sqrt(mean(knn$error_lat_2^2))
knn$rsquared_lat_2 <- (1 - (sum(knn$error_lat_2^2) 
                            / sum((validset$build_2_lat$validset.build_2.LATITUDE - 
                                     mean(validset$build_2_lat$validset.build_2.LATITUDE)
                            )^2)))*100

# k-NN for Longitud ----
## Build 0:
knn$lon_0 <- train(trainset.build_0.LONGITUDE ~ ., 
                   data = trainset$build_0_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn$pred_lon_0 <- predict(knn$lon_0, newdata = validset$build_0_lon)
knn$error_lon_0 <- knn$pred_lon_0 - validset$build_0_lon$validset.build_0.LONGITUDE
knn$rmse_lon_0 <- sqrt(mean(knn$error_lon_0^2))
knn$rsquared_lon_0 <- (1 - (sum(knn$error_lon_0^2) 
                            / sum((validset$build_0_lon$validset.build_0.LONGITUDE - 
                                     mean(validset$build_0_lon$validset.build_0.LONGITUDE)
                            )^2)))*100

## Build 1:
knn$lon_1 <- train(trainset.build_1.LONGITUDE ~ ., 
                   data = trainset$build_1_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn$pred_lon_1 <- predict(knn$lon_1, newdata = validset$build_1_lon)
knn$error_lon_1 <- knn$pred_lon_1 - validset$build_1_lon$validset.build_1.LONGITUDE
knn$rmse_lon_1 <- sqrt(mean(knn$error_lon_1^2))
knn$rsquared_lon_1 <- (1 - (sum(knn$error_lon_1^2) 
                            / sum((validset$build_1_lon$validset.build_1.LONGITUDE - 
                                     mean(validset$build_1_lon$validset.build_1.LONGITUDE)
                            )^2)))*100

## Build 2:
knn$lon_2 <- train(trainset.build_2.LONGITUDE ~ ., 
                   data = trainset$build_2_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn$pred_lon_2 <- predict(knn$lon_2, newdata = validset$build_2_lon)
knn$error_lon_2 <- knn$pred_lon_2 - validset$build_2_lon$validset.build_2.LONGITUDE
knn$rmse_lon_2 <- sqrt(mean(knn$error_lon_2^2))
knn$rsquared_lon_2 <- (1 - (sum(knn$error_lon_2^2) 
                            / sum((validset$build_2_lon$validset.build_2.LONGITUDE - 
                                     mean(validset$build_2_lon$validset.build_2.LONGITUDE)
                            )^2)))*100

# k-NN for FLOOR ----
## Build 0:
knn$floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                     data = trainset$build_0_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

knn$pred_floor_0 <- predict(knn$floor_0, newdata = validset$build_0_floor)
knn$conf_mat_floor_0 <- table(knn$pred_floor_0, 
                              validset$build_0_floor$validset.build_0.FLOOR)
knn$accuracy_floor_0 <- ((sum(diag(knn$conf_mat_floor_)))/
                           (sum(knn$conf_mat_floor_)))*100

## Build 1:
knn$floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                     data = trainset$build_1_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

knn$pred_floor_1 <- predict(knn$floor_1, newdata = validset$build_1_floor)
knn$conf_mat_floor_1 <- table(knn$pred_floor_1, 
                              validset$build_1_floor$validset.build_1.FLOOR)
knn$accuracy_floor_1 <- ((sum(diag(knn$conf_mat_floor_1)))/
                           (sum(knn$conf_mat_floor_1)))*100

## Build 2:
knn$floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                     data = trainset$build_2_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

knn$pred_floor_2 <- predict(knn$floor_2, newdata = validset$build_2_floor)
knn$conf_mat_floor_2 <- table(knn$pred_floor_2, 
                              validset$build_2_floor$validset.build_2.FLOOR)
knn$accuracy_floor_2 <- ((sum(diag(knn$conf_mat_floor_2)))/
                           (sum(knn$conf_mat_floor_2)))*100

# RF for Latitude ----
# method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
rf <- list(c())
## Build 0:
rf$lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                  data = trainset$build_0_lat,
                  method = "rf",
                  tuneGrid=data.frame(mtry=44),
                  trControl = trainControl(method = "cv", 
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf$pred_lat_0 <- predict(rf$lat_0, newdata = validset$build_0_lat)
rf$error_lat_0 <- rf$pred_lat_0 - validset$build_0_lat$validset.build_0.LATITUDE
rf$rmse_lat_0 <- sqrt(mean(rf$error_lat_0^2))
rf$rsquared_lat_0 <- (1 - (sum(rf$error_lat_0^2) 
                           / sum((validset$build_0_lat$validset.build_0.LATITUDE - 
                                    mean(validset$build_0_lat$validset.build_0.LATITUDE)
                           )^2)))*100

## Build 1:
rf$lat_1 <- train(trainset.build_1.LATITUDE ~ ., 
                  data = trainset$build_1_lat,
                  method = "rf",
                  tuneGrid=data.frame(mtry=35),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf$pred_lat_1 <- predict(rf$lat_1, newdata = validset$build_1_lat)
rf$error_lat_1 <- rf$pred_lat_1  - validset$build_1_lat$validset.build_1.LATITUDE
rf$rmse_lat_1 <- sqrt(mean(rf$error_lat_1^2))
rf$rsquared_lat_1 <- (1 - (sum(rf$error_lat_1^2) 
                           / sum((validset$build_1_lat$validset.build_1.LATITUDE - 
                                    mean(validset$build_1_lat$validset.build_1.LATITUDE)
                           )^2)))*100

## Build 2:
rf$lat_2 <- train(trainset.build_2.LATITUDE ~ ., 
                  data = trainset$build_2_lat,
                  tuneGrid=data.frame(mtry=48),
                  method = "rf",
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf$pred_lat_2 <- predict(rf$lat_2, newdata = validset$build_2_lat)
rf$error_lat_2 <- rf$pred_lat_2 - validset$build_2_lat$validset.build_2.LATITUDE
rf$rmse_lat_2 <- sqrt(mean(rf$error_lat_2^2))
rf$rsquared_lat_2 <- (1 - (sum(rf$error_lat_2^2) 
                           / sum((validset$build_2_lat$validset.build_2.LATITUDE - 
                                    mean(validset$build_2_lat$validset.build_2.LATITUDE)
                           )^2)))*100

# RF for Longitud ----
## Build 0:
rf$lon_0 <- train(trainset.build_0.LONGITUDE ~ ., 
                  data = trainset$build_0_lon,
                  method = "rf",
                  tuneGrid=data.frame(mtry=32),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf$pred_lon_0 <- predict(rf$lon_0, newdata = validset$build_0_lon)
rf$error_lon_0 <- rf$pred_lon_0 - validset$build_0_lon$validset.build_0.LONGITUDE
rf$rmse_lon_0 <- sqrt(mean(rf$error_lon_0^2))
rf$rsquared_lon_0 <- (1 - (sum(rf$error_lon_0^2) 
                           / sum((validset$build_0_lon$validset.build_0.LONGITUDE - 
                                    mean(validset$build_0_lon$validset.build_0.LONGITUDE)
                           )^2)))*100

## Build 1:
rf$lon_1 <- train(trainset.build_1.LONGITUDE ~ ., 
                  data = trainset$build_1_lon,
                  method = "rf",
                  tuneGrid=data.frame(mtry=15),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf$pred_lon_1 <- predict(rf$lon_1, newdata = validset$build_1_lon)
rf$error_lon_1 <- rf$pred_lon_1 - validset$build_1_lon$validset.build_1.LONGITUDE
rf$rmse_lon_1 <- sqrt(mean(rf$error_lon_1^2))
rf$rsquared_lon_1 <- (1 - (sum(rf$error_lon_1^2) 
                           / sum((validset$build_1_lon$validset.build_1.LONGITUDE - 
                                    mean(validset$build_1_lon$validset.build_1.LONGITUDE)
                           )^2)))*100

## Build 2:
rf$lon_2 <- train(trainset.build_2.LONGITUDE ~ ., 
                  data = trainset$build_2_lon,
                  method = "rf",
                  tuneGrid=data.frame(mtry=21),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf$pred_lon_2 <- predict(rf$lon_2, newdata = validset$build_2_lon)
rf$error_lon_2 <- rf$pred_lon_2 - validset$build_2_lon$validset.build_2.LONGITUDE
rf$rmse_lon_2 <- sqrt(mean(rf$error_lon_2^2))
rf$rsquared_lon_2 <- (1 - (sum(rf$error_lon_2^2) 
                           / sum((validset$build_2_lon$validset.build_2.LONGITUDE - 
                                    mean(validset$build_2_lon$validset.build_2.LONGITUDE)
                           )^2)))*100

# RF for FLOOR ----
## Build 0:
rf$floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                    data = trainset$build_0_floor,
                    method = "rf",
                    tuneGrid=data.frame(mtry=40),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    metric = "Accuracy",
                    preProcess = "zv")

rf$pred_floor_0 <- predict(rf$floor_0, newdata = validset$build_0_floor)
rf$conf_mat_floor_0 <- table(rf$pred_floor_0, 
                             validset$build_0_floor$validset.build_0.FLOOR)
rf$accuracy_floor_0 <- ((sum(diag(rf$conf_mat_floor_0)))/
                          (sum(rf$conf_mat_floor_0)))*100

## Build 1:
rf$floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                    data = trainset$build_1_floor,
                    method = "rf",
                    tuneGrid=data.frame(mtry=18),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    metric = "Accuracy",
                    preProcess = "zv")

rf$pred_floor_1 <- predict(rf$floor_1, newdata = validset$build_1_floor)
rf$conf_mat_floor_1 <- table(rf$pred_floor_1, 
                             validset$build_1_floor$validset.build_1.FLOOR)
rf$accuracy_floor_1 <- ((sum(diag(rf$conf_mat_floor_1)))/
                          (sum(rf$conf_mat_floor_1)))*100

## Build 2:
rf$floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                    data = trainset$build_2_floor,
                    method = "rf",
                    tuneGrid=data.frame(mtry=25),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    metric = "Accuracy",
                    preProcess = "zv")

rf$pred_floor_2 <- predict(rf$floor_2, newdata = validset$build_2_floor)
rf$conf_mat_floor_2 <- table(rf$pred_floor_2, 
                             validset$build_2_floor$validset.build_2.FLOOR)
rf$accuracy_floor_2 <- ((sum(diag(rf$conf_mat_floor_2)))/
                          (sum(rf$conf_mat_floor_2)))*100

# GBM for FLOOR ----
gbm <- list(c())
## Build 0:
gbm$floor_0 <- train(trainset.build_0.FLOOR ~ ., 
                     data = trainset$build_0_floor,
                     method = "gbm",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     preProcess = "zv",
                     metric = "Accuracy")

gbm$pred_floor_0 <- predict(gbm$floor_0, newdata = validset$build_0_floor)
gbm$conf_mat_floor_0 <- table(gbm$pred_floor_0, 
                              validset$build_0_floor$validset.build_0.FLOOR)
gbm$accuracy_floor_0 <- ((sum(diag(gbm$conf_mat_floor_0)))/
                           (sum(gbm$conf_mat_floor_0)))*100

## Build 1:
gbm$floor_1 <- train(trainset.build_1.FLOOR ~ ., 
                     data = trainset$build_1_floor,
                     method = "gbm",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     preProcess = "zv",
                     metric = "Accuracy")

gbm$pred_floor_1 <- predict(gbm$floor_1, newdata = validset$build_1_floor)
gbm$conf_mat_floor_1 <- table(gbm$pred_floor_1, 
                              validset$build_1_floor$validset.build_1.FLOOR)
gbm$accuracy_floor_1 <- ((sum(diag(gbm$conf_mat_floor_1)))/
                           (sum(gbm$conf_mat_floor_1)))*100

## Build 2:
gbm$floor_2 <- train(trainset.build_2.FLOOR ~ ., 
                     data = trainset$build_2_floor,
                     method = "gbm",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     preProcess = "zv",
                     metric = "Accuracy")

gbm$pred_floor_2 <- predict(gbm$floor_2, newdata = validset$build_2_floor)
gbm$conf_mat_floor_2 <- table(gbm$pred_floor_2, 
                              validset$build_2_floor$validset.build_2.FLOOR)
gbm$accuracy_floor_2 <- ((sum(diag(gbm$conf_mat_floor_2)))/
                           (sum(gbm$conf_mat_floor_2)))*100
#FLOOR: The majority vote ----
majority <- list(c())
# Build 0:
majority$pred_0 <- as.factor(
  ifelse(knn$pred_floor_0 == rf$pred_floor_0, rf$pred_floor_0, gbm$pred_floor_0))

majority$conf_mat_0 <- table(majority$pred_0, 
                             validset$build_0_floor$validset.build_0.FLOOR)
majority$accuracy_0 <- ((sum(diag(majority$conf_mat_0)))/
                          (sum(majority$conf_mat_0)))*100
# Build 1:
majority$pred_1 <- as.factor(
  ifelse(rf$pred_floor_1 == gbm$pred_floor_1, gbm$pred_floor_1, knn$pred_floor_1))

majority$conf_mat_1 <- table(majority$pred_1, 
                             validset$build_1_floor$validset.build_1.FLOOR)
majority$accuracy_1 <- ((sum(diag(majority$conf_mat_1)))/
                          (sum(majority$conf_mat_1)))*100
# Build 2:
majority$pred_2 <- as.factor(
  ifelse(rf$pred_floor_2 == knn$pred_floor_2, knn$pred_floor_2, gbm$pred_floor_2))

majority$conf_mat_2 <- table(majority$pred_2, 
                             validset$build_2_floor$validset.build_2.FLOOR)
majority$accuracy_2 <- ((sum(diag(majority$conf_mat_2)))/
                          (sum(majority$conf_mat_2)))*100
# Creating data frames to compare models ----
metrics <- list(c())
metrics$building_accuracy <- data.frame(metrics = c("RF", "k-NN", "GBM", 
                                                    "Majority Vote"), 
                                        values = c(build$accuracy_rf, 
                                                   build$accuracy_knn, 
                                                   build$accuracy_gbm, 
                                                   build$accuracy_majority))

metrics$latitude_rmse <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                "RF_1",  "kNN_2", "RF_2"),
                                    values = c(knn$rmse_lat_0, rf$rmse_lat_0, 
                                               knn$rmse_lat_1, rf$rmse_lat_1,
                                               knn$rmse_lat_2, rf$rmse_lat_2))

metrics$latitude_rsquared <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                    "RF_1",  "kNN_2", "RF_2"),
                                        values = c(knn$rsquared_lat_0, rf$rsquared_lat_0, 
                                                   knn$rsquared_lat_1, rf$rsquared_lat_1,
                                                   knn$rsquared_lat_2, rf$rsquared_lat_2))

metrics$longitude_rmse <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                 "RF_1", "kNN_2", "RF_2"),
                                     values = c(knn$rmse_lon_0, rf$rmse_lon_0, 
                                                knn$rmse_lon_1, rf$rmse_lon_1,
                                                knn$rmse_lon_2, rf$rmse_lon_2))

metrics$longitude_rsquared <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                     "RF_1",  "kNN_2", "RF_2"),
                                         values = c(knn$rsquared_lon_0, rf$rsquared_lon_0, 
                                                    knn$rsquared_lon_1, rf$rsquared_lon_1,
                                                    knn$rsquared_lon_2, rf$rsquared_lon_2))

metrics$floor_accuracy <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                 "RF_1", "kNN_2", "RF_2",
                                                 "GBM_1", "GBM_2", "GBM_3",
                                                 "Majority_1", "Majority_2",
                                                 "Majority_3"),
                                     values = c(knn$accuracy_floor_0, rf$accuracy_floor_0, 
                                                knn$accuracy_floor_1, rf$accuracy_floor_1,
                                                knn$accuracy_floor_2, rf$accuracy_floor_2,
                                                gbm$accuracy_floor_0, gbm$accuracy_floor_1,
                                                gbm$accuracy_floor_2, majority$accuracy_0,
                                                majority$accuracy_1, majority$accuracy_2))

# Plotting Metrics----
plots <- list(c())
plots$a <- metrics$latitude_rmse %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each Building",
       y = "RMSE",
       title = "LATITUDE") +
  theme_light() +
  scale_fill_brewer(palette = "GnBu") +
  theme(legend.position="none")

plots$b <- metrics$latitude_rsquared %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each Building",
       y = "RSquared",
       title = "LATITUDE") +
  theme_light() +
  scale_fill_brewer(palette = "PuRd") +
  theme(legend.position="none")

plots$c <- metrics$longitude_rmse %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each Building",
       y = "RMSE",
       title = "LONGITUDE") +
  theme_light() +
  scale_fill_brewer(palette = "GnBu") +
  theme(legend.position="none")

plots$d <- metrics$longitude_rsquared %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each Building",
       y = "RSquared",
       title = "LONGITUDE") +
  theme_light() +
  scale_fill_brewer(palette = "PuRd") +
  theme(legend.position="none")

plots$e <- metrics$building_accuracy %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each method",
       y = "Accuracy",
       title = "Building") +
  theme_light() +
  scale_fill_brewer(palette = "PuBu") +
  theme(legend.position="none")

plots$f <- metrics$floor_accuracy %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each Building",
       y = "Accuracy",
       title = "Floor") +
  theme_light() +
  scale_fill_brewer(palette = "PuBu") +
  theme(legend.position="none")

plots$m <- grid.arrange(plots$a, plots$b, plots$c, plots$d, plots$e, plots$f, 
                        ncol = 2)