# The idea is to reduce the size of the train dataset, and combine it with the validation dataset as well.

# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, png, caret,
               lubridate, cluster, caTools, RColorBrewer, gridExtra, ISLR,
               gbm, caretEnsemble, parallel, doMC, randomForest, DescTools,
               import, RRF, inTrees)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)
registerDoMC(cores = detectCores())

# Loadding the data ----
train2 <- read.csv("datasets/trainingData.csv")
valid2 <- read.csv("datasets/validationData.csv")

#Summary some atributes 
summary(train2[,521:529])

# Converting Floor and Phone ID into character variables ----
train2$BUILDINGID <- as.factor(train2$BUILDINGID)
valid2$BUILDINGID <- as.factor(valid2$BUILDINGID)

# Plotting with plotly ----
plot_ly(train2, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~BUILDINGID, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Training Data")

plot_ly(valid2, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~BUILDINGID, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Validation Data")

# UJI in real life:
grid::grid.raster(readPNG("pictures/UJI_map.png"))

# Removing duplicates: -----
train2$TIMESTAMP <- NULL
valid2$TIMESTAMP <- NULL
train2 <- train2[!duplicated(train2), ]
valid2 <- valid2[order(valid2$PHONEID), ]

# Preprocessing ----
# Rescale WAPs units:
train2 <- cbind(apply(train2[1:520],2, function(x) 10^(x/10)*100), 
               train2[521:528])
train2 <- cbind(apply(train2[1:520], c(1,2), function(y) 
  ifelse(y == 10^12, y <- 0, y <- y)), train2[521:528])

valid2 <- cbind(apply(valid2[1:520],2, function(x) 10^(x/10)*100), 
               valid2[521:528])
valid2 <- cbind(apply(valid2[1:520], c(1,2), function(y) 
  ifelse(y == 10^12, y <- 0, y <- y)), valid2[521:528])

# Removing Near Zero Variance WAPs:
x <- nearZeroVar(train2[ ,1:520], saveMetrics = TRUE)
train2 <- train2[ ,c(which(x$percentUnique > 0.010425899), 521:528)]
valid2 <- valid2[ ,c(which(x$percentUnique > 0.010425899), 521:528)]
# New datasets
trainsample <- train2%>% 
  dplyr::group_by(BUILDINGID, FLOOR, LATITUDE, LONGITUDE, PHONEID, SPACEID) %>%
  dplyr::sample_frac(0.4)
newdf <- rbind.data.frame(trainsample, valid2[1:420, ])
set.seed(123)
sample <- sample.split(newdf, SplitRatio = .80)
newtrain2 <- subset(newdf, sample == TRUE)
newvalid2 <- subset(newdf, sample == FALSE)
newtest <- train2[421:1111, ]

# Predicting Building ----
build2 <- list(c())
build2$train <- data.frame(newtrain2$BUILDINGID, newtrain2[,c(1:(ncol(newtrain2)-8))])

build2$valid <- data.frame(newvalid2$BUILDINGID, newvalid2[,c(1:(ncol(newvalid2)-8))])

methods <- c("rf", "knn", "gbm")
control <- c()
grid <- c()

for (i in methods) {
  ifelse(i == "rf", control <- trainControl(method = "cv",
                                            number = 2,
                                            verboseIter = TRUE),
         control <- trainControl(method = "cv",
                                 number = 5,
                                 verboseIter = TRUE))
  ifelse(i=="rf",grid <- data.frame(mtry=c(22,29,30)),
         ifelse(i=="knn", grid <- expand.grid(k=c(3,5,7,9)),
                grid <- expand.grid(interaction.depth = 1:2,
                                    shrinkage = .1,
                                    n.trees = c(10, 50, 100),
                                    n.minobsinnode = 10)))
  
  build2[[i]] <- train(newtrain2.BUILDINGID ~ ., 
                      data = build2$train, 
                      method = i,
                      tuneGrid = grid,
                      trControl = control,
                      metric = "Accuracy",
                      preProcess = "zv")
                      
  build2[[paste0("pred_",i)]] <- predict(build2[[i]], newdata = build2$valid)
  build2[[paste0("conf_mat_",i)]] <- table(build2[[paste0("pred_",i)]],
                                          build2$valid$newvalid2.BUILDINGID)
  build2[[paste0("accuracy_",i)]] <- ((sum(diag(build2[[paste0("conf_mat_",i)]])))/
                                       (sum(build2[[paste0("conf_mat_",i)]])))*100
}

# The majority vote
build2$pred_majority <- as.factor(
  ifelse(build2$pred_knn=='0' & build2$pred_rf=='0' | 
           build2$pred_knn=='0' & build2$pred_gbm=='0' | 
           build2$pred_rf=='0' & build2$pred_gbm=='0','0',
         ifelse(build2$pred_knn=='1' & build2$pred_rf=='1' | 
                  build2$pred_knn=='1' & build2$pred_gbm=='1' | 
                  build2$pred_rf=='1' & build2$pred_gbm=='1','1',
                ifelse(build2$pred_knn=='2' & build2$pred_rf=='2' |
                         build2$pred_knn=='2' & build2$pred_gbm=='2' | 
                         build2$pred_rf=='2' & build2$pred_gbm=='2',
                       '2',build2$pred_gbm))))

build2$conf_mat_majority <- table(build2$pred_majority, build2$valid$newvalid2.BUILDINGID)
build2$accuracy_majority <- ((sum(diag(build2$conf_mat_majority)))/
                              (sum(build2$conf_mat_majority)))*100
 # Predicting the actual dataset:
train2$rf <- predict(build2$rf, train2)
build2$conf_mat_rfT <- table(train2$rf, train2$BUILDINGID)
build2$accuracy_rfT <- ((sum(diag(build2$conf_mat_rfT)))/
                               (sum(build2$conf_mat_rfT)))*100

train2$knn <- predict(build2$knn, train2)
build2$conf_mat_knnT <- table(train2$knn, train2$BUILDINGID)
build2$accuracy_knnT <- ((sum(diag(build2$conf_mat_knnT)))/
                          (sum(build2$conf_mat_knnT)))*100

train2$gbm <- predict(build2$gbm, train2)
build2$conf_mat_gbmT <- table(train2$gbm, train2$BUILDINGID)
build2$accuracy_gbmT <- ((sum(diag(build2$conf_mat_gbmT)))/
                          (sum(build2$conf_mat_gbmT)))*100

train2$majority_vote <- ifelse(
  predict(build2$gbm, train2) == predict(build2$rf, train2) |
    predict(build2$gbm, train2) == predict(build2$knn, train2), 
  predict(build2$gbm, train2), ifelse(predict(build2$rf, train2) == 
                                      predict(build2$gbm, train2) |
                                      predict(build2$rf, train2) ==
                                      predict(build2$knn, train2), 
                                    predict(build2$rf, train2),
                                       predict(build2$gbm, train2)))
build2$conf_mat_mvT <- table(train2$majority_vote, train2$BUILDINGID)
build2$accuracy_mvT <- ((sum(diag(build2$conf_mat_mvT)))/
                          (sum(build2$conf_mat_mvT)))*100
                              
train2$compare_building <- ifelse(train2$majority_vote == 
                                         build2$train$newtrain2.BUILDINGID, "TRUE",
                                       "FALSE")
# Separate data by building in train and valid----
trainset2 <- c()
for (i in 0:2) {
  trainset2[[paste0("build_",i)]] <- newtrain2 %>% filter(BUILDINGID == i)
}

validset2 <- c()
for (i in 0:2) {
  validset2[[paste0("build_",i)]] <- newvalid2 %>% filter(BUILDINGID == i)
}

rm(i, newtrain2, newvalid2)

# Create data frames per each feature ----
# Lattitude <- list(c())
# Longitude <- list(c())
# Floor <- list(c())
# 
# for (i in 0:2) {
#   Lattitude[[paste0("build_",i)]] <- data.frame(trainset[paste0(("build_",i))])
# }
trainset2$build_0_lat <- data.frame(trainset2$build_0$LATITUDE, 
                                   trainset2$build_0[,c(1:428)])
trainset2$build_0_lon <- data.frame(trainset2$build_0$LONGITUDE, 
                                   trainset2$build_0[,c(1:428)])
trainset2$build_0_floor <- data.frame(trainset2$build_0$FLOOR, 
                                     trainset2$build_0[,c(1:428)])
trainset2$build_0_floor$trainset2.build_0.FLOOR <- as.factor(
  trainset2$build_0_floor$trainset2.build_0.FLOOR)

trainset2$build_1_lat <- data.frame(trainset2$build_1$LATITUDE, 
                                   trainset2$build_1[,c(1:428)])
trainset2$build_1_lon <- data.frame(trainset2$build_1$LONGITUDE, 
                                   trainset2$build_1[,c(1:428)])
trainset2$build_1_floor <- data.frame(trainset2$build_1$FLOOR, 
                                     trainset2$build_1[,c(1:428)])
trainset2$build_1_floor$trainset2.build_1.FLOOR <- as.factor(
  trainset2$build_1_floor$trainset2.build_1.FLOOR)

trainset2$build_2_lat <- data.frame(trainset2$build_2$LATITUDE, 
                                   trainset2$build_2[,c(1:428)])
trainset2$build_2_lon <- data.frame(trainset2$build_2$LONGITUDE, 
                                   trainset2$build_2[,c(1:428)])
trainset2$build_2_floor <- data.frame(trainset2$build_2$FLOOR, 
                                     trainset2$build_2[,c(1:428)])
trainset2$build_2_floor$trainset2.build_2.FLOOR <- as.factor(
  trainset2$build_2_floor$trainset2.build_2.FLOOR)

validset2$build_0_lat <- data.frame(validset2$build_0$LATITUDE, 
                                   validset2$build_0[,c(1:428)])
validset2$build_0_lon <- data.frame(validset2$build_0$LONGITUDE, 
                                   validset2$build_0[,c(1:428)])
validset2$build_0_floor <- data.frame(validset2$build_0$FLOOR, 
                                     validset2$build_0[,c(1:428)])
validset2$build_0_floor$validset2.build_0.FLOOR <- as.factor(
  validset2$build_0_floor$validset2.build_0.FLOOR)

validset2$build_1_lat <- data.frame(validset2$build_1$LATITUDE, 
                                   validset2$build_1[,c(1:428)])
validset2$build_1_lon <- data.frame(validset2$build_1$LONGITUDE, 
                                   validset2$build_1[,c(1:428)])
validset2$build_1_floor <- data.frame(validset2$build_1$FLOOR, 
                                     validset2$build_1[,c(1:428)])
validset2$build_1_floor$validset2.build_1.FLOOR <- as.factor(
  validset2$build_1_floor$validset2.build_1.FLOOR)

validset2$build_2_lat <- data.frame(validset2$build_2$LATITUDE, 
                                   validset2$build_2[,c(1:428)])
validset2$build_2_lon <- data.frame(validset2$build_2$LONGITUDE, 
                                   validset2$build_2[,c(1:428)])
validset2$build_2_floor <- data.frame(validset2$build_2$FLOOR, 
                                     validset2$build_2[,c(1:428)])
validset2$build_2_floor$validset2.build_2.FLOOR <- as.factor(
  validset2$build_2_floor$validset2.build_2.FLOOR)

# k-NN for Latitude ----
knn2 <- list(c())
# method = "zv" remove attributes with a near zero variance (close to the same value)
## Build 0:
knn2$lat_0 <- train(trainset2.build_0.LATITUDE ~ ., 
                   data = trainset2$build_0_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn2$pred_lat_0 <- predict(knn2$lat_0, newdata = validset2$build_0_lat)
knn2$error_lat_0 <- knn2$pred_lat_0 - validset2$build_0_lat$validset2.build_0.LATITUDE
knn2$rmse_lat_0 <- sqrt(mean(knn2$error_lat_0^2))
knn2$rsquared_lat_0 <- (1 - (sum(knn2$error_lat_0^2) 
                            / sum((validset2$build_0_lat$validset2.build_0.LATITUDE - 
                                     mean(validset2$build_0_lat$validset2.build_0.LATITUDE)
                            )^2)))*100

## Build 1:
knn2$lat_1 <- train(trainset2.build_1.LATITUDE ~ ., 
                   data = trainset2$build_1_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn2$pred_lat_1 <- predict(knn2$lat_1, newdata = validset2$build_1_lat)
knn2$error_lat_1 <- knn2$pred_lat_1 - validset2$build_1_lat$validset2.build_1.LATITUDE
knn2$rmse_lat_1 <- sqrt(mean(knn2$error_lat_1^2))
knn2$rsquared_lat_1 <- (1 - (sum(knn2$error_lat_1^2) 
                            / sum((validset2$build_1_lat$validset2.build_1.LATITUDE - 
                                     mean(validset2$build_1_lat$validset2.build_1.LATITUDE)
                            )^2)))*100

## Build 2:
knn2$lat_2 <- train(trainset2.build_2.LATITUDE ~ ., 
                   data = trainset2$build_2_lat,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn2$pred_lat_2 <- predict(knn2$lat_2, newdata = validset2$build_2_lat)
knn2$error_lat_2 <- knn2$pred_lat_2 - validset2$build_2_lat$validset2.build_2.LATITUDE
knn2$rmse_lat_2 <- sqrt(mean(knn2$error_lat_2^2))
knn2$rsquared_lat_2 <- (1 - (sum(knn2$error_lat_2^2) 
                            / sum((validset2$build_2_lat$validset2.build_2.LATITUDE - 
                                     mean(validset2$build_2_lat$validset2.build_2.LATITUDE)
                            )^2)))*100

# k-NN for Longitud ----
## Build 0:
knn2$lon_0 <- train(trainset2.build_0.LONGITUDE ~ ., 
                   data = trainset2$build_0_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn2$pred_lon_0 <- predict(knn2$lon_0, newdata = validset2$build_0_lon)
knn2$error_lon_0 <- knn2$pred_lon_0 - validset2$build_0_lon$validset2.build_0.LONGITUDE
knn2$rmse_lon_0 <- sqrt(mean(knn2$error_lon_0^2))
knn2$rsquared_lon_0 <- (1 - (sum(knn2$error_lon_0^2) 
                            / sum((validset2$build_0_lon$validset2.build_0.LONGITUDE - 
                                     mean(validset2$build_0_lon$validset2.build_0.LONGITUDE)
                            )^2)))*100

## Build 1:
knn2$lon_1 <- train(trainset2.build_1.LONGITUDE ~ ., 
                   data = trainset2$build_1_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn2$pred_lon_1 <- predict(knn2$lon_1, newdata = validset2$build_1_lon)
knn2$error_lon_1 <- knn2$pred_lon_1 - validset2$build_1_lon$validset2.build_1.LONGITUDE
knn2$rmse_lon_1 <- sqrt(mean(knn2$error_lon_1^2))
knn2$rsquared_lon_1 <- (1 - (sum(knn2$error_lon_1^2) 
                            / sum((validset2$build_1_lon$validset2.build_1.LONGITUDE - 
                                     mean(validset2$build_1_lon$validset2.build_1.LONGITUDE)
                            )^2)))*100

## Build 2:
knn2$lon_2 <- train(trainset2.build_2.LONGITUDE ~ ., 
                   data = trainset2$build_2_lon,
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   preProcess = "zv")

knn2$pred_lon_2 <- predict(knn2$lon_2, newdata = validset2$build_2_lon)
knn2$error_lon_2 <- knn2$pred_lon_2 - validset2$build_2_lon$validset2.build_2.LONGITUDE
knn2$rmse_lon_2 <- sqrt(mean(knn2$error_lon_2^2))
knn2$rsquared_lon_2 <- (1 - (sum(knn2$error_lon_2^2) 
                            / sum((validset2$build_2_lon$validset2.build_2.LONGITUDE - 
                                     mean(validset2$build_2_lon$validset2.build_2.LONGITUDE)
                            )^2)))*100

# k-NN for FLOOR ----
## Build 0:
knn2$floor_0 <- train(trainset2.build_0.FLOOR ~ ., 
                     data = trainset2$build_0_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

knn2$pred_floor_0 <- predict(knn2$floor_0, newdata = validset2$build_0_floor)
knn2$conf_mat_floor_0 <- table(knn2$pred_floor_0, 
                              validset2$build_0_floor$validset2.build_0.FLOOR)
knn2$accuracy_floor_0 <- ((sum(diag(knn2$conf_mat_floor_0)))/
                           (sum(knn2$conf_mat_floor_0)))*100

## Build 1:
knn2$floor_1 <- train(trainset2.build_1.FLOOR ~ ., 
                     data = trainset2$build_1_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

knn2$pred_floor_1 <- predict(knn2$floor_1, newdata = validset2$build_1_floor)
knn2$conf_mat_floor_1 <- table(knn2$pred_floor_1, 
                              validset2$build_1_floor$validset2.build_1.FLOOR)
knn2$accuracy_floor_1 <- ((sum(diag(knn2$conf_mat_floor_1)))/
                           (sum(knn2$conf_mat_floor_1)))*100

## Build 2:
knn2$floor_2 <- train(trainset2.build_2.FLOOR ~ ., 
                     data = trainset2$build_2_floor,
                     method = "knn",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     metric = "Accuracy",
                     preProcess = "zv")

knn2$pred_floor_2 <- predict(knn2$floor_2, newdata = validset2$build_2_floor)
knn2$conf_mat_floor_2 <- table(knn2$pred_floor_2, 
                              validset2$build_2_floor$validset2.build_2.FLOOR)
knn2$accuracy_floor_2 <- ((sum(diag(knn2$conf_mat_floor_2)))/
                           (sum(knn2$conf_mat_floor_2)))*100

# RF for Latitude ----
# method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.
rf2 <- list(c())
## Build 0:
rf2$lat_0 <- train(trainset2.build_0.LATITUDE ~ ., 
                  data = trainset2$build_0_lat,
                  method = "rf",
                  tuneGrid=data.frame(mtry=44),
                  trControl = trainControl(method = "cv", 
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf2$pred_lat_0 <- predict(rf2$lat_0, newdata = validset2$build_0_lat)
rf2$error_lat_0 <- rf2$pred_lat_0 - validset2$build_0_lat$validset2.build_0.LATITUDE
rf2$rmse_lat_0 <- sqrt(mean(rf2$error_lat_0^2))
rf2$rsquared_lat_0 <- (1 - (sum(rf2$error_lat_0^2) 
                           / sum((validset2$build_0_lat$validset2.build_0.LATITUDE - 
                                    mean(validset2$build_0_lat$validset2.build_0.LATITUDE)
                           )^2)))*100

## Build 1:
rf2$lat_1 <- train(trainset2.build_1.LATITUDE ~ ., 
                  data = trainset2$build_1_lat,
                  method = "rf",
                  tuneGrid=data.frame(mtry=35),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf2$pred_lat_1 <- predict(rf2$lat_1, newdata = validset2$build_1_lat)
rf2$error_lat_1 <- rf2$pred_lat_1  - validset2$build_1_lat$validset2.build_1.LATITUDE
rf2$rmse_lat_1 <- sqrt(mean(rf2$error_lat_1^2))
rf2$rsquared_lat_1 <- (1 - (sum(rf2$error_lat_1^2) 
                           / sum((validset2$build_1_lat$validset2.build_1.LATITUDE - 
                                    mean(validset2$build_1_lat$validset2.build_1.LATITUDE)
                           )^2)))*100

## Build 2:
rf2$lat_2 <- train(trainset2.build_2.LATITUDE ~ ., 
                  data = trainset2$build_2_lat,
                  tuneGrid=data.frame(mtry=48),
                  method = "rf",
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf2$pred_lat_2 <- predict(rf2$lat_2, newdata = validset2$build_2_lat)
rf2$error_lat_2 <- rf2$pred_lat_2 - validset2$build_2_lat$validset2.build_2.LATITUDE
rf2$rmse_lat_2 <- sqrt(mean(rf2$error_lat_2^2))
rf2$rsquared_lat_2 <- (1 - (sum(rf2$error_lat_2^2) 
                           / sum((validset2$build_2_lat$validset2.build_2.LATITUDE - 
                                    mean(validset2$build_2_lat$validset2.build_2.LATITUDE)
                           )^2)))*100

# RF for Longitud ----
## Build 0:
rf2$lon_0 <- train(trainset2.build_0.LONGITUDE ~ ., 
                  data = trainset2$build_0_lon,
                  method = "rf",
                  tuneGrid=data.frame(mtry=32),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf2$pred_lon_0 <- predict(rf2$lon_0, newdata = validset2$build_0_lon)
rf2$error_lon_0 <- rf2$pred_lon_0 - validset2$build_0_lon$validset2.build_0.LONGITUDE
rf2$rmse_lon_0 <- sqrt(mean(rf2$error_lon_0^2))
rf2$rsquared_lon_0 <- (1 - (sum(rf2$error_lon_0^2) 
                           / sum((validset2$build_0_lon$validset2.build_0.LONGITUDE - 
                                    mean(validset2$build_0_lon$validset2.build_0.LONGITUDE)
                           )^2)))*100

## Build 1:
rf2$lon_1 <- train(trainset2.build_1.LONGITUDE ~ ., 
                  data = trainset2$build_1_lon,
                  method = "rf",
                  tuneGrid=data.frame(mtry=15),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf2$pred_lon_1 <- predict(rf2$lon_1, newdata = validset2$build_1_lon)
rf2$error_lon_1 <- rf2$pred_lon_1 - validset2$build_1_lon$validset2.build_1.LONGITUDE
rf2$rmse_lon_1 <- sqrt(mean(rf2$error_lon_1^2))
rf2$rsquared_lon_1 <- (1 - (sum(rf2$error_lon_1^2) 
                           / sum((validset2$build_1_lon$validset2.build_1.LONGITUDE - 
                                    mean(validset2$build_1_lon$validset2.build_1.LONGITUDE)
                           )^2)))*100

## Build 2:
rf2$lon_2 <- train(trainset2.build_2.LONGITUDE ~ ., 
                  data = trainset2$build_2_lon,
                  method = "rf",
                  tuneGrid=data.frame(mtry=21),
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

rf2$pred_lon_2 <- predict(rf2$lon_2, newdata = validset2$build_2_lon)
rf2$error_lon_2 <- rf2$pred_lon_2 - validset2$build_2_lon$validset2.build_2.LONGITUDE
rf2$rmse_lon_2 <- sqrt(mean(rf2$error_lon_2^2))
rf2$rsquared_lon_2 <- (1 - (sum(rf2$error_lon_2^2) 
                           / sum((validset2$build_2_lon$validset2.build_2.LONGITUDE - 
                                    mean(validset2$build_2_lon$validset2.build_2.LONGITUDE)
                           )^2)))*100

# RF for FLOOR ----
## Build 0:
rf2$floor_0 <- train(trainset2.build_0.FLOOR ~ ., 
                    data = trainset2$build_0_floor,
                    method = "rf",
                    tuneGrid=data.frame(mtry=40),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    metric = "Accuracy",
                    preProcess = "zv")

rf2$pred_floor_0 <- predict(rf2$floor_0, newdata = validset2$build_0_floor)
rf2$conf_mat_floor_0 <- table(rf2$pred_floor_0, 
                             validset2$build_0_floor$validset2.build_0.FLOOR)
rf2$accuracy_floor_0 <- ((sum(diag(rf2$conf_mat_floor_0)))/
                          (sum(rf2$conf_mat_floor_0)))*100

## Build 1:
rf2$floor_1 <- train(trainset2.build_1.FLOOR ~ ., 
                    data = trainset2$build_1_floor,
                    method = "rf",
                    tuneGrid=data.frame(mtry=18),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    metric = "Accuracy",
                    preProcess = "zv")

rf2$pred_floor_1 <- predict(rf2$floor_1, newdata = validset2$build_1_floor)
rf2$conf_mat_floor_1 <- table(rf2$pred_floor_1, 
                             validset2$build_1_floor$validset2.build_1.FLOOR)
rf2$accuracy_floor_1 <- ((sum(diag(rf2$conf_mat_floor_1)))/
                          (sum(rf2$conf_mat_floor_1)))*100

## Build 2:
rf2$floor_2 <- train(trainset2.build_2.FLOOR ~ ., 
                    data = trainset2$build_2_floor,
                    method = "rf",
                    tuneGrid=data.frame(mtry=25),
                    trControl = trainControl(method = "cv",
                                             number = 5,
                                             verboseIter = TRUE),
                    metric = "Accuracy",
                    preProcess = "zv")

rf2$pred_floor_2 <- predict(rf2$floor_2, newdata = validset2$build_2_floor)
rf2$conf_mat_floor_2 <- table(rf2$pred_floor_2, 
                             validset2$build_2_floor$validset2.build_2.FLOOR)
rf2$accuracy_floor_2 <- ((sum(diag(rf2$conf_mat_floor_2)))/
                          (sum(rf2$conf_mat_floor_2)))*100

# GBM for FLOOR ----
gbm2 <- list(c())
## Build 0:
gbm2$floor_0 <- train(trainset2.build_0.FLOOR ~ ., 
                     data = trainset2$build_0_floor,
                     method = "gbm",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     preProcess = "zv",
                     metric = "Accuracy")

gbm2$pred_floor_0 <- predict(gbm2$floor_0, newdata = validset2$build_0_floor)
gbm2$conf_mat_floor_0 <- table(gbm2$pred_floor_0, 
                              validset2$build_0_floor$validset2.build_0.FLOOR)
gbm2$accuracy_floor_0 <- ((sum(diag(gbm2$conf_mat_floor_0)))/
                           (sum(gbm2$conf_mat_floor_0)))*100

## Build 1:
gbm2$floor_1 <- train(trainset2.build_1.FLOOR ~ ., 
                     data = trainset2$build_1_floor,
                     method = "gbm",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     preProcess = "zv",
                     metric = "Accuracy")

gbm2$pred_floor_1 <- predict(gbm2$floor_1, newdata = validset2$build_1_floor)
gbm2$conf_mat_floor_1 <- table(gbm2$pred_floor_1, 
                              validset2$build_1_floor$validset2.build_1.FLOOR)
gbm2$accuracy_floor_1 <- ((sum(diag(gbm2$conf_mat_floor_1)))/
                           (sum(gbm2$conf_mat_floor_1)))*100

## Build 2:
gbm2$floor_2 <- train(trainset2.build_2.FLOOR ~ ., 
                     data = trainset2$build_2_floor,
                     method = "gbm",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              verboseIter = TRUE),
                     preProcess = "zv",
                     metric = "Accuracy")

gbm2$pred_floor_2 <- predict(gbm2$floor_2, newdata = validset2$build_2_floor)
gbm2$conf_mat_floor_2 <- table(gbm2$pred_floor_2, 
                              validset2$build_2_floor$validset2.build_2.FLOOR)
gbm2$accuracy_floor_2 <- ((sum(diag(gbm2$conf_mat_floor_2)))/
                           (sum(gbm2$conf_mat_floor_2)))*100
# Bagged Model for Lattitude ----
bag <- list(c())
## Build 0:
seeds <- vector(mode = "list", length = nrow(trainset$build_0_lat) + 1)
seeds <- lapply(seeds, function(x) 1:20)

bag$lat_0 <- train(trainset.build_0.LATITUDE ~ ., 
                   data = trainset$build_0_lat,
                   method = "bag",
                   trControl = trainControl(method = "cv", 
                                            number = 2,
                                            verboseIter = TRUE,
                                            seeds = seeds),
                   preProcess = "zv",
                   tuneGrid = data.frame(vars = seq(1, 15, by = 2)),
                   bagControl = bagControl(fit = ldaBag$fit,
                                           predict = ldaBag$pred,
                                           aggregate = ldaBag$aggregate))

bag$pred_lat_0 <- predict(bag$lat_0, newdata = validset$build_0_lat)
bag$error_lat_0 <- bag$pred_lat_0 - validset$build_0_lat$validset.build_0.LATITUDE
bag$rmse_lat_0 <- sqrt(mean(bag$error_lat_0^2))
bag$rsquared_lat_0 <- (1 - (sum(bag$error_lat_0^2) 
                           / sum((validset$build_0_lat$validset.build_0.LATITUDE - 
                                    mean(validset$build_0_lat$validset.build_0.LATITUDE)
                           )^2)))*100

## Build 1:
bag$lat_1 <- train(trainset.build_1.LATITUDE ~ ., 
                  data = trainset$build_1_lat,
                  method = "bag",
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

bag$pred_lat_1 <- predict(bag$lat_1, newdata = validset$build_1_lat)
bag$error_lat_1 <- bag$pred_lat_1  - validset$build_1_lat$validset.build_1.LATITUDE
bag$rmse_lat_1 <- sqrt(mean(bag$error_lat_1^2))
bag$rsquared_lat_1 <- (1 - (sum(bag$error_lat_1^2) 
                           / sum((validset$build_1_lat$validset.build_1.LATITUDE - 
                                    mean(validset$build_1_lat$validset.build_1.LATITUDE)
                           )^2)))*100

## Build 2:
bag$lat_2 <- train(trainset.build_2.LATITUDE ~ ., 
                  data = trainset$build_2_lat,
                  method = "bag",
                  trControl = trainControl(method = "cv",
                                           number = 5,
                                           verboseIter = TRUE),
                  preProcess = "zv",
                  metric = "RMSE")

bag$pred_lat_2 <- predict(bag$lat_2, newdata = validset$build_2_lat)
bag$error_lat_2 <- bag$pred_lat_2 - validset$build_2_lat$validset.build_2.LATITUDE
bag$rmse_lat_2 <- sqrt(mean(bag$error_lat_2^2))
bag$rsquared_lat_2 <- (1 - (sum(bag$error_lat_2^2) 
                           / sum((validset$build_2_lat$validset.build_2.LATITUDE - 
                                    mean(validset$build_2_lat$validset.build_2.LATITUDE)
                           )^2)))*100
#FLOOR: The majority vote ----
majority2 <- list(c())
# Build 0:
majority2$pred_0 <- as.factor(
  ifelse(knn2$pred_floor_0 == rf2$pred_floor_0, rf2$pred_floor_0, gbm2$pred_floor_0))

majority2$conf_mat_0 <- table(majority2$pred_0, 
                             validset2$build_0_floor$validset2.build_0.FLOOR)
majority2$accuracy_0 <- ((sum(diag(majority2$conf_mat_0)))/
                         (sum(majority2$conf_mat_0)))*100
# Build 1:
majority2$pred_1 <- as.factor(
  ifelse(rf2$pred_floor_1 == gbm2$pred_floor_1, gbm2$pred_floor_1, knn2$pred_floor_1))

majority2$conf_mat_1 <- table(majority2$pred_1, 
                             validset2$build_1_floor$validset2.build_1.FLOOR)
majority2$accuracy_1 <- ((sum(diag(majority2$conf_mat_1)))/
                          (sum(majority2$conf_mat_1)))*100
# Build 2:
majority2$pred_2 <- as.factor(
  ifelse(rf2$pred_floor_2 == knn2$pred_floor_2, knn2$pred_floor_2, gbm2$pred_floor_2))

majority2$conf_mat_2 <- table(majority2$pred_2, 
                             validset2$build_2_floor$validset2.build_2.FLOOR)
majority2$accuracy_2 <- ((sum(diag(majority2$conf_mat_2)))/
                          (sum(majority2$conf_mat_2)))*100
# Creating data frames to compare models ----
metrics2 <- list(c())
metrics2$building_accuracy <- data.frame(metrics = c("RF", "k-NN", "GBM", 
                                                    "Majority Vote"), 
                                        values = c(build2$accuracy_rf, 
                                                   build2$accuracy_knn, 
                                                   build2$accuracy_gbm, 
                                                   build2$accuracy_majority))

metrics2$latitude_rmse <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                "RF_1",  "kNN_2", "RF_2"),
                                    values = c(knn2$rmse_lat_0, rf2$rmse_lat_0, 
                                               knn2$rmse_lat_1, rf2$rmse_lat_1,
                                               knn2$rmse_lat_2, rf2$rmse_lat_2))

metrics2$latitude_rsquared <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                    "RF_1",  "kNN_2", "RF_2"),
                                        values = c(knn2$rsquared_lat_0, rf2$rsquared_lat_0, 
                                                   knn2$rsquared_lat_1, rf2$rsquared_lat_1,
                                                   knn2$rsquared_lat_2, rf2$rsquared_lat_2))

metrics2$longitude_rmse <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                 "RF_1", "kNN_2", "RF_2"),
                                     values = c(knn2$rmse_lon_0, rf2$rmse_lon_0, 
                                                knn2$rmse_lon_1, rf2$rmse_lon_1,
                                                knn2$rmse_lon_2, rf2$rmse_lon_2))

metrics2$longitude_rsquared <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                     "RF_1",  "kNN_2", "RF_2"),
                                         values = c(knn2$rsquared_lon_0, rf2$rsquared_lon_0, 
                                                    knn2$rsquared_lon_1, rf2$rsquared_lon_1,
                                                    knn2$rsquared_lon_2, rf2$rsquared_lon_2))

metrics2$floor_accuracy <- data.frame(metrics = c("kNN_0", "RF_0", "kNN_1", 
                                                 "RF_1", "kNN_2", "RF_2",
                                                 "GBM_0", "GBM_1", "GBM_2",
                                                 "Majority_0", "Majority_1",
                                                 "Majority_2"),
                                     values = c(knn2$accuracy_floor_0, rf2$accuracy_floor_0, 
                                                knn2$accuracy_floor_1, rf2$accuracy_floor_1,
                                                knn2$accuracy_floor_2, rf2$accuracy_floor_2,
                                                gbm2$accuracy_floor_0, gbm2$accuracy_floor_1,
                                                gbm2$accuracy_floor_2, majority2$accuracy_0,
                                                majority2$accuracy_1, majority2$accuracy_2))

# Plotting Metrics----
plots2 <- list(c())
plots2$a <- metrics2$latitude_rmse %>% 
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

plots2$b <- metrics2$latitude_rsquared %>% 
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

plots2$c <- metrics2$longitude_rmse %>% 
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

plots2$d <- metrics2$longitude_rsquared %>% 
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

plots2$e <- metrics2$building_accuracy %>% 
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

plots2$f <- metrics2$floor_accuracy %>% 
  ggplot(aes(x = metrics, y = values)) + 
  geom_col(aes(fill = metrics)) +
  geom_text(aes(fill = metrics, label = round(values, digits = 3)), 
            colour = "black") +
  coord_flip() +
  labs(x = "Metrics for each Building",
       y = "Accuracy",
       title = "Floor") +
  theme_light() +
  scale_fill_brewer(palette = "Set3") +
  theme(legend.position="none")

plots2$m <- grid.arrange(plots2$a, plots2$b, plots2$c, plots2$d, plots2$e, 
                         plots2$f, ncol = 2)
