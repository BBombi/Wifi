# The idea is to reduce the size of the train dataset, and combine it with the validation dataset as well.

# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, plotly, png, caret,
               lubridate, cluster, caTools, RColorBrewer, gridExtra, ISLR,
               gbm, caretEnsemble, parallel, doMC, randomForest, DescTools,
               import, RRF, inTrees, ggpubr, ggthemes)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)
registerDoMC(cores = detectCores())

# Loadding the data ----
train <- read.csv("datasets/trainingData.csv")
valid <- read.csv("datasets/validationData.csv")

#Summary some atributes 
summary(train[,521:529])

# Converting factor variables ----
train$BUILDINGID <- as.factor(train$BUILDINGID)
valid$BUILDINGID <- as.factor(valid$BUILDINGID)
train$PHONEID <- as.factor(train$PHONEID)
valid$PHONEID <- as.factor(valid$PHONEID)
train$USERID <- as.factor(train$USERID)
valid$USERID <- as.factor(valid$USERID)
train$SPACEID <- as.factor(train$SPACEID)
valid$SPACEID <- as.factor(valid$SPACEID)

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
train <- train[!duplicated(train[1:528]), ]
valid <- valid[!duplicated(valid[1:528]), ]

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
near0var <- nearZeroVar(train[ ,1:520], saveMetrics = TRUE, uniqueCut = 0.015)
# New datasets
trainsample <- train[ ,c(which(near0var$nzv == FALSE), 521:529)] %>% 
  filter(apply(train[ ,1:520], 1, max) < 100, apply(train[ ,1:520], 1, sum) > 0) %>%
  dplyr::group_by(BUILDINGID, FLOOR, LATITUDE, LONGITUDE, PHONEID) %>%
  dplyr::sample_n(4, replace = TRUE)
newdf <- rbind.data.frame(trainsample, valid[ ,c(which(near0var$nzv == FALSE), 
                                                 521:529)])
set.seed(637)
sample <- sample.split(newdf, SplitRatio = .80)
newtrain <- subset(newdf, sample == TRUE)
newvalid <- subset(newdf, sample == FALSE)

rm(sample, trainsample, newdf)

# Predicting Building ----
build <- list(c())
build$train <- data.frame(newtrain[,c(1:428, 432)])
build$valid <- data.frame(newvalid[,c(1:428, 432)])

# k-NN to predict Building:
build$knn <- train(BUILDINGID ~ ., 
                   data = build$train, 
                   method = "knn",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            verboseIter = TRUE),
                   metric = "Accuracy",
                   preProcess = "zv")

build$pred_knn <- predict(build$knn, newdata = build$valid)
build$conf_mat_knn <- table(build$pred_knn, build$valid$BUILDINGID)
build$accuracy_knn <- ((sum(diag(build$conf_mat_knn)))/
                         (sum(build$conf_mat_knn)))*100
# RF for Building:
tuneRF(build$train[grep("WAP", names(build$train), value=T)], 
       build$train$BUILDINGID, ntreeTry=200, stepFactor=2,
       improve=0.05, trace=TRUE, plot=T) 

build$rf <- randomForest(y = build$train$BUILDINGID,
                         x = build$train[grep("WAP", names(build$train), value=T)],
                         importance = T, method = "rf", ntree=200, mtry=5)

build$pred_rf <- predict(build$rf, newdata = build$valid)
build$conf_mat_rf <- table(build$pred_rf, build$valid$BUILDINGID)
build$accuracy_rf <- ((sum(diag(build$conf_mat_rf)))/
                        (sum(build$conf_mat_rf)))*100
# GBM for Building:
build$gbm <- train(BUILDINGID ~ ., 
                   data = build$train, 
                   method = "gbm",
                   trControl = trainControl(method = "cv",
                                            number = 2,
                                            verboseIter = TRUE),
                   metric = "Accuracy",
                   preProcess = "zv")

build$pred_gbm <- predict(build$rf, newdata = build$valid)
build$conf_mat_gbm <- table(build$pred_rf, build$valid$BUILDINGID)
build$accuracy_gbm <- ((sum(diag(build$conf_mat_rf)))/
                         (sum(build$conf_mat_rf)))*100
#Saving models:
saveRDS(build$knn, file="Models/Idea2/build_knn.rds")
saveRDS(build$rf, file="Models/Idea2/build_rf.rds")
saveRDS(build$gbm, file="Models/Idea2/build_gbm.rds")
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
                       '2',build$pred_rf))))

build$conf_mat_majority <- table(build$pred_majority, build$valid$BUILDINGID)
build$accuracy_majority <- ((sum(diag(build$conf_mat_majority)))/
                              (sum(build$conf_mat_majority)))*100

# Models for Latitude ----
LAT <- list(c())
# method = "zv" remove attributes with a near zero variance (close to the same value)
newtrain$build <- newtrain$BUILDINGID
newvalid$build <- predict(build$rf, newvalid)
LAT$train <- newtrain[ ,c(1:428,430,438)]
LAT$valid <- newvalid[ ,c(1:428,430,438)]
## k-NN model:
LAT$knn <- train(LATITUDE ~ .,
                 data = LAT$train,
                 method = "knn",
                 trControl = trainControl(method = "cv",
                                          number = 5,
                                          verboseIter = TRUE),
                 preProcess = "zv",
                 metric = "RMSE")

LAT$pred_knn <- predict(LAT$knn, newdata = LAT$valid)
LAT$error_knn <- LAT$pred_knn - LAT$valid$LATITUDE
LAT$rmse_knn <- sqrt(mean(LAT$error_knn^2))
LAT$rsquared_knn <- (1 - (sum(LAT$error_knn^2) / 
                            sum((LAT$valid$LATITUDE - mean(LAT$valid$LATITUDE)
                            )^2)))*100

## RRF model:
LAT$RRF <- train(LATITUDE ~ ., 
                 data = LAT$train,
                 method = "RRF",
                 tuneGrid = data.frame(mtry=40, 
                                       coefReg = 1,
                                       coefImp = 1),
                 trControl = trainControl(method = "cv",
                                          number = 2,
                                          verboseIter = TRUE),
                 preProcess = "zv",
                 metric = "RMSE")

LAT$pred_RRF <- predict(LAT$RRF, newdata = LAT$valid)
LAT$error_RRF <- LAT$pred_RRF - LAT$valid$LATITUDE
LAT$rmse_RRF <- sqrt(mean(LAT$error_RRF^2))
LAT$rsquared_RRF <- (1 - (sum(LAT$error_RRF^2) / 
                            sum((LAT$valid$LATITUDE - mean(LAT$valid$LATITUDE)
                            )^2)))*100

## RF model:
LAT$rf <- train(LATITUDE ~ ., 
                data = LAT$train,
                method = "rf",
                tuneGrid = data.frame(mtry=40),
                trControl = trainControl(method = "cv",
                                         number = 2,
                                         verboseIter = TRUE),
                preProcess = "zv",
                metric = "RMSE")

LAT$pred_rf<- predict(LAT$rf, newdata = LAT$valid)
LAT$error_rf <- LAT$pred_rf - LAT$valid$LATITUDE
LAT$rmse_rf <- sqrt(mean(LAT$error_rf^2))
LAT$rsquared_rf <- (1 - (sum(LAT$error_rf^2) / 
                           sum((LAT$valid$LATITUDE - mean(LAT$valid$LATITUDE)
                           )^2)))*100
#Saving models:
saveRDS(LAT$knn, file="Models/Idea2/LAT_knn.rds")
saveRDS(LAT$RRF, file="Models/Idea2/LAT_RRF.rds")
saveRDS(LAT$rf, file="Models/Idea2/LAT_rf.rds")

# Ensembling models:
LAT$pred_ensembled <- (predict(LAT$RRF, newdata = LAT$valid) +
                         predict(LAT$knn, newdata = LAT$valid) +
                         predict(LAT$rf, newdata = LAT$valid))/3
LAT$error_ensembled <- LAT$pred_ensembled - LAT$valid$LATITUDE
LAT$rmse_ensembled <- sqrt(mean(LAT$error_ensembled^2))
LAT$rsquared_ensembled <- (1 - (sum(LAT$error_ensembled^2) / 
                                  sum((LAT$valid$LATITUDE - mean(LAT$valid$LATITUDE)
                                  )^2)))*100

# Plotting the errors:
ggqqplot(LAT$error_knn, title = "Normal Q-Q plot of Longitude k-NN Errors") +
  theme_economist()
ggqqplot(LAT$error_rf, title = "Normal Q-Q plot of Longitude RF Errors") +
  theme_economist()
ggqqplot(LAT$error_RRF, title = "Normal Q-Q plot of Longitude RRF Errors") +
  theme_economist()
ggqqplot(LAT$error_ensembled, title = "Normal Q-Q plot of Longitude Ensembled Errors") +
  theme_economist()

LAT$valid$lat_errors <- LAT$valid$LATITUDE - 
  ((predict(LAT$RRF, newdata = LAT$valid) + 
      predict(LAT$rf, newdata = LAT$valid) + 
      predict(LAT$knn, newdata = LAT$valid))/3)

ggplot(LAT$valid,aes(x=lat_errors)) +
  geom_density(fill="lightblue")+
  geom_vline(xintercept = mean(LAT$valid$lat_errors), size = 1, colour="red") +
  annotate("text", x = -10, y = 0.08, colour="red",
           label = paste0("ME (",round(mean(LAT$valid$lat_errors),2),")")) +
  geom_vline(xintercept = sqrt(mean(LAT$valid$lat_errors^2)), size = 1, colour="blue") +
  annotate("text", x = 19, y = 0.07, 
           label = paste0("RMSE (",round(sqrt(mean(LON$valid$lon_errors^2)),2),
                          ")"),colour="blue") +
  xlab("Error (meters)") +
  ggtitle("Distribution of errors")+
  theme_economist()

# Methods for Longitud ----
LON <- list(c())

LON$train <- newtrain[ ,c(1:428,429,438)]
LON$valid <- newvalid[ ,c(1:428,429,438)]

## k-NN model:
LON$knn <- train(LONGITUDE ~ .,
                 data = LON$train,
                 method = "knn",
                 trControl = trainControl(method = "cv",
                                          number = 5,
                                          verboseIter = TRUE),
                 preProcess = "zv",
                 metric = "RMSE")

LON$pred_knn<- predict(LON$knn, newdata = LON$valid)
LON$error_knn <- LON$pred_knn - LON$valid$LONGITUDE
LON$rmse_knn <- sqrt(mean(LON$error_knn^2))
LON$rsquared_knn <- (1 - (sum(LON$error_knn^2) / 
                            sum((LON$valid$LONGITUDE - mean(LON$valid$LONGITUDE)
                            )^2)))*100

## RRF model:
LON$RRF <- train(LONGITUDE ~ ., 
                 data = LON$train,
                 method = "RRF",
                 tuneGrid = data.frame(mtry=40, 
                                       coefReg = 1,
                                       coefImp = 1),
                 trControl = trainControl(method = "cv",
                                          number = 2,
                                          verboseIter = TRUE),
                 preProcess = "zv",
                 metric = "RMSE")

LON$pred_RRF<- predict(LON$RRF, newdata = LON$valid)
LON$error_RRF <- LON$pred_RRF - LON$valid$LONGITUDE
LON$rmse_RRF <- sqrt(mean(LON$error_RRF^2))
LON$rsquared_RRF <- (1 - (sum(LON$error_RRF^2) / 
                            sum((LON$valid$LONGITUDE - mean(LON$valid$LONGITUDE)
                            )^2)))*100

## RF model:
LON$rf <- train(LONGITUDE ~ ., 
                data = LON$train,
                method = "rf",
                tuneGrid = data.frame(mtry=40),
                trControl = trainControl(method = "cv",
                                         number = 2,
                                         verboseIter = TRUE),
                preProcess = "zv",
                metric = "RMSE")

LON$pred_rf<- predict(LON$rf, newdata = LON$valid)
LON$error_rf <- LON$pred_rf - LON$valid$LONGITUDE
LON$rmse_rf <- sqrt(mean(LON$error_rf^2))
LON$rsquared_rf <- (1 - (sum(LON$error_rf^2) / 
                           sum((LON$valid$LONGITUDE - mean(LON$valid$LONGITUDE)
                           )^2)))*100
#Saving models:
saveRDS(LON$knn, file="Models/Idea2/LON_knn.rds")
saveRDS(LON$RRF, file="Models/Idea2/LON_RRF.rds")
saveRDS(LON$rf, file="Models/Idea2/LON_rf.rds")

# Ensembling models:
LON$pred_ensembled <- (predict(LON$RRF, newdata = LON$valid) +
                         predict(LON$rf, newdata = LON$valid) +
                         predict(LON$knn, newdata = LON$valid))/3
LON$error_ensembled <- LON$pred_ensembled - LON$valid$LONGITUDE
LON$rmse_ensembled <- sqrt(mean(LON$error_ensembled^2))
LON$rsquared_ensembled <- (1 - (sum(LON$error_ensembled^2) / 
                                  sum((LON$valid$LONGITUDE - mean(LON$valid$LONGITUDE)
                                  )^2)))*100
# Plotting the errors:
ggqqplot(LON$error_knn, title = "Normal Q-Q plot of Longitude k-NN Errors") +
  theme_economist()
ggqqplot(LON$error_rf, title = "Normal Q-Q plot of Longitude RF Errors") +
  theme_economist()
ggqqplot(LON$error_RRF, title = "Normal Q-Q plot of Longitude RRF Errors") +
  theme_economist()
ggqqplot(LON$error_ensembled, title = "Normal Q-Q plot of Longitude Ensembled Errors") +
  theme_economist()

LON$valid$lon_errors <- LON$valid$LONGITUDE - ((predict(LON$RRF, newdata = LON$valid) +
                                                  predict(LON$rf, newdata = LON$valid) +
                                                  predict(LON$knn, newdata = LON$valid))/3)
ggplot(LON$valid,aes(x=lon_errors)) +
  geom_density(fill="lightblue")+
  geom_vline(xintercept = mean(LON$valid$lon_errors), size = 1, colour="red") +
  annotate("text", x = -11, y = 0.08, colour="red",
           label = paste0("ME (",round(mean(LON$valid$lon_errors),2),")")) +
  geom_vline(xintercept = sqrt(mean(LON$valid$lon_errors^2)), size = 1, colour="blue") +
  annotate("text", x = 21, y = 0.07, 
           label = paste0("RMSE (",round(sqrt(mean(LON$valid$lon_errors^2)),2),
                          ")"),colour="blue") +
  xlab("Error (meters)") +
  ggtitle("Distribution of errors")+
  theme_economist()
# Methods for FLOOR ----
FL <- list(c())
# k-NN:
for (i in 0:2) {
  trainnew <- newtrain[,c(1:428,431,438)] %>% filter(build == i)
  validnew <- newvalid[,c(1:428,431,438)] %>% filter(build == i)
  trainnew$FLOOR <- as.factor(trainnew$FLOOR)
  validnew$FLOOR <- as.factor(validnew$FLOOR)
  FL[[paste0("knn_",i)]] <- train(FLOOR ~ .,
                                  data = trainnew,
                                  method = "knn",
                                  trControl = trainControl(method = "cv",
                                                           number = 3,
                                                           verboseIter = TRUE),
                                  preProcess = "zv",
                                  metric = "Accuracy")
  
  FL[[paste0("pred_knn_",i)]] <- predict(FL[[paste0("knn_",i)]], validnew)
  FL[[paste0("conf_mat_knn_",i)]] <- table(FL[[paste0("pred_knn_",i)]], 
                                           validnew$FLOOR)
  FL[[paste0("accuracy_knn_",i)]] <- ((sum(diag(FL[[paste0("conf_mat_knn_",i)]])))/
                                        (sum(FL[[paste0("conf_mat_knn_",i)]])))*100
  #Saving Models:
  saveRDS(FL[[paste0("knn_",i)]], file=paste0("Models/Idea2/Floor_knn_",i,".rds"))
}

# RF for FLOOR:
for (i in 0:2) {
  trainnew <- newtrain[,c(1:428,431,438)] %>% filter(build == i)
  validnew <- newvalid[,c(1:428,431,438)] %>% filter(build == i)
  trainnew$FLOOR <- as.factor(trainnew$FLOOR)
  validnew$FLOOR <- as.factor(validnew$FLOOR)
  FL[[paste0("rf_",i)]] <- train(FLOOR ~ .,
                                 data = trainnew,
                                 method = "rf",
                                 trControl = trainControl(method = "cv",
                                                          number = 3,
                                                          verboseIter = TRUE),
                                 preProcess = "zv",
                                 metric = "Accuracy")
  
  FL[[paste0("pred_rf_",i)]] <- predict(FL[[paste0("rf_",i)]], validnew)
  FL[[paste0("conf_mat_rf_",i)]] <- table(FL[[paste0("pred_rf_",i)]], 
                                          validnew$FLOOR)
  FL[[paste0("accuracy_rf_",i)]] <- ((sum(diag(FL[[paste0("conf_mat_rf_",i)]])))/
                                       (sum(FL[[paste0("conf_mat_rf_",i)]])))*100
  #Saving Models:
  saveRDS(FL[[paste0("rf_",i)]], file=paste0("Models/Idea2/Floor_rf_",i,".rds"))
}

# RRF for FLOOR:
for (i in 0:2) {
  trainnew <- newtrain[,c(1:428,431,438)] %>% filter(build == i)
  validnew <- newvalid[,c(1:428,431,438)] %>% filter(build == i)
  trainnew$FLOOR <- as.factor(trainnew$FLOOR)
  validnew$FLOOR <- as.factor(validnew$FLOOR)
  FL[[paste0("RRF_",i)]] <- train(FLOOR ~ .,
                                  data = trainnew,
                                  method = "RRF",
                                  trControl = trainControl(method = "cv",
                                                           number = 3,
                                                           verboseIter = TRUE),
                                  preProcess = "zv",
                                  metric = "Accuracy")

  FL[[paste0("pred_RRF_",i)]] <- predict(FL[[paste0("RRF_",i)]], validnew)
  FL[[paste0("conf_mat_RRF_",i)]] <- table(FL[[paste0("pred_RRF_",i)]], 
                                           validnew$FLOOR)
  FL[[paste0("accuracy_RRF_",i)]] <- ((sum(diag(FL[[paste0("conf_mat_RRF_",i)]])))/
                                        (sum(FL[[paste0("conf_mat_RRF_",i)]])))*100
  
  #Saving Models:
  saveRDS(FL[[paste0("RRF_",i)]], file=paste0("Models/Idea2/Floor_RRF_",i,".rds"))
}

# GBM for FLOOR:
for (i in 0:2) {
  trainnew <- newtrain[,c(1:428,431,438)] %>% filter(build == i)
  validnew <- newvalid[,c(1:428,431,438)] %>% filter(build == i)
  trainnew$FLOOR <- as.factor(trainnew$FLOOR)
  validnew$FLOOR <- as.factor(validnew$FLOOR)
  FL[[paste0("gbm_",i)]] <- train(FLOOR ~ .,
                                  data = trainnew,
                                  method = "gbm",
                                  trControl = trainControl(method = "cv",
                                                           number = 3,
                                                           verboseIter = TRUE),
                                  preProcess = "zv",
                                  metric = "Accuracy")
  
  FL[[paste0("pred_gbm_",i)]] <- predict(FL[[paste0("gbm_",i)]], validnew)
  FL[[paste0("conf_mat_gbm_",i)]] <- table(FL[[paste0("pred_gbm_",i)]], 
                                           validnew$FLOOR)
  FL[[paste0("accuracy_gbm_",i)]] <- ((sum(diag(FL[[paste0("conf_mat_gbm_",i)]])))/
                                        (sum(FL[[paste0("conf_mat_gbm_",i)]])))*100
  #Saving Models:
  saveRDS(FL[[paste0("gbm_",i)]], file=paste0("Models/Idea2/Floor_gbm_",i,".rds"))
  
  #Majority vote:
  FL[[paste0("pred_mv_",i)]] <- as.factor(
    ifelse(FL[[paste0("pred_knn_",i)]]=='0' & 
             FL[[paste0("pred_RRF_",i)]]=='0' | 
             FL[[paste0("pred_knn_",i)]]=='0' & 
             FL[[paste0("pred_gbm_",i)]]=='0' | 
             FL[[paste0("pred_RRF_",i)]]=='0' & 
             FL[[paste0("pred_gbm_",i)]]=='0','0',
           ifelse(FL[[paste0("pred_knn_",i)]]=='1' & 
                    FL[[paste0("pred_RRF_",i)]]=='1' | 
                    FL[[paste0("pred_knn_",i)]]=='1' & 
                    FL[[paste0("pred_gbm_",i)]]=='1' | 
                    FL[[paste0("pred_RRF_",i)]]=='1' & 
                    FL[[paste0("pred_gbm_",i)]]=='1','1',
                  ifelse(FL[[paste0("pred_knn_",i)]]=='2' & 
                           FL[[paste0("pred_RRF_",i)]]=='2' |
                           FL[[paste0("pred_knn_",i)]]=='2' & 
                           FL[[paste0("pred_gbm_",i)]]=='2' | 
                           FL[[paste0("pred_RRF_",i)]]=='2' & 
                           FL[[paste0("pred_gbm_",i)]]=='2',
                         '2',ifelse(FL[[paste0("pred_knn_",i)]]=='3' & 
                                      FL[[paste0("pred_RRF_",i)]]=='3' |
                                      FL[[paste0("pred_knn_",i)]]=='3' & 
                                      FL[[paste0("pred_gbm_",i)]]=='3' | 
                                      FL[[paste0("pred_RRF_",i)]]=='3' & 
                                      FL[[paste0("pred_gbm_",i)]]=='3',
                                    '3',ifelse(FL[[paste0("pred_knn_",i)]]=='4' & 
                                                 FL[[paste0("pred_RRF_",i)]]=='4' |
                                                 FL[[paste0("pred_knn_",i)]]=='4' & 
                                                 FL[[paste0("pred_gbm_",i)]]=='4' | 
                                                 FL[[paste0("pred_RRF_",i)]]=='4' & 
                                                 FL[[paste0("pred_gbm_",i)]]=='4',
                                               '4',FL[[paste0("pred_rf_",i)]]
                                    ))))))
  
  FL[[paste0("conf_mat_mv_",i)]] <- table(FL[[paste0("pred_mv_",i)]], 
                                          validnew$FLOOR)
  FL[[paste0("accuracy_mv_",i)]] <- ((sum(diag(FL[[paste0("conf_mat_mv_",i)]])))/
                                       (sum(FL[[paste0("conf_mat_mv_",i)]])))*100
}


# Creating data frames to compare models ----
metrics2 <- list(c())
metrics2$building_accuracy <- data.frame(metrics = c("RF", "k-NN", "GBM", 
                                                    "Majority Vote"), 
                                        values = c(build$accuracy_rf, 
                                                   build$accuracy_knn, 
                                                   build$accuracy_gbm, 
                                                   build$accuracy_majority))

metrics2$latitude_rmse <- data.frame(metrics = c("k-NN", "RF", "RRF", 
                                                "Ensembled"),
                                    values = c(LAT$rmse_knn, LAT$rmse_rf, 
                                               LAT$rmse_RRF, LAT$rmse_ensembled))

metrics2$latitude_rsquared <- data.frame(metrics = c("k-NN", "RF", "RRF", 
                                                     "Ensembled"),
                                         values = c(LAT$rsquared_knn, LAT$rsquared_rf, 
                                                    LAT$rsquared_RRF, 
                                                    LAT$rsquared_ensembled))

metrics2$longitude_rmse <- data.frame(metrics = c("k-NN", "RF", "RRF", 
                                                  "Ensembled"),
                                      values = c(LON$rmse_knn, LON$rmse_rf, 
                                                 LON$rmse_RRF, LON$rmse_ensembled))

metrics2$longitude_rsquared <- data.frame(metrics = c("k-NN", "RF", "RRF", 
                                                      "Ensembled"),
                                          values = c(LON$rsquared_knn, LON$rsquared_rf, 
                                                     LON$rsquared_RRF, 
                                                     LON$rsquared_ensembled))

metrics2$floor_accuracy <- data.frame(metrics = c("0_k-NN", "0_RF", "1_k-NN", 
                                                 "1_RF", "2_k-NN", "2_RF",
                                                 "0_GBM", "1_GBM", "2_GBM",
                                                 "0_Majority", "1_Majority",
                                                 "2_Majority", "0_RRF", 
                                                 "1_RRF", "2_RRF"),
                                     values = c(FL$accuracy_knn_0, FL$accuracy_rf_0, 
                                                FL$accuracy_knn_1, FL$accuracy_rf_1,
                                                FL$accuracy_knn_2, FL$accuracy_rf_2,
                                                FL$accuracy_gbm_0, FL$accuracy_gbm_1,
                                                FL$accuracy_gbm_2, FL$accuracy_mv_0,
                                                FL$accuracy_mv_1, FL$accuracy_mv_2,
                                                FL$accuracy_RRF_0, FL$accuracy_RRF_1,
                                                FL$accuracy_RRF_2))

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
  labs(x = "Method used",
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
  labs(x = "Method used",
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
  labs(x = "Method used",
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
  labs(x = "Metrics and method per each building",
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
  scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(15)) +
  theme(legend.position="none")

plots2$m <- grid.arrange(plots2$a, plots2$b, plots2$c, plots2$d, plots2$e, 
                         plots2$f, ncol = 2)

# Reloading the models ----
FL$gbm_0 <- readRDS("Models/Idea2/Floor_gbm_0.rds")
FL$gbm_1 <- readRDS("Models/Idea2/Floor_gbm_1.rds")
FL$gbm_2 <- readRDS("Models/Idea2/Floor_gbm_2.rds")
FL$knn_0 <- readRDS("Models/Idea2/Floor_knn_0.rds")
FL$knn_1 <- readRDS("Models/Idea2/Floor_knn_1.rds")
FL$knn_2 <- readRDS("Models/Idea2/Floor_knn_2.rds")
FL$rf_0 <- readRDS("Models/Idea2/Floor_rf_0.rds")
FL$rf_1 <- readRDS("Models/Idea2/Floor_rf_1.rds")
FL$rf_2 <- readRDS("Models/Idea2/Floor_rf_2.rds")
FL$RRF_0 <- readRDS("Models/Idea2/Floor_RRF_0.rds")
FL$RRF_1 <- readRDS("Models/Idea2/Floor_RRF_1.rds")
FL$RRF_2 <- readRDS("Models/Idea2/Floor_RRF_2.rds")
build$gbm <- readRDS("Models/Idea2/build_gbm.rds")
build$knn <- readRDS("Models/Idea2/build_knn.rds")
build$rf <- readRDS("Models/Idea2/build_rf.rds")
LAT$knn <- readRDS("Models/Idea2/LAT_knn.rds")
LAT$rf <- readRDS("Models/Idea2/LAT_rf.rds")
LAT$RRF <- readRDS("Models/Idea2/LAT_RRF.rds")
LON$knn <- readRDS("Models/Idea2/LON_knn.rds")
LON$rf <- readRDS("Models/Idea2/LON_rf.rds")
LON$RRF <- readRDS("Models/Idea2/LON_RRF.rds")

# Predicting over the TEST set----
test <- read.csv("datasets/testData.csv")

test <- cbind(apply(test[1:520],2, function(x) 10^(x/10)*100), 
                 test[521:529])
test <- cbind(apply(test[1:520], c(1,2), function(y) 
  ifelse(y == 10^12, y <- 0, y <- y)), test[521:529])

test$ID <- c(1:nrow(test))

# Building
build$pred_rf <- predict(build$rf, newdata = test)
build$pred_knn <- predict(build$knn, newdata = test)
build$pred_gbm <- predict(build$gbm, newdata = test)

test$build <- as.factor(
  ifelse(build$pred_knn=='0' & build$pred_rf=='0' | 
           build$pred_knn=='0' & build$pred_gbm=='0' | 
           build$pred_rf=='0' & build$pred_gbm=='0','0',
         ifelse(build$pred_knn=='1' & build$pred_rf=='1' | 
                  build$pred_knn=='1' & build$pred_gbm=='1' | 
                  build$pred_rf=='1' & build$pred_gbm=='1','1',
                ifelse(build$pred_knn=='2' & build$pred_rf=='2' |
                         build$pred_knn=='2' & build$pred_gbm=='2' | 
                         build$pred_rf=='2' & build$pred_gbm=='2',
                       '2',build$pred_rf))))
# Latitude:
test$LATITUDE <- (predict(LAT$knn, newdata = test) +
                       predict(LAT$RRF, newdata = test) +
                       predict(LAT$rf, newdata = test))/3
# Longitude:
test$LONGITUDE <- (predict(LON$RRF, newdata = test) +
                        predict(LON$rf, newdata = test) +
                        predict(LON$knn, newdata = test))/3
# Floor:
test0 <- test %>% filter(build == 0)
test1 <- test %>% filter(build == 1)
test2 <- test %>% filter(build == 2)
test0$FLOOR <- predict(FL$gbm_0, test0)
test1$FLOOR <- predict(FL$knn_1, test1)
test2$FLOOR <- predict(FL$gbm_2, test2)
test <- rbind(test0, test1, test2)
rm(test0, test1, test2)
test <- test[order(test$ID), ]

write_csv(test[c(522,521,523)], "predictions/BorjaBombi_Idea2.csv")

plot_ly(test, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~build, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Training Data")
