# Implement the "Idea2" regarding the ensemble of datasets, to have a better trainset.
# The idea is to predict first the building, then create a new feature with this prediction.
# Then we can split both datasets with the predictions, one to train the model, and anotherone
# to test it. We should be able to predict Latitude Longitude and Floor with those models.

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

# Predicting the actual dataset----
train2$rf <- predict(build2$rf, train2)
build2$conf_mtrx_rf <- table(train2$rf, train2$BUILDINGID)
build2$acc_rf <- ((sum(diag(build2$conf_mtrx_rf)))/sum(build2$conf_mtrx_rf))*100
valid2$rf <- predict(build2$rf, valid2)

train2$knn <- predict(build2$knn, train2)
build2$conf_mtrx_knn <- table(train2$knn, train2$BUILDINGID)
build2$acc_knn <- ((sum(diag(build2$conf_mtrx_knn)))/sum(build2$conf_mtrx_knn))*100
valid2$knn <- predict(build2$knn, valid2)

train2$gbm <- predict(build2$gbm, train2)
build2$conf_mtrx_gbm <- table(train2$gbm, train2$BUILDINGID)
build2$acc_gbm <- ((sum(diag(build2$conf_mtrx_gbm)))/sum(build2$conf_mtrx_gbm))*100
valid2$gbm <- predict(build2$gbm, valid2)

train2$majority_vote <- 
  apply(train2, 1, 
        FUN = function(x){ifelse(x[438]=='0' & x[437]=='0' | 
                                   x[438]=='0' & x[439]=='0' | 
                                   x[437]=='0' & x[439]=='0','0',
                                 ifelse(x[438]=='1' & x[437]=='1' | 
                                          x[438]=='1' & x[439]=='1' | 
                                          x[437]=='1' & x[439]=='1','1',
                                        ifelse(x[438]=='2' & x[437]=='2' |
                                                 x[438]=='2' & x[439]=='2' | 
                                                 x[437]=='2' & x[439]=='2',
                                               '2',x[439])))})

build2$conf_mtrx_mv <- table(train2$majority_vote, train2$BUILDINGID)
build2$acc_mv <- ((sum(diag(build2$conf_mtrx_mv)))/sum(build2$conf_mtrx_mv))*100

valid2$majority_vote <- 
  apply(valid2, 1, 
        FUN = function(x){ifelse(x[438]=='0' & x[437]=='0' | 
                                   x[438]=='0' & x[439]=='0' | 
                                   x[437]=='0' & x[439]=='0','0',
                                 ifelse(x[438]=='1' & x[437]=='1' | 
                                          x[438]=='1' & x[439]=='1' | 
                                          x[437]=='1' & x[439]=='1','1',
                                        ifelse(x[438]=='2' & x[437]=='2' |
                                                 x[438]=='2' & x[439]=='2' | 
                                                 x[437]=='2' & x[439]=='2',
                                               '2',x[439])))})

train2$compare_building <- ifelse(train2$majority_vote == 
                                   train2$BUILDINGID, "TRUE",
                                 "FALSE")
train2[ ,429:441] %>% dplyr::filter(compare_building == "FALSE")

# Plotting ERRORS with plotly ----
plot_ly(train2, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~majority_vote, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Training Data")

plot_ly(valid2, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~majority_vote, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Validation Data")

# Predicting Latitude, Longitude and Floow ----
train2$building <- train2$BUILDINGID
valid2$building <- valid2$BUILDINGID
methods <- c("rf", "knn", "RRF")
control <- c()
grid <- c()
LAT <- list(c())
LON <- list(c())
FL <- list(c())

for (i in methods) {
  ifelse(i == "rf" | i == "RRF", control <- trainControl(method = "cv",
                                            number = 3,
                                            verboseIter = TRUE),
         control <- trainControl(method = "cv",
                                 number = 5,
                                 verboseIter = TRUE))
  ifelse(i=="rf" | i == "RRF",grid <- data.frame(mtry=c(36,37,39)),
         grid <- expand.grid(k=c(3,5,7,9)))
  for (j in 0:2) {
    trainnew <- train2[ ,c(1:428, 430, 442)] %>% dplyr::filter(building == j)
    validnew <- valid2[ ,c(1:428, 430, 441)] %>% dplyr::filter(building == j)
    
    LAT[[paste0(i,"_",j)]] <- train(LATITUDE ~ ., 
                                          data = trainnew, 
                                          method = i,
                                          tuneGrid = grid,
                                          trControl = control,
                                          metric = "RMSE",
                                          preProcess = "zv")
    
    LAT[[paste0("pred_",i,"_",j)]] <- 
      predict(LAT[[paste0(i,"_",j)]], newdata = validnew)
    LAT[[paste0("error_",i,"_",j)]] <- LAT[[paste0("pred_",i,"_",j)]] - 
      validnew$LATITUDE
    LAT[[paste0("rmse_",i,"_",j)]] <- sqrt(mean(LAT[[paste0("error_",i,"_",j)]]^2))
    LAT[[paste0("rsquared_",i,"_",j)]] <- 
      (1 - (sum(LAT[[paste0("error_",i,"_",j)]]^2) 
            / sum((validnew$LATITUDE - mean(validnew$LATITUDE))^2)))*100
    
    trainnew <- train2[ ,c(1:428, 429, 442)] %>% dplyr::filter(building == j)
    validnew <- valid2[ ,c(1:428, 429, 441)] %>% dplyr::filter(building == j)
    
    LON[[paste0(i,"_",j)]] <- train(LONGITUDE ~ ., 
                                          data = trainnew, 
                                          method = i,
                                          tuneGrid = grid,
                                          trControl = control,
                                          metric = "RMSE",
                                          preProcess = "zv")
    
    LON[[paste0("pred_",i,"_",j)]] <- 
      predict(LON[[paste0(i,"_",j)]], newdata = validnew)
    LON[[paste0("error_",i,"_",j)]] <- LON[[paste0("pred_",i,"_",j)]] - 
      validnew$LONGITUDE
    LON[[paste0("rmse_",i,"_",j)]] <- sqrt(mean(LON[[paste0("error_",i,"_",j)]]^2))
    LON[[paste0("rsquared_",i,"_",j)]] <- 
      (1 - (sum(LON[[paste0("error_",i,"_",j)]]^2) 
            / sum((validnew$LONGITUDE - mean(validnew$LONGITUDE))^2)))*100
    
    trainnew <- train2[ ,c(1:428, 431, 442)] %>% dplyr::filter(building == j)
    trainnew$FLOOR <- as.factor(trainnew$FLOOR)
    validnew <- valid2[ ,c(1:428, 431, 441)] %>% dplyr::filter(building == j)
    validnew$FLOOR <- as.factor(validnew$FLOOR)
    
    FL[[paste0(i,"_",j)]] <- train(FLOOR ~ ., 
                          data = trainnew,
                          method = i,
                          trControl = control,
                          metric = "Accuracy",
                          preProcess = "zv")
    
    FL[[paste0("pred_",i,"_",j)]] <- predict(FL[[paste0(i,"_",j)]], 
                                             newdata = validnew)
    FL[[paste0("conf_mat_",i,"_",j)]] <- table(FL[[paste0("pred_",i,"_",j)]], 
                                   validnew$FLOOR)
    FL[[paste0("accuracy_",i,"_",j)]] <- ((sum(diag(FL[[paste0("conf_mat_",i,"_",j)]])))/
                                (sum(FL[[paste0("conf_mat_",i,"_",j)]])))*100
  }
}
validnew$rf <- predict(FL$rf_2, validnew)
validnew$knn <- predict(FL$knn_2, validnew)
validnew$gbm <- predict(FL$gbm_2, validnew)
validnew$mv <- 
  apply(validnew, 1, 
        FUN = function(x){ifelse(x[432]=='0' & x[431]=='0' | 
                                   x[432]=='0' & x[433]=='0' | 
                                   x[431]=='0' & x[433]=='0','0',
                                 ifelse(x[432]=='1' & x[431]=='1' | 
                                          x[432]=='1' & x[433]=='1' | 
                                          x[431]=='1' & x[433]=='1','1',
                                        ifelse(x[432]=='2' & x[431]=='2' |
                                                 x[432]=='2' & x[433]=='2' | 
                                                 x[431]=='2' & x[433]=='2',
                                               '2',x[431])))})
x <- table(validnew$mv, validnew$FLOOR)
FL$conf_mat_gbm_2
