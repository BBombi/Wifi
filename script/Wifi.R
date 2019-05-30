# Settings -----
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}
pacman::p_load(ggplot2, rstudioapi, plyr, purrr, readr, caret)

current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)

# Loadding the data ----

trainingData <- read.csv("datasets/trainingData.csv")
validationData <- read.csv("datasets/validationData.csv")

summary(trainingData$LONGITUDE)
summary(trainingData$LATITUDE)

trainingData$BFS <- paste0("B", trainingData$BUILDINGID, "F", 
                           trainingData$FLOOR, "S", trainingData$SPACEID)

x <- unique(trainingData$BFS)
TrainingList <- c()
for (i in x) {
  TrainingList[[i]] <- trainingData %>% dplyr::filter(BFS == i)
}

