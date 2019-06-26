#Idea4 is a combination between all 3 previous ideas.
# For Floor predictions, we'll be using "Majority vote" method.
# For Latitude and Longitude, we'll be calculating the mean of all previous 3 predictions.

test <- read.csv("datasets/testData.csv")
Idea1 <- read.csv("predictions/BorjaBombi_Idea1.csv")
Idea2 <- read.csv("predictions/BorjaBombi_Idea2.csv")
Idea3 <- read.csv("predictions/BorjaBombi_Idea3.csv")

# Floor
test$FLOOR <- as.factor(
  ifelse(Idea1$FLOOR == "0" & Idea2$FLOOR == "0" |
           Idea1$FLOOR == "0" & Idea3$FLOOR == "0" |
           Idea2$FLOOR == "0" & Idea3$FLOOR == "0", "0",
         ifelse(Idea1$FLOOR == "1" & Idea2$FLOOR == "1" |
                  Idea1$FLOOR == "1" & Idea3$FLOOR == "1" |
                  Idea2$FLOOR == "1" & Idea3$FLOOR == "1", "1",
                ifelse(Idea1$FLOOR == "2" & Idea2$FLOOR == "2" |
                         Idea1$FLOOR == "2" & Idea3$FLOOR == "2" |
                         Idea2$FLOOR == "2" & Idea3$FLOOR == "2", "2",
                       ifelse(Idea1$FLOOR == "3" & Idea2$FLOOR == "3" |
                                Idea1$FLOOR == "3" & Idea3$FLOOR == "3" |
                                Idea2$FLOOR == "3" & Idea3$FLOOR == "3", "3",
                              ifelse(Idea1$FLOOR == "4" & Idea2$FLOOR == "4" |
                                       Idea1$FLOOR == "4" & Idea3$FLOOR == "4" |
                                       Idea2$FLOOR == "4" & Idea3$FLOOR == "4", "4",
                                     Idea3$FLOOR))))))

# Latitude:
test$LATITUDE <- (Idea1$LATITUDE + Idea2$LATITUDE + Idea3$LATITUDE)/3

# Longitude:
test$LONGITUDE <- (Idea1$LONGITUDE + Idea2$LONGITUDE + Idea3$LONGITUDE)/3

rm(Idea1, Idea2, Idea3)

write_csv(test[c(522,521,523)], "predictions/BorjaBombi_Idea4.csv")

plot_ly(test, x = ~LATITUDE, y = ~LONGITUDE, z = ~FLOOR, 
        color = ~build, colors = c("#BF382A", "#1ABC9C", "#0C4B8E")) %>% 
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Latitude'),
                                        yaxis = list(title = 'Longitude'),
                                        zaxis = list(title = 'Floor')),
                           title = "Training Data")
