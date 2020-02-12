#################################################################
#                                                               #
# Preprocessing                                                 #
#                                                               #
# Data Mining: Preprocessing and classification UGR             #
#                                                               #
# Ángel García Malagón                                          #
#                                                               #
#################################################################

library(tibble)
library(dplyr)
library(tidyverse)
library(caret)
library(leaflet)
library(e1071)   

dataFile <- read.csv("data/train_pca.csv", header = T)

test <- read.csv("data/test_pca.csv", header = T)

datos <- tibble::as_tibble(dataFile)
test <- tibble::as_tibble(test)

datos$date_recorded <- as.Date(datos$date_recorded)
test$date_recorded <- as.Date(test$date_recorded) 

model <- svm(status_group~., data=datos, 
             method="C-classification", kernel="radial", 
             gamma=0.1, cost=10)


x <- predict(model, test)
test <- read.csv("data/test.csv", header = T)
test$status_group <- x
submit <- test %>% select(id, status_group)
write.csv(submit, file = "data/submit.csv", row.names = F)
