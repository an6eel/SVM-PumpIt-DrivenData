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

dataFile <- read.csv("data/train_cleaned.csv", header = T, row.names = 1)

test <- read.csv("data/test_cleaned.csv", header = T, row.names = 1)

datos <- tibble::as_tibble(dataFile)
test <- tibble::as_tibble(test)

datos$date_recorded <- as.Date(datos$date_recorded)
test$date_recorded <- as.Date(test$date_recorded) 

model <- train(status_group ~ ., data = datos[1:10000,], method = "knn")

x <- predict(model, test)
test$status_group <- x
submit <- test %>% select(id, status_group)
write.csv(submit, file = "data/submit.csv", row.names = F)
