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
source("cleaning.R")

dataFile <- read.csv("data/train_cleaned.csv", header = T, row.names = 1)

test <- read.csv("data/test_cleaned.csv", header = T, row.names = 1)

datos <- tibble::as_tibble(dataFile)
test <- tibble::as_tibble(test)

datos$date_recorded <- as.Date(datos$date_recorded)
test$date_recorded <- as.Date(test$date_recorded) 

model <- svm(status_group~., data=datos, 
             method="C-classification", kernel="radial", 
             gamma=0.1, cost=10)


x <- predict(model, test)
table(x)
