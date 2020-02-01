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

dataFile <- read.csv("./data/train.csv", header = T, sep = ",", na.strings = c(".", "NA", "?", ""))

datos <- tibble::as_tibble(dataFile)

# lectura etiquetas
etiquetas <- tibble::as_tibble(read.csv("labels.csv", header = T, sep = ","))

# union fina de ambos datasets

datos <- datos %>% left_join(etiquetas, by = "id")

# summary
summary(datos)
summary(datos$id)

# funcion similar
str(datos$id)
