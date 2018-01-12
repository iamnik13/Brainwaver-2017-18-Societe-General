setwd("D:\\R Projects\\Hacker earth Brainwaves")
library(readr)
df_train <- read.csv("train.csv")
df_test <- read.csv("test.csv")
view(train)
sapply(df_train, function(x) sum(is.na(x)))
sum(is.na(df_train)) / (nrow(df_train) *ncol(df_train))

library(base)
colnames(df_train)[apply(is.na(df_train), 2, any)]

sapply(df_train,class)

df_train <- df_train[-13]
df_train <- df_train[-15]
df_train <- df_train[-15]

df_test <- df_test[-13]
df_test <- df_test[-15]
df_test <- df_test[-15]


#Cheching Missing Values
library(Amelia)
missmap(df_train, main="Missing Vs Observed")

missmap(df_test, main="Missing vs Observed")


age <- c(df_train$libor_rate, df_test$libor_rate)
avg.age <- mean(age, na.rm=T)
df_train$libor_rate[is.na(df_train$libor_rate)] <- avg.age
df_test$libor_rate[is.na(df_test$libor_rate)] <- avg.age

df_train_numeric <- df_train[ ,sapply(df_train, is.numeric)]
df_train_cat <- df_train[ ,!sapply(df_train, is.numeric)]

df_test_numeric <- df_test[ ,sapply(df_test, is.numeric)]
df_test_cat <- df_test[ ,!sapply(df_test, is.numeric)]

#Correlation
library(corrplot)
correlation <- cor(df_train_numeric[ , ])
cor(df_train_numeric, df_train_numeric$return)
corrplot(correlation, method = "circle")


library(scales)
library(ggplot2)
ggplot(df_train_numeric, aes(x=return)) + geom_histogram(col = 'white') + 
  theme_light() +scale_x_continuous(labels = comma)

#LInear Regression
linear <- lm(return ~ ., data = df_train_numeric)
prediction <- predict(linear, df_test_numeric)
summary(linear)
plot(linear)


# Stepwise Regression
library(MASS)
step <- stepAIC(linear, direction = "both")
step$anova


df_train1 <- df_train1[-1]
df_test1 <- df_test1[-1]

#Linear Regression Model
linear <- lm(return ~ ., data = df_train1, subset = (1:length(return)!=1666))
#removing outlier point due to which we get good prediction

df_test_numeric$return <- predict(linear, df_test1)
plot(linear)
testout<- df_test_cat$portfolio_id
testout <- data.frame(testout)
testout$return <- df_test_numeric$return
colnames(testout) <- c("portfolio_id", "return")
testout
write.csv(testout, "submission.csv", row.names = F)

#Decision tree

