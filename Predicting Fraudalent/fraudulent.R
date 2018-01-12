library(readr)
df_train_1 <- read.csv("train.csv")
df_test_1 <- read.csv("test.csv")
summary(df_train_1)
str(df_train_1)
view(df_train_1)
sapply(df_train_1, function(x) sum(is.na(x)))

library(Amelia)
missmap(df_train_1, main = "Missing vs Observed")

df_train_numeric <- df_train_1[ ,sapply(df_train_1, is.numeric)]
df_train_cat <- df_train_1[ ,!sapply(df_train_1, is.numeric)]

df_test_numeric <- df_test_1[ ,sapply(df_test_1, is.numeric)]
df_test_cat <- df_test_1[ ,!sapply(df_test_1, is.numeric)]

levels(df_train_1$cat_var_1)


df_train_1[0:3, ]
train_id <- df_train_1['transaction_id']
test_id <- df_test_1['transaction_id']
train_target <- df_train_1['target']

table(df_train_1$target)
df_train_1$cat_var_1[df_train_1$cat_var_1==""] <- "gf"
df_train_1$cat_var_1[df_train_1$cat_var_2==""] <- "ce"

max(table(df_train_1$cat_var_3))
df_train_1$cat_var_3[df_train_1$cat_var_3==""] <- "yv"
table(df_train_1$cat_var_4)
df_train_1$cat_var_4[df_train_1$cat_var_4==""] <- "tn"
table(df_train_1$cat_var_5)
table(df_train_1$cat_var_6)
table(df_train_1$cat_var_7)
max(table(df_train_1$cat_var_8))
df_train_1$cat_var_8[df_train_1$cat_var_8==""] <- "dn"
table(df_train_1$cat_var_9)
table(df_train_1$cat_var_10)
table(df_train_1$cat_var_11)
table(df_train_1$cat_var_12)
table(df_train_1$cat_var_13)
table(df_train_1$cat_var_14)
table(df_train_1$cat_var_15)
table(df_train_1$cat_var_16)
table(df_train_1$cat_var_17)
table(df_train_1$cat_var_18)


df_test_cat$transaction_id <- NULL
df_test_1$transaction_id <- NULL
df_test_1$cat_var_1 <- NULL
df_test_1$cat_var_2 <- NULL
df_test_1$cat_var_3 <- NULL
df_test_1$cat_var_6 <- NULL
df_test_1$cat_var_7 <- NULL
df_test_1$cat_var_8 <- NULL
df_test_1$cat_var_9 <- NULL
df_test_1$cat_var_10 <- NULL
df_test_1$cat_var_11 <- NULL
df_test_1$cat_var_12 <- NULL
df_test_1$cat_var_13 <- NULL
df_test_1$cat_var_14 <- NULL
df_test_1$cat_var_23 <- NULL
df_test_1$cat_var_31 <- NULL
df_test_1$cat_var_20 <- NULL
df_test_1$cat_var_35 <- NULL
df_test_1$cat_var_36 <- NULL
df_test_1$cat_var_37 <- NULL
df_test_1$cat_var_38 <- NULL
df_test_1$cat_var_39 <- NULL
df_test_1$cat_var_40 <- NULL
df_test_1$cat_var_41<- NULL
df_test_1$cat_var_42 <- NULL

df_test_1$target <- NULL

logit <- glm(target ~ cat_var_19+cat_var_21+
               cat_var_24+cat_var_25+cat_var_26, 
             data = df_train_1, family = "binomial",control = list(maxit = 1000))

library(MASS)
step <- stepAIC(logit, direction = "both")
step$anova

library(ROCR)
p <- predict.glm(logit, newdata=df_test_1, type="response")
ROCRPred <- prediction(predict(logit), df_train_1$target)
ROCRPref <- performance(ROCRPred, "tpr", "fpr")
plot(ROCRPref, colorize=TRUE, print.cutoffs.at=seq(0.1,by=0.1))


library(caret)
p <- predict.glm(logit, newdata = df_test_1, type = "response")
conf <- table(Actualvalue=df_train_1$target, PredictedValue=p>0.4)
confusionMatrix(p, reference = df_test_1$target)

df_test_1$target <- predict.glm(logit, newdata = df_test_1, type = "response")
test2 <- sample$transaction_id
test2<- data.frame(test2)
test2$target <- df_test_1$target
colnames(test2) <- c("transaction_id", "target")
test2

sample <- read.csv("sample_submissions.csv")

write.csv(test2, file = "submission_fraud.csv", row.names = F )
write.csv(test2, file = "submission_fraud_1.csv", row.names = F )

#correlation
library(corrplot)
correlations <- cor(df_train_numeric[ , ])
correlations
library(reshape2)
ggplot(melt(correlations),aes(x=Var1,y=Var2)) + geom_tile(aes(fill=value), col="grey") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1),axis.text.y = element_text(size=7))+
  scale_fill_gradient(low="white", high="#F1C40F")

#ANOVA chi square test
anova(logit, test = 'Chisq')

#Cross validation
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

mod_fit <- train(target ~  cat_var_19+cat_var_21+
                   cat_var_24+cat_var_25+cat_var_26, data= df_train_1, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)

pred <- predict(mod_fit, newdata= df_test_1)



