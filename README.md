# revenue-prediction
## loading libraries
rm(list=ls())
library(caret)
library(dummies)
library(plyr)
library(vegan)
library(reshape)
## loading data 
setwd("C:/Users/Rightway/Desktop/final project")
#loading the data
train<-read.csv("train.csv",header = T,sep=',',stringsAsFactors=F)
test<-read.csv("test.csv",header=T,sep=',',stringsAsFactors=F)
#checking the distribution of target variable
library(lattice)
histogram(train$revenue)
#checking for outliers and removing the outliers
boxplot(train$revenue)
#checking the maximum and minimum values in the target variable
which.max(train$revenue)
which.min(train$revenue)
summary(train$revenue)
train<-train[-c(17,22),]

str(train)
#converting the city variable into factors in test and train
train$City<-as.factor(train$City)
str(train$City)
test$City<-as.factor(test$City)
str(test$City)
library(ggplot2)
#count of each city in train data
p <- ggplot(data = train,  aes(x=City, y=revenue))
p+geom_bar(stat='identity',fill="blue")+theme(axis.text.x = element_text(angle = 90, hjust = 1))
#on test
pt <- ggplot(data = test, mapping = aes(City))
pt+geom_bar(fill="yellow", colour="darkgreen")+theme(axis.text.x = element_text(angle = 90, hjust = 1))
#cont of big cities and others in train data
cg <- ggplot(data = train, mapping = aes(x=City.Group,y=revenue))
cg+geom_point(stat='identity',fill="grey")
#on test data
tyt<- ggplot(data =test, mapping = aes(City.Group))
tyt+geom_bar(fill="blue", colour="darkgreen")
#count of type in train data
#cont of big cities and others in train data
typ <- ggplot(data = train, mapping = aes(x=Type,y=revenue))
typ+geom_point(stat='identity',fill="black")
#on test data
typt<- ggplot(data =test, mapping = aes(Type))
typt+geom_bar(fill="brown", colour="darkgreen")

#each p varibale count on train data
d <- melt(train[,-c(1:5)])
ggplot(d, aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()


## combining the test and train for preprocessing
panel <- rbind(train[,-ncol(train)], test)
str(panel)
#################### DATA PREPROCESSING#####################
#converting the date varibale into data format and again converting it into years to make it continuous

library(lubridate)
str(panel$Open.Date)
#converting the open.date into date format
panel$Open.Date<-as.Date(panel$Open.Date, "%m/%d/%Y")
str(panel$Open.Date)
#making the open.date into continue number by differencing it with the present date to get the numberof days
panel$Open.Date <- floor((as.numeric(Sys.Date() - panel$Open.Date))/365)
str(panel)


#converting the  city groupp into factor
panel$City.Group<-as.factor(panel$City.Group)
str(panel$City.Group)
summary(panel$City.Group)
#converting type of restaurent dt to il and mb to fc

panel$Type<-as.factor(panel$Type)
summary(panel$Type)

panel$Type[panel$Type == "DT"] <- "IL"
panel$Type[panel$Type == "MB"] <- "FC"
panel$Type <- as.factor(panel$Type)
summary(panel$Type)
panel$Type<-factor(panel$Type)
str(panel)


panel <- subset(panel, select = -c( City))
#from the p variables converting the integer varibles into categorical variables
summary(panel[,c(5:41)])
catg =c("P1","P5","P6","P7","P8","P9","P10","P11",
        "P12","P14", "P15", "P16", "P17", "P18", "P19",
        "P20", "P21", "P22", "P23", "P24", "P25", 
        "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37")


# converting some categorical variables into dummies
panel <- dummy.data.frame(panel, names=c("P1","P5","P6","P7","P8","P9","P10","P11","P12","P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23", "P24", "P25", "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37"), all=T)
su<-apply(panel[,-c(2:4)],2,sum)
which(su>1)


# splitting into train and test
X_train <- panel[1:135,-1]
years<-X_train$Open.Date
X_test <- panel[136:100135,-1]

# building model on log of revenue
result <- log(train$revenue)
histogram(result)
plot(years,result)

## loading libraries

library(randomForest)

RandomForestRegression_CV <- function(X_train,y,X_test=data.frame(),cv=5,ntree=50,nodesize=5,seed=123,metric="rmse")
{
  score <- function(a,b,metric)
  {
    switch(metric,
   #error metric        
           rmse = sqrt(sum((a-b)^2)/length(a)))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  set.seed(seed)
  #creating numbers for each row for 5 fold cv
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    #train and test for the cv
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    #bulding model on train cv
    model_rf <- randomForest(result ~., data = X_build, ntree = ntree, nodesize = nodesize)
    
    pred_rf <- predict(model_rf, X_val)
    X_val <- cbind(X_val, pred_rf)
    
    if (nrow(X_test) > 0)
    {
      pred_rf <- predict(model_rf, X_test)
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_rf, metric), "\n", sep = "")
    
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_rf)
      }      
    }
    
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_rf <- (X_test$pred_rf * (i-1) + pred_rf)/i
      }            
    }
    
    gc()
  } 
  
  output <- output[order(output$order),]
  
  output <- subset(output, select = c("order", "pred_rf"))
  return(list(output, X_test))  
}





# 5-fold cross validation and scoring
model_rf_1 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=235,metric="rmse")
model_rf_2 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=357,metric="rmse")
model_rf_3 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=13,metric="rmse")
model_rf_4 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=753,metric="rmse")
model_rf_5 <- RandomForestRegression_CV(X_train,result,X_test,cv=5,ntree=25,nodesize=5,seed=532,metric="rmse")



## submission
test_rf_1 <- model_rf_1[[2]]
test_rf_2 <- model_rf_2[[2]]
test_rf_3 <- model_rf_3[[2]]
test_rf_4 <- model_rf_4[[2]]
test_rf_5 <- model_rf_5[[2]]

submit <- data.frame("Id" = test$Id,
                     "Prediction" = 0.2*exp(test_rf_1$pred_rf) + 0.2*exp(test_rf_2$pred_rf) + 0.2*exp(test_rf_3$pred_rf) + 0.2*exp(test_rf_4$pred_rf) + 0.2*exp(test_rf_5$pred_rf))

write.csv(submit, "submit02.csv", row.names=F)

