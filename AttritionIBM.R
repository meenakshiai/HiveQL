##import libraries

library("RColorBrewer")
library(corrplot)
library(rpart)
library(rpart.plot)
library(dplyr)

##read data. R's default behavior is to convert all the characters in factors 
##because of which character columns does not give the result we are looking for. 
##To avoid this problem, the data is read as stringsAsfactors = False.

ibmattrition <- read.csv("Attrition.csv", header = T, stringsAsFactors = FALSE)
names(ibmattrition)
attach(ibmattrition)


summary(ibmattrition)


levels(ibmattrition$Department)
levels(ibmattrition$EducationField)
levels(ibmattrition$JobRole)


typeof(ibmattrition$attrition)
typeof(ibmattrition$OverTime)

##Converting to numeric
ibmattrition$attrition <- ifelse(ibmattrition$Attrition == "Yes",1,0)

ibmattrition$OT <- ifelse(ibmattrition$OverTime == "Yes",1,0)

ibmattrition$dept <- ifelse(ibmattrition$Department == "Human Resources",0,
                            ifelse(ibmattrition$Department == "Research & Development",1,2))

ibmattrition$genderdummy <- ifelse(ibmattrition$Gender == "Female",0,1)

ibmattrition$marital <- ifelse(ibmattrition$MaritalStatus == "Single",1,
                               ifelse(ibmattrition$MaritalStatus == "Married",0,2))

ibmattrition$travel <- ifelse(ibmattrition$BusinessTravel == "Non-Travel",0,
                              ifelse(ibmattrition$BusinessTravel == "Travel_Rarely",1,2))

ibmattrition$edufield <- ifelse(ibmattrition$EducationField == "Human Resources",0,
                              ifelse(ibmattrition$BusinessTravel == "Life Sciences",1,
                                     ifelse(ibmattrition$EducationField == "Marketing",2,
                                            ifelse(ibmattrition$EducationField == "Medical",3,
                                                   ifelse(ibmattrition$EducationField == "Other",4,5)))))

ibmattrition$role <- ifelse(ibmattrition$JobRole == "Healthcare Representative",0,
                                ifelse(ibmattrition$JobRole == "Human Resources",1,
                                       ifelse(ibmattrition$JobRole == "Laboratory Technician",2,
                                              ifelse(ibmattrition$JobRole == "Manager",3,
                                                     ifelse(ibmattrition$JobRole == "Manufacturing Director",4,
                                                            ifelse(ibmattrition$JobRole == "Research Director",5,
                                                                   ifelse(ibmattrition$JobRole == "Research Scientist",6,
                                                                          ifelse(ibmattrition$JobRole == "Sales Executive",7,8))))))))



## HeatMap. Correlation matrix.
heatmap(cor(ibmattrition[sapply(ibmattrition, is.numeric)]),
        Rowv = NA, Colv = NA,col=brewer.pal(9,"Greens"),margins=c(9,9))


##Assigning a new variable for the dataset to work on Decision tree and 
##create a new data frame using only the variables which look relevant and 
##have a correlation with Attrition.

## Create new data frame.
ibmattritionDT <- read.csv("Attrition.csv", header = T, stringsAsFactors = FALSE)


attritiondf <- data.frame(cbind(ibmattritionDT$OverTime,
                                ibmattritionDT$BusinessTravel,
                                ibmattritionDT$YearsWithCurrManager,
                                ibmattritionDT$YearsSinceLastPromotion,
                                ibmattritionDT$Attrition,
                                ibmattritionDT$EnvironmentSatisfaction,
                                ibmattritionDT$JobLevel,
                                ibmattritionDT$JobRole,
                                ibmattritionDT$PercentSalaryHike,
                                ibmattritionDT$TotalWorkingYears,
                                ibmattritionDT$WorkLifeBalance,
                                ibmattritionDT$YearsInCurrentRole))

colnames(attritiondf) <- c("Overtime",
                            "Businesstravel",
                            "Yrswithmanager",
                            "Yrssincepromotion",
                           "Attrition",
                           "EnvironmentSatisfaction",
                           "Joblevel",
                           "JobRole",
                           "PercentSalaryHike",
                           "TotalWorkingYears",
                           "WorkLifeBalance",
                           "YearsInCurrentRole")

## mutate to add new variables to the data frame.
attritiondf <- mutate(attritiondf,
                       Overtime = factor(Overtime, levels = c("Yes", "No"), labels = c("Yes", "No")),
                      Businesstravel = factor(Businesstravel, levels=c("Non-Travel","Travel_Frequently","Travel_Rarely"),
                                                  labels=c("Non-Travel","Travel_Frequently","Travel_Rarely")),
                      Yrswithmanager = as.integer(Yrswithmanager),
                      Yrssincepromotion = as.integer(Yrssincepromotion),
                      Attrition = factor(Attrition, levels = c("No","Yes"), labels = c("No","Yes")),
                      EnvironmentSatisfaction = as.integer(EnvironmentSatisfaction),
                      Joblevel = as.integer(Joblevel),
                      JobRole = factor(JobRole, levels = c("Healthcare Representative","Human Resources",
                                                           "Laboratory Technician","Manager","Manufacturing Director",
                                                           "Research Director","Research Scientist","Sales Executive",
                                                           "Sales Representative"), 
                                       labels = c("Healthcare Representative","Human Resources","Laboratory Technician",
                                                  "Manager","Manufacturing Director","Research Director",
                                                  "Research Scientist","Sales Executive","Sales Representative")),
                      PercentSalaryHike = as.integer(PercentSalaryHike),
                      TotalWorkingYears = as.integer(TotalWorkingYears),
                      WorkLifeBalance = as.integer(WorkLifeBalance),
                      YearsInCurrentRole = as.integer(YearsInCurrentRole))



glimpse(attritiondf)


##Creating a data frame to test and train for decision tree. 
##Assigning 80 % data to the train table and 20% to test of the 1470 observations.

attritiontrainDT <- attritiondf[1:1176,]
attritiontestDT <- attritiondf[1177:1470,]

##Using prop.table function to check if the randomization process is right.
prop.table(table(attritiontrainDT$Attrition))

prop.table(table(attritiontestDT$Attrition))

##Now run the function to fit the model and plot the decision tree.

fit <- rpart(Attrition ~., data = attritiontrainDT, method = 'class')
rpart.plot(fit,extra = 104, cex=0.7)

## predict the data set.

predict_unseen <-predict(fit, attritiontestDT, type = 'class')

temp <- table(attritiontestDT$Attrition, predict_unseen)
temp


## accuracy test from the confusion matrix.
accuracy_Test <- sum(diag(temp)) / sum(temp)

print(paste('Accuracy for test', accuracy_Test))




