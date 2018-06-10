# predicting passing target of football

library(randomForest)
#library(caret)


setwd("/Users/hengli/Projects/mlsa18-pass-prediction/src")
#######################################################################
##            Loading data and preprocessing data                    ##
#######################################################################
rawData <- read.csv("passingfeatures.tsv", sep = "\t")
#names(rawData)
#data <- rawData[1:round(nrow(rawData)/10), ] # using a smaller data set for testing and debugging
data <- rawData

#######################################################################
##                           Pre-Modeling                            ##
#######################################################################
#fit.data <- data
response <- "label"
data[, response] <- as.factor(data[, response]) # use classification later, thus we do as.factor here
predictors <- names(data)[!(names(data) %in% 
                              c("pass_id", "label", "sender_id", "player_id", "time_start"))]

#######################################################################
##            Random Forest Modeling (10-fold cross validation)      ##
#######################################################################
numPasses <- max(data$pass_id) + 1

## use pass_id instead of row idx to create 10 folds
# shuffle the data
shuffledPassId <- seq(0, numPasses - 1)[sample(numPasses)]
# create 10 equally sized folds
folds <- cut(seq(1, numPasses), breaks = 10, labels = FALSE)

top1AccVec <- c(); top3AccVec <- c(); top5AccVec <- c()

for(i in 1:10) {
  cat("Fold count ", i, "\n")
  
  testIdx <- which(folds==i, arr.ind = TRUE)
  testPassId <- shuffledPassId[testIdx]
  trainPassId <- shuffledPassId[-testIdx]
  
  testData <- data[data$pass_id %in% testPassId, ]
  trainData <- data[data$pass_id %in% trainPassId, ]
  
  ## train random forest model
  fit.rf <- randomForest(x = trainData[, predictors], y = trainData[, response],
                         ntree = 500)#, mtry = 6, importance = TRUE)
  ## evaluate on test data
  predicted.classes <- predict(fit.rf, newdata=testData[, predictors], type="response")
  predicted.prob <- predict(fit.rf, newdata=testData[, predictors], type="prob")
  
  ## calculate top K accuracy
  top1TP <- 0; top3TP <- 0; top5TP <- 0
  validPassCount <- 0
  for (passId in testPassId) {
    testIdxForPassId <- which(testData$pass_id == passId)
    testDataForPassId <- testData[testIdxForPassId, ]
    if (sum(testDataForPassId$label == "1") == 1) {
      receiverId <- testDataForPassId[testDataForPassId$label == "1", "player_id"]
      validPassCount <- validPassCount + 1
    } else {
      next
    }
    
    probForPassId <- predicted.prob[testIdxForPassId, "1"]
    testDataForPassId$predicted <- probForPassId
    testDataForPassIdSorted <- testDataForPassId[order(testDataForPassId$predicted, decreasing = TRUE), ]
    if (receiverId %in% testDataForPassIdSorted[1, "player_id"]) top1TP <- top1TP + 1
    if (receiverId %in% testDataForPassIdSorted[1:3, "player_id"]) top3TP <- top3TP + 1
    if (receiverId %in% testDataForPassIdSorted[1:5, "player_id"]) top5TP <- top5TP + 1
    #break
  }
  
  top1Acc <- top1TP / validPassCount; top3Acc <- top3TP / validPassCount; 
  top5Acc <- top5TP / validPassCount 
  top1AccVec <- c(top1AccVec, top1Acc); top3AccVec <- c(top3AccVec, top3Acc); top5AccVec <- c(top5AccVec, top5Acc);
  print(validPassCount)
  print(top1Acc); print(top3Acc); print(top5Acc)
  
  #break
}

mean(top1AccVec); mean(top3AccVec); mean(top5AccVec)
