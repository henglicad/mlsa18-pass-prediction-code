# predicting passing target of football

library(randomForest)
library(caret)
library(Hmisc)


setwd("/Users/hengli/Projects/mlsa18-pass-prediction/src")
#######################################################################
##            Loading data and preprocessing data                    ##
#######################################################################
rawData <- read.csv("passingfeatures.tsv", sep = "\t")
#names(rawData)
#data <- rawData[1:round(nrow(rawData)/10), ] # using a smaller data set for testing and debugging
data <- rawData

#######################################################################
##                         Passing pattern                           ##
#######################################################################
## passing accuracy
getPassAccuracy <- function(subData) {
  sum(as.numeric(as.character(subData$is_in_same_team)) & 
                      as.numeric(as.character(subData$label))) / 
  sum(as.numeric(as.character(subData$label)))
}
passAccuracy <- getPassAccuracy(data)
# 0.8285738
passAccuracyBack <- getPassAccuracy(data[as.character(data$is_sender_in_back_field)=="1", ])
# 0.8561359
passAccuracyMiddle <- getPassAccuracy(data[as.character(data$is_sender_in_middle_field)=="1", ])
# 0.829286
passAccuracyFront <- getPassAccuracy(data[as.character(data$is_sender_in_front_field)=="1", ])
# 0.7856468

## passing distance
getPassDist <- function(subData) {
  subData[as.character(subData$label) == "1", "distance"]
}
dists <- getPassDist(data)
median(dists) # 1405.969
distsBack <- getPassDist(data[as.character(data$is_sender_in_back_field)=="1", ])
median(distsBack) # 1699.559
distsMiddle <- getPassDist(data[as.character(data$is_sender_in_middle_field)=="1", ])
median(distsMiddle) # 1394.597
distsFront <- getPassDist(data[as.character(data$is_sender_in_front_field)=="1", ])
median(distsFront) # 1073.825

## passing forwards
#data$is_player_in_offense_direction_relative_to_sender
getPassForwardsRatio <- function(subData) {
  sum(as.character(subData$label) == "1" & 
            as.character(subData$is_player_in_offense_direction_relative_to_sender) == "1") /
    sum(as.character(subData$label) == "1")
}
forwardsRatio <- getPassForwardsRatio(data)
forwardsRatio # 0.6216573
forwardsRatioBack <- getPassForwardsRatio(data[as.character(data$is_sender_in_back_field)=="1", ])
forwardsRatioBack # 0.7427107
forwardsRatioMiddle <- getPassForwardsRatio(data[as.character(data$is_sender_in_middle_field)=="1", ])
forwardsRatioMiddle # 0.6050603
forwardsRatioFront <- getPassForwardsRatio(data[as.character(data$is_sender_in_front_field)=="1", ])
forwardsRatioFront # 0.4971671

#######################################################################
##                           Pre-Modeling                            ##
#######################################################################
#fit.data <- data
response <- "label"
data[, response] <- as.factor(data[, response]) # use classification later, thus we do as.factor here
predictors <- names(data)[!(names(data) %in% 
                              c("pass_id",  "line_num", "label", "sender_id", "player_id"))]

#######################################################################
##                     Correlation analysis                          ##
#######################################################################
explVars <- data[, predictors]
explVarMatrix <- as.matrix(explVars)
v <- varclus(explVarMatrix, similarity="spear", trans="abs")
plot(v, cex = 1.0, cex.axis=1.0, cex.lab=1.0, cex.main=1.0)#, labels=labels)
abline(h=0.2, lty=3, col = "red")#, lwd = 3) # correlation threshold: 0.8
keptVars <- c("min_pass_angle", "abs_y_diff", "player_closest_friend_to_sender_dist",
              "distance", "num_dangerous_opponents_along_passing_line",
              "player_to_sender_dist_rank_among_friends", "norm_player_sender_x_diff",
              "player_to_offense_gate_dist_rank_relative_to_friends",
              "player_to_offense_gate_dist_rank_relative_to_opponents",
              "player_closest_friend_dist", "is_player_goal_keeper",
              "player_closest_opponent_dist", #"is_sender_player_in_same_field",
              "is_in_same_team", "is_sender_in_front_field",
              "sender_team_closest_dist_to_offense_goal_line", "is_player_in_center_circle",
              #"is_player_in_middle_field", 
              "player_to_center_distance",
              "is_player_in_front_field", "is_player_in_back_field",
              "player_to_offense_gate_dist", "time_start", #"is_start_of_game",
              "sender_closest_friend_dist", "sender_to_offense_gate_dist_rank_relative_to_friends",
              "sender_closest_opponent_dist", #"player_closest_3_friends_dist",
              "is_sender_goal_keeper", "sender_to_center_distance",
              "is_sender_in_back_field", #"is_sender_in_middle_field",
              "sender_x", "player_x", "player_y",
              "player_to_top_sideline_dist_rank_relative_to_friends",
              "sender_team_cloeset_dist_to_bottom_sideline",
              "sender_team_median_dist_to_top_sideline",
              "sender_team_closest_dist_to_top_sideline",
              "sender_y"
              )
predictors <- predictors[predictors %in% keptVars]
#######################################################################
##            Random Forest Modeling (10-fold cross validation)      ##
#######################################################################

# field-wise modeling
#data <- data[as.character(data$is_sender_in_front_field) == "1", ]

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
  #trainData <- downSample(trainData, trainData[, response])
  #trainData <- upSample(trainData, trainData[, response])
  #break
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
