#Load up the needed packages 
library(gmodels)
library(rpart)
library(rpart.plot)
library(pROC)

#Load the data into the workspace
loan_data <- readRDS(file="loan_data.rds")
str(loan_data)

#Look at the number of defaults, non-defaults, and their proportions
  #There data contains 11% defaults
CrossTable(loan_data$loan_status)

#Use crosstable to look at the number/proportion of defaults and non defaults
  #for each grade
CrossTable(loan_data$grade,
           loan_data$loan_status,
           prop.r = TRUE,
           prop.c=FALSE,
           prop.t = FALSE,
           prop.chisq = FALSE)
#The proportion of defaults increases when the credit rating moves from A to G

#Create histogram of loan_amnt: hist_1
  #Most of the loan amounts are under 15k
hist_1 <- hist(loan_data$loan_amnt)

#Print locations of the breaks in hist_1
hist_1$breaks

#Change number of breaks to 200 and add labels: hist_2
hist_2 <- hist(loan_data$loan_amnt,
               breaks = 200,
               xlab = "Loan Amount", 
               main = "Histogram of the loan amount")

#Plot the age variable
  #Someone has an age of over 120, which is incredibly unlikely
plot(loan_data$age, ylab="Age")

#Save the outlier's index to index_highage
index_highage <- which(loan_data$age>122)

#Remove the outlier
loan_data <- loan_data[-index_highage, ]

#Make bivariate scatterplot of age and annual income
plot(loan_data$age,
     loan_data$annual_inc,
     xlab = "Age",
     ylab = "Annual Income")

#Look at summary of the interest rates for the data
summary(loan_data$int_rate)

#Get indices of missing interest rates: na_index
na_index <- which(is.na(loan_data$int_rate))

#Compute the median of int_rate
median_ir <- median(loan_data$int_rate, na.rm=TRUE)

#Replace missing interest rates with median
loan_data$int_rate[na_index] <- median_ir

#Check if the NAs are gone
summary(loan_data$int_rate)

#Engineer a new feature vector for interest rate categories/buckets
loan_data$ir_cat <- rep(NA, length(loan_data$int_rate))

loan_data$ir_cat[which(loan_data$int_rate <= 8)] <- "0-8"
loan_data$ir_cat[which(loan_data$int_rate > 8 & loan_data$int_rate <= 11)] <- "8-11"
loan_data$ir_cat[which(loan_data$int_rate > 11 & loan_data$int_rate <= 13.5)] <- "11-13.5"
loan_data$ir_cat[which(loan_data$int_rate > 13.5)] <- "13.5+"
loan_data$ir_cat[which(is.na(loan_data$int_rate))] <- "Missing"

loan_data$ir_cat <- as.factor(loan_data$ir_cat)

#Look at the bins and their distribution
plot(loan_data$ir_cat)

#Look at the different categories in ir_cat using table()
table(loan_data$ir_cat)

#Engineer a new feature vector for employment length categories/buckets

loan_data$emp_cat <- rep(NA, length(loan_data$emp_length))
loan_data$emp_cat[which(loan_data$emp_length >= 0 & loan_data$emp_length <= 15)] <- "0-15"
loan_data$emp_cat[which(loan_data$emp_length > 15 & loan_data$emp_length <= 30)] <- "15-30"
loan_data$emp_cat[which(loan_data$emp_length > 30 & loan_data$emp_length <= 45)] <- "30-45"
loan_data$emp_cat[which(loan_data$emp_length > 45)] <- "45+"
loan_data$emp_cat[which(is.na(loan_data$emp_length))] <- "Missing"

loan_data$emp_cat <- as.factor(loan_data$emp_cat)
table(loan_data$emp_cat)

#Set a seed for reproducibility
set.seed(567)

#Store row numbers for training set: index_train
index_train <- sample(1:nrow(loan_data), (2/3)*(nrow(loan_data)))

#Create training set: training_set
training_set <- loan_data[index_train, ]

#Create test set: test_set
test_set <- loan_data[-index_train,]

#Build the logistic regression model
log_model_multi <- glm(loan_status ~ age + ir_cat+ grade+ loan_amnt+ annual_inc+ emp_cat,
                       data=training_set,
                       family="binomial")

#Obtain significance levels using summary()
summary(log_model_multi)

#Look at the predictions range
predictions_multi <- predict(log_model_multi,
                                 newdata = test_set,
                                 type = "response")

range(predictions_multi)

#Make a binary predictions-vector using a cut-off of 15% and 20%
pred_cutoff_15 <- ifelse(predictions_multi > 0.15, 1, 0)
pred_cutoff_20 <- ifelse(predictions_multi > 0.20, 1, 0)

#Check the cutoff with a confusion matrix
table(test_set$loan_status, pred_cutoff_15)
table(test_set$loan_status, pred_cutoff_20)
#Accuracy and specificity increase, but sensitivity decreases

#Check out the logit, probit and cloglog logistic regression models
log_model_logit <- glm(loan_status ~ age + emp_cat + ir_cat + loan_amnt,
                       family = binomial(link = logit),
                       data = training_set)

log_model_probit <- glm(loan_status ~ age + emp_cat + ir_cat + loan_amnt,
                        family = binomial(link = probit),
                        data = training_set)

log_model_cloglog <-glm(loan_status ~ age + emp_cat + ir_cat + loan_amnt,
                        family = binomial(link = cloglog),
                        data = training_set)  

log_model_all_full <- glm(loan_status ~ loan_amnt + int_rate + grade + emp_length 
                          + home_ownership + annual_inc + age + ir_cat,
                          family = binomial(link = logit),
                          data = training_set)  

#Make predictions for all models using the test set
predictions_logit <- predict(log_model_logit,
                             newdata = test_set,
                             type = "response")
predictions_probit <- predict(log_model_probit,
                              newdata = test_set,
                              type = "response")
predictions_cloglog <- predict(log_model_cloglog,
                               newdata = test_set,
                               type = "response")
predictions_all_full <- predict(log_model_all_full,
                               newdata = test_set,
                               type = "response")

#Using a cut-off of 14% to make binary predictions-vectors
cutoff <- 0.14
class_pred_logit <- ifelse(predictions_logit > cutoff, 1, 0)
class_pred_probit <- ifelse(predictions_probit > cutoff, 1, 0)
class_pred_cloglog <- ifelse(predictions_cloglog > cutoff, 1, 0)

#Creating a vector to store the actual loan default status values
true_val <- test_set$loan_status

#Make a confusion matrix for the three models
tab_class_logit <- table(true_val,class_pred_logit)
tab_class_probit <- table(true_val,class_pred_probit)
tab_class_cloglog <- table(true_val,class_pred_cloglog)

#Check out the matrices
tab_class_logit
tab_class_probit
tab_class_cloglog

#Compute the classification accuracy for all three models
acc_logit <- sum(diag(tab_class_logit)) / nrow(test_set)
acc_probit <- sum(diag(tab_class_probit)) / nrow(test_set)
acc_cloglog <- sum(diag(tab_class_cloglog)) / nrow(test_set)

#Check out each accuracy
acc_logit
acc_probit
acc_cloglog
#They're all about 70%

#Create an undersampled training set with 2/3 non-defaults and 1/3 defaults
defaults <- loan_data[loan_data$loan_status==1,]
nondefaults <- loan_data[loan_data$loan_status==0,]

part1 <- nondefaults[sample(nrow(nondefaults), 4380,
                        replace = FALSE,
                        prob = NULL),]

part2 <- defaults[sample(nrow(defaults), 2190,
                            replace = FALSE,
                            prob = NULL),]

undersampled_training_set <- rbind(part1,part2)
table(undersampled_training_set$loan_status)

#Change the code provided in the video such that a decision tree is constructed using the undersampled training set. Include rpart.control to relax the complexity parameter to 0.001.
tree_undersample <- rpart(loan_status ~ .,
                          method = "class",
                          data =  undersampled_training_set,
                          control = rpart.control(cp=0.001))

#Plot the decision tree
plot(tree_undersample,
     uniform = TRUE)

#Add labels to the decision tree
text(tree_undersample)

#Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_undersample)

#Use printcp() to identify for which complexity parameter the cross-validated error rate is minimized.
printcp(tree_undersample)

#Create an index for of the row with the minimum xerror
index <- which.min(tree_undersample$cptable[ , "xerror"])

#Create tree_min
tree_min <- tree_undersample$cptable[index, "CP"]

#Prune the tree using tree_min
ptree_undersample <- prune(tree_undersample, cp = tree_min)

#Use prp() to plot the pruned tree
prp(ptree_undersample)

#Change the code below such that a tree is constructed with adjusted prior probabilities.
tree_prior <- rpart(loan_status ~ .,
                    method = "class",
                    data = training_set,
                    control = rpart.control(cp=0.001),
                    parms = list(prior=c(0.7,0.3)))

#Plot the decision tree
plot(tree_prior,uniform=TRUE)

#Add labels to the decision tree
text(tree_prior)

#Try changing the cost matrix to penalize defaults as non-defaults more heavily
tree_loss_matrix <- rpart(loan_status ~ .,
                          method = "class",
                          data =  training_set,
                          control = rpart.control(cp = 0.001),
                          parms = list(loss=matrix(c(0,10,1,0),ncol=2)))



#Plot the decision tree
plot(tree_loss_matrix,
     uniform = TRUE)

#Add labels to the decision tree
text(tree_loss_matrix)


#Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_prior)

#Use printcp() to identify for which complexity parameter the cross-validated error rate is minimized.
printcp(tree_prior)

#Create an index for of the row with the minimum xerror
index <- which.min(tree_prior$cptable[ , "xerror"])

#Create tree_min
tree_min <- tree_prior$cptable[index, "CP"]

#Prune the tree using tree_min
ptree_prior <- prune(tree_prior, cp = tree_min)

#Use prp() to plot the pruned tree
prp(ptree_prior)

#set a seed and run the code to construct the tree with the loss matrix again
set.seed(345)
tree_loss_matrix  <- rpart(loan_status ~ .,
                           method = "class", 
                           data = training_set,
                           parms = list(loss=matrix(c(0, 10, 1, 0), ncol = 2)),
                           control = rpart.control(cp = 0.001))

#Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_loss_matrix)

#Prune the tree using cp = 0.0012788
ptree_loss_matrix <- prune(tree_loss_matrix, cp=0.0012788)

#Use prp() and argument extra = 1 to plot the pruned tree
prp(ptree_loss_matrix, extra = 1)
case_weights <- ifelse(training_set$loan_status==0,1,3)

#Make a tree while adjusting the minsplit and minbucket parameters
tree_weights <- rpart(loan_status ~ .,
                      method = "class",
                      data = training_set,
                      weights = case_weights,
                      control = rpart.control(minsplit = 5, minbucket = 2, cp = 0.001))

#Plot the cross-validated error rate for a changing cp
plotcp(tree_weights)

#Create an index for of the row with the minimum xerror
index <- which.min(tree_weights$cp[ , "xerror"])

#Create tree_min
tree_min <- tree_weights$cp[index, "CP"]

#Prune the tree using tree_min
ptree_weights <- prune(tree_weights, cp =tree_min)

#Plot the pruned tree using the rpart.plot()-package
prp(ptree_weights, extra = 1)

#Make predictions for each of the pruned trees using the test set.
pred_undersample <- predict(ptree_undersample, newdata=test_set, type="class")
pred_prior <- predict(ptree_prior, newdata=test_set, type="class")
pred_loss_matrix <- predict(ptree_loss_matrix, newdata=test_set, type="class")
pred_weights <- predict(ptree_weights, newdata=test_set, type="class")

#Construct confusion matrices using the predictions.
confmat_undersample <- table(test_set$loan_status,pred_undersample)
confmat_prior <- table(test_set$loan_status, pred_prior)
confmat_loss_matrix <- table(test_set$loan_status, pred_loss_matrix)
confmat_weights <- table(test_set$loan_status, pred_weights)

#Compute the accuracies
acc_undersample <- sum(diag(confmat_undersample)) / nrow(test_set)
acc_prior <- sum(diag(confmat_prior)) / nrow(test_set)
acc_loss_matrix <- sum(diag(confmat_loss_matrix)) / nrow(test_set)
acc_weights <- sum(diag(confmat_weights)) / nrow(test_set)

#Make predictions for the probability of default using the pruned tree and the test set.
prob_default_prior <- predict(ptree_prior, newdata = test_set)[ ,2]

#Obtain the cutoff for acceptance rate 80%
cutoff_prior <- quantile(prob_default_prior, 0.8)

#Obtain the binary predictions.
bin_pred_prior_80 <- ifelse(prob_default_prior> cutoff_prior,1,0)

#Obtain the actual default status for the accepted loans
accepted_status_prior_80 <- cbind(prob_default_prior, bin_pred_prior_80)

test_set$loan_status[bin_pred_prior_80 == 0]

#Obtain the bad rate for the accepted loans
bad_rate <- sum(accepted_status_prior_80)/length(accepted_status_prior_80)
bad_rate

strategy_bank <- function(prob_of_def){
  cutoff=rep(NA, 21)
  bad_rate=rep(NA, 21)
  accept_rate=seq(1,0,by=-0.05)
  for (i in 1:21){
    cutoff[i]=quantile(prob_of_def,accept_rate[i])
    pred_i=ifelse(prob_of_def> cutoff[i], 1, 0)
    pred_as_good=test_set$loan_status[pred_i==0]
    bad_rate[i]=sum(pred_as_good)/length(pred_as_good)}
  table=cbind(accept_rate,cutoff=round(cutoff,4),bad_rate=round(bad_rate,4))
  return(list(table=table,bad_rate=bad_rate, accept_rate=accept_rate, cutoff=cutoff))
}

#Apply the function strategy_bank to both predictions_cloglog and predictions_loss_matrix
strategy_cloglog <- strategy_bank(predictions_cloglog)

#Obtain the strategy tables for both prediction-vectors
strategy_cloglog$table

#Plot the strategy functions
par(mfrow = c(1,2))
plot(strategy_cloglog$accept_rate, strategy_cloglog$bad_rate, 
     type = "l", xlab = "Acceptance rate", ylab = "Bad rate", 
     lwd = 2, main = "logistic regression")

#Construct the objects containing ROC-information
ROC_logit <- roc(test_set$loan_status, predictions_logit)
ROC_probit <- roc(test_set$loan_status, predictions_probit)
ROC_cloglog <- roc(test_set$loan_status, predictions_cloglog)
ROC_all_full <- roc(test_set$loan_status, predictions_all_full)##

#Draw all ROCs on one plot
plot(ROC_logit)
lines(ROC_probit, col="blue")
lines(ROC_cloglog, col="red")
lines(ROC_all_full, col="green")

#Compute the AUCs
auc(ROC_logit)
auc(ROC_probit)
auc(ROC_cloglog)
auc(ROC_all_full)

predictions_undersample <- predict(ptree_undersample, newdata=test_set, type="vector")
predictions_prior <- predict(ptree_prior, newdata=test_set, type="vector")
predictions_loss_matrix <- predict(ptree_loss_matrix, newdata=test_set, type="vector")
predictions_weights <- predict(ptree_weights, newdata=test_set, type="vector")

#Construct the objects containing ROC-information
ROC_undersample <- roc(test_set$loan_status,predictions_undersample)
ROC_prior <- roc(test_set$loan_status,predictions_prior)
ROC_loss_matrix <- roc(test_set$loan_status,predictions_loss_matrix)
ROC_weights <- roc(test_set$loan_status,predictions_weights)

#Draw the ROC-curves in one plot
plot(ROC_undersample)
lines(ROC_prior,col="blue")
lines(ROC_loss_matrix,col="red")
lines(ROC_weights,col="green")

#Compute the AUCs
auc(ROC_undersample)
auc(ROC_prior)
auc(ROC_loss_matrix)
auc(ROC_weights)

#Build four models each time deleting one variable in log_3_remove_ir
log_4_remove_amnt <- glm(loan_status ~ grade + annual_inc + emp_cat, 
                         family = binomial, data = training_set) 
log_4_remove_grade <- glm(loan_status ~ loan_amnt + annual_inc + emp_cat, family = binomial, data = training_set)
log_4_remove_inc <- glm(loan_status ~ loan_amnt + grade + emp_cat, family = binomial, data = training_set)
log_4_remove_emp <- glm(loan_status ~ loan_amnt + grade + annual_inc, family = binomial, data = training_set)

#Make PD-predictions for each of the models
pred_4_remove_amnt <- predict(log_4_remove_amnt, newdata = test_set, type = "response")
pred_4_remove_grade <- predict(log_4_remove_grade, newdata = test_set, type = "response")
pred_4_remove_inc <- predict(log_4_remove_inc, newdata = test_set, type = "response")
pred_4_remove_emp <- predict(log_4_remove_emp, newdata = test_set, type = "response")

#Compute the AUCs
auc(test_set$loan_status,pred_4_remove_amnt)
auc(test_set$loan_status,pred_4_remove_grade)
auc(test_set$loan_status,pred_4_remove_inc)
auc(test_set$loan_status,pred_4_remove_emp)

#Build three models each time deleting one variable in log_4_remove_amnt
log_5_remove_grade <- glm(loan_status ~ annual_inc + emp_cat, family = binomial, data = training_set) 
log_5_remove_inc <- glm(loan_status ~ grade + emp_cat, family = binomial, data = training_set) 
log_5_remove_emp <- glm(loan_status ~ grade + annual_inc, family = binomial, data = training_set) 

#Make PD-predictions for each of the models
pred_5_remove_grade <- predict(log_5_remove_grade, newdata = test_set, type = "response")
pred_5_remove_inc <- predict(log_5_remove_inc, newdata = test_set, type = "response")
pred_5_remove_emp <- predict(log_5_remove_emp, newdata = test_set, type = "response")

#Compute the AUCs
auc(test_set$loan_status,pred_5_remove_grade)
auc(test_set$loan_status,pred_5_remove_inc)
auc(test_set$loan_status,pred_5_remove_emp)

#Plot the ROC-curve for the best model here
plot(roc(test_set$loan_status,pred_4_remove_amnt))

str(pred_4_remove_amnt)

#Obtain the cutoff for acceptance rate 80%
cutoff_remove_amnt <- quantile(pred_4_remove_amnt, 0.8)

#Obtain the binary predictions.
bin_pred_remove_amnt <- ifelse(pred_4_remove_amnt> cutoff_remove_amnt,1,0)

#Obtain the actual default status for the accepted loans
accepted_status_remove_amnt <- cbind(pred_4_remove_amnt, bin_pred_remove_amnt)

#Check the structure of our new matrix
str(accepted_status_remove_amnt)

#Transform it into a dataframe for merging
accepted_status_remove_amnt <- data.frame(accepted_status_remove_amnt)

#Create a final dataframe which combines the predictions and actual test set
results_df <- cbind(test_set,accepted_status_remove_amnt)
table(results_df$loan_status,results_df$bin_pred_remove_amnt)