# Install and load required libraries
install.packages(c("tidytext", "tm", "quanteda", "caret", "e1071", "randomForest", "gbm", "caretEnsemble", "reshape2", "ggplot2"))
library(tidytext)
library(tm)
library(quanteda)
library(caret)
library(e1071)
library(randomForest)
library(gbm)
library(caretEnsemble)
library(reshape2)
library(ggplot2)

# Step 1: Data Preparation
# Load the labeled text dataset (example: using the "20 Newsgroups" dataset)
data <- read.csv("20_newsgroups.csv", stringsAsFactors = FALSE)

# Preprocess the text data
corpus <- Corpus(VectorSource(data$text))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

# Convert corpus to document-term matrix
dtm <- DocumentTermMatrix(corpus)

# Split the dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$label, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Step 2: Feature Extraction
# Create TF-IDF features
tfidf <- weightTfIdf(dtm)
trainMatrix <- as.matrix(tfidf[trainIndex, ])
testMatrix <- as.matrix(tfidf[-trainIndex, ])

# Dimensionality reduction (optional)
# pca <- prcomp(trainMatrix)
# trainMatrix <- pca$x[, 1:100]
# testMatrix <- predict(pca, testMatrix)[, 1:100]

# Step 3: Model Building
# Train individual models
ctrl <- trainControl(method = "cv", number = 5)

logisticModel <- train(trainMatrix, trainData$label, method = "multinom", trControl = ctrl)
naiveBayesModel <- train(trainMatrix, trainData$label, method = "naive_bayes", trControl = ctrl)
svmModel <- train(trainMatrix, trainData$label, method = "svmLinear", trControl = ctrl)
randomForestModel <- train(trainMatrix, trainData$label, method = "rf", trControl = ctrl)
gbmModel <- train(trainMatrix, trainData$label, method = "gbm", trControl = ctrl, verbose = FALSE)

# Step 4: Ensemble Learning
# Create a list of base models
baseModels <- list(logisticModel, naiveBayesModel, svmModel, randomForestModel, gbmModel)

# Ensemble using voting
votingModel <- caretEnsemble(baseModels, metric = "Accuracy", trControl = ctrl)

# Make predictions on the testing set
logisticPred <- predict(logisticModel, testMatrix)
naiveBayesPred <- predict(naiveBayesModel, testMatrix)
svmPred <- predict(svmModel, testMatrix)
randomForestPred <- predict(randomForestModel, testMatrix)
gbmPred <- predict(gbmModel, testMatrix)
votingPred <- predict(votingModel, testMatrix)

# Step 5: Model Interpretation
# Feature importance (example: using Random Forest)
importance <- varImp(randomForestModel)
print(importance)

# Step 6: Visualization
# Confusion matrix
confusionMatrix(votingPred, testData$label)

# Model performance comparison
results <- data.frame(
  Model = c("Logistic Regression", "Naive Bayes", "SVM", "Random Forest", "GBM", "Voting Ensemble"),
  Accuracy = c(
    sum(logisticPred == testData$label) / length(testData$label),
    sum(naiveBayesPred == testData$label) / length(testData$label),
    sum(svmPred == testData$label) / length(testData$label),
    sum(randomForestPred == testData$label) / length(testData$label),
    sum(gbmPred == testData$label) / length(testData$label),
    sum(votingPred == testData$label) / length(testData$label)
  )
)
ggplot(results, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  xlab("Model") +
  ylab("Accuracy") +
  ggtitle("Model Performance Comparison")
