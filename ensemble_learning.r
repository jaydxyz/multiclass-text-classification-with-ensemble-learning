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

# Function to preprocess text data
preprocess_text <- function(corpus) {
  corpus %>%
    tm_map(tolower) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords("english")) %>%
    tm_map(stemDocument)
}

# Function to create features
create_features <- function(dtm, use_tfidf = TRUE) {
  if (use_tfidf) {
    return(weightTfIdf(dtm))
  } else {
    return(dtm)
  }
}

# Function to train individual models
train_model <- function(X, y, method, ctrl) {
  tryCatch({
    train(X, y, method = method, trControl = ctrl)
  }, error = function(e) {
    message(paste("Error training", method, "model:", e$message))
    NULL
  })
}

# Function to make predictions
predict_model <- function(model, newdata) {
  if (!is.null(model)) {
    predict(model, newdata)
  } else {
    NULL
  }
}

# Main function
text_classification_pipeline <- function(data_file, test_size = 0.2, use_tfidf = TRUE, use_pca = FALSE, n_components = 100) {
  # Step 1: Data Preparation
  data <- read.csv(data_file, stringsAsFactors = FALSE)
  corpus <- Corpus(VectorSource(data$text))
  corpus <- preprocess_text(corpus)
  dtm <- DocumentTermMatrix(corpus)
  
  # Split the dataset
  set.seed(123)
  trainIndex <- createDataPartition(data$label, p = 1 - test_size, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  
  # Step 2: Feature Extraction
  features <- create_features(dtm, use_tfidf)
  trainMatrix <- as.matrix(features[trainIndex, ])
  testMatrix <- as.matrix(features[-trainIndex, ])
  
  if (use_pca) {
    pca <- prcomp(trainMatrix)
    trainMatrix <- pca$x[, 1:n_components]
    testMatrix <- predict(pca, testMatrix)[, 1:n_components]
  }
  
  # Step 3: Model Building
  ctrl <- trainControl(method = "cv", number = 5)
  models <- list(
    logistic = train_model(trainMatrix, trainData$label, "multinom", ctrl),
    naive_bayes = train_model(trainMatrix, trainData$label, "naive_bayes", ctrl),
    svm = train_model(trainMatrix, trainData$label, "svmLinear", ctrl),
    random_forest = train_model(trainMatrix, trainData$label, "rf", ctrl),
    gbm = train_model(trainMatrix, trainData$label, "gbm", ctrl)
  )
  
  # Step 4: Ensemble Learning
  valid_models <- models[!sapply(models, is.null)]
  if (length(valid_models) > 1) {
    ensemble <- caretEnsemble(valid_models, metric = "Accuracy", trControl = ctrl)
  } else {
    ensemble <- NULL
    message("Not enough valid models for ensemble learning.")
  }
  
  # Make predictions
  predictions <- lapply(models, predict_model, newdata = testMatrix)
  if (!is.null(ensemble)) {
    predictions$ensemble <- predict(ensemble, testMatrix)
  }
  
  # Step 5: Model Interpretation
  if (!is.null(models$random_forest)) {
    importance <- varImp(models$random_forest)
    print(importance)
  }
  
  # Step 6: Visualization
  results <- data.frame(
    Model = names(predictions),
    Accuracy = sapply(predictions, function(pred) sum(pred == testData$label) / length(testData$label))
  )
  
  ggplot(results, aes(x = Model, y = Accuracy)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    theme_minimal() +
    xlab("Model") +
    ylab("Accuracy") +
    ggtitle("Model Performance Comparison")
  
  return(list(models = models, ensemble = ensemble, predictions = predictions, results = results))
}

# Run the pipeline
results <- text_classification_pipeline("20_newsgroups.csv")
print(results$results)
