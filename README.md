# Multiclass Text Classification with Ensemble Learning

This project demonstrates how to build a modular and efficient machine learning pipeline in R to classify text documents into multiple categories using an ensemble learning approach. The project covers data preparation, feature extraction, model building, ensemble learning, model interpretation, and visualization.

## Dataset

The project uses a labeled text dataset (e.g., news articles, product reviews) in CSV format. The dataset should have a column named "text" containing the text documents and a column named "label" containing the corresponding category labels.

Example dataset: "20_newsgroups.csv"

## Prerequisites

To run the script, make sure you have the following libraries installed in R:

- tidytext
- tm
- quanteda
- caret
- e1071
- randomForest
- gbm
- caretEnsemble
- reshape2
- ggplot2

You can install these libraries by running the following command:

```R
install.packages(c("tidytext", "tm", "quanteda", "caret", "e1071", "randomForest", "gbm", "caretEnsemble", "reshape2", "ggplot2"))
```

## Usage

1. Place your labeled text dataset in CSV format in the same directory as the script.
2. Open the script in an R environment (e.g., RStudio).
3. Run the entire script to load the functions and create the `text_classification_pipeline` function.
4. Use the pipeline function with your dataset:

```R
results <- text_classification_pipeline("your_dataset.csv")
print(results$results)
```

5. The function will return a list containing trained models, predictions, and results. You can access specific parts of the results as needed.

## Customization

The `text_classification_pipeline` function allows for easy customization:

- `test_size`: Proportion of data to use for testing (default: 0.2)
- `use_tfidf`: Whether to use TF-IDF for feature extraction (default: TRUE)
- `use_pca`: Whether to use PCA for dimensionality reduction (default: FALSE)
- `n_components`: Number of PCA components to use if PCA is enabled (default: 100)

Example with customization:

```R
results <- text_classification_pipeline("your_dataset.csv", test_size = 0.3, use_tfidf = FALSE, use_pca = TRUE, n_components = 50)
```

## Methodology

The project follows these main steps:

1. Data Preparation:
   - Load the labeled text dataset
   - Preprocess the text data by cleaning, tokenizing, and removing stop words
   - Split the dataset into training and testing sets

2. Feature Extraction:
   - Convert text into numerical features using TF-IDF (optional)
   - Apply dimensionality reduction using PCA (optional)

3. Model Building:
   - Train multiple machine learning models suitable for multiclass classification
   - Handle errors during model training to ensure pipeline continuity

4. Ensemble Learning:
   - Combine the predictions of multiple models using the voting ensemble technique
   - Evaluate the performance of individual models and the ensemble model

5. Model Interpretation:
   - Interpret the model's predictions using feature importance (if Random Forest model is available)

6. Visualization:
   - Create a bar plot visualizing the accuracy of individual models and the ensemble model

## Results

The script provides the following results:

- Trained models: A list of trained models for each algorithm
- Predictions: Predictions for each model on the test set
- Feature importance: The most influential features identified using the Random Forest model (if available)
- Model performance comparison: A data frame and visualization of model accuracies

## Error Handling

The pipeline includes error handling to ensure that it continues running even if one or more models fail to train. This makes the pipeline more robust and allows it to adapt to different datasets and potential issues.

## Extensibility

The modular design of the pipeline makes it easy to extend or modify:

- Add new preprocessing steps in the `preprocess_text` function
- Implement new feature extraction methods in the `create_features` function
- Add or remove models in the `train_model` function and the main pipeline

## License

This project is licensed under the [MIT License](LICENSE).
