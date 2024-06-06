# Multiclass Text Classification with Ensemble Learning

This project demonstrates how to build a machine learning model in R to classify text documents into multiple categories using an ensemble learning approach. The project covers data preparation, feature extraction, model building, ensemble learning, model interpretation, and visualization.

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

3. Modify the following line in the script to match your dataset file name:

   ```R
   data <- read.csv("20_newsgroups.csv", stringsAsFactors = FALSE)
   ```

4. Run the entire script to execute the multiclass text classification pipeline.

5. The script will output the following:
   - Feature importance using the Random Forest model
   - Confusion matrix for the ensemble model
   - Visualization of model performance comparison

## Methodology

The project follows these main steps:

1. Data Preparation:
   - Load the labeled text dataset
   - Preprocess the text data by cleaning, tokenizing, and removing stop words
   - Split the dataset into training and testing sets

2. Feature Extraction:
   - Convert text into numerical features using TF-IDF
   - (Optional) Explore dimensionality reduction techniques like PCA

3. Model Building:
   - Train multiple machine learning models suitable for multiclass classification
   - Tune hyperparameters using cross-validation

4. Ensemble Learning:
   - Combine the predictions of multiple models using the voting ensemble technique
   - Evaluate the performance of individual models and the ensemble model

5. Model Interpretation:
   - Interpret the model's predictions using feature importance

6. Visualization:
   - Create visualizations for the confusion matrix and model performance comparison

## Results

The script provides the following results:

- Feature importance: The most influential features for each class are identified using the Random Forest model.
- Confusion matrix: The confusion matrix shows the performance of the ensemble model on the testing set.
- Model performance comparison: A bar plot visualizes the accuracy of individual models and the ensemble model.

## License

This project is licensed under the [MIT License](LICENSE).
