# Purpose: Train and evaluate Naive Bayes model

library(caret)
library(quanteda)
library(quanteda.textmodels)
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)

# Load processed features and labels
train_tfidf <- readRDS("Data/Processed/train_tfidf.rds")
test_tfidf  <- readRDS("Data/Processed/test_tfidf.rds")
y_train     <- readRDS("Data/Processed/y_train.rds")
y_test      <- readRDS("Data/Processed/y_test.rds")

cat("Train TF-IDF dimensions:\n")
print(dim(train_tfidf))

cat("\nTest TF-IDF dimensions:\n")
print(dim(test_tfidf))


# Train Naive Bayes model
nb_model <- textmodel_nb(train_tfidf, y = y_train)

cat("\nNaive Bayes model trained successfully.\n")


# Make predictions
nb_pred <- predict(nb_model, newdata = test_tfidf)

# Convert predictions to factor
nb_pred <- factor(nb_pred, levels = levels(y_test))


# Evaluate model
nb_cm <- confusionMatrix(nb_pred, y_test)

cat("\nNaive Bayes Confusion Matrix:\n")
print(nb_cm)

# Visualize confusion matrix
# Convert confusion matrix to table
cm_table <- as.data.frame(nb_cm$table)

# Create heatmap
cm_plot <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Naive Bayes Confusion Matrix",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Count"
  ) +
  theme_minimal()

print(cm_plot)

# -----------------------------
# Extract overall accuracy
# -----------------------------
accuracy_results <- data.frame(
  Model = "Naive Bayes",
  Accuracy = as.numeric(nb_cm$overall["Accuracy"])
)

print(accuracy_results)

# -----------------------------
# Extract class-level metrics
# -----------------------------
class_metrics <- as.data.frame(nb_cm$byClass)
class_metrics$class <- rownames(class_metrics)
class_metrics$model <- "Naive Bayes"

print(class_metrics)

# -----------------------------
# Save outputs
# -----------------------------
write_csv(accuracy_results, "Outputs/tables/naive_bayes_accuracy.csv")
write_csv(class_metrics, "Outputs/tables/naive_bayes_class_metrics.csv")

saveRDS(nb_model, "Data/Processed/naive_bayes_model.rds")

cat("\nNaive Bayes results saved successfully.\n")
