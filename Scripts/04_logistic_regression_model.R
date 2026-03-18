
# Purpose: Train and evaluate Multinomial Logistic Regression using glmnet
library(caret)
library(readr)
library(dplyr)
library(ggplot2)
library(glmnet)
library(Matrix)


# Load processed features and labels
train_tfidf <- readRDS("Data/Processed/train_tfidf.rds")
test_tfidf  <- readRDS("Data/Processed/test_tfidf.rds")
y_train     <- readRDS("Data/Processed/y_train.rds")
y_test      <- readRDS("Data/Processed/y_test.rds")

cat("Train TF-IDF dimensions:\n")
print(dim(train_tfidf))

cat("\nTest TF-IDF dimensions:\n")
print(dim(test_tfidf))


# Convert to sparse matrices for glmnet
x_train <- as(train_tfidf, "dgCMatrix")
x_test  <- as(test_tfidf, "dgCMatrix")

# Ensure target is factor
y_train <- as.factor(y_train)
y_test  <- as.factor(y_test)

cat("\nSparse training matrix dimensions:\n")
print(dim(x_train))

cat("\nSparse testing matrix dimensions:\n")
print(dim(x_test))


# Train multinomial logistic regression with cross-validation
set.seed(42)

lr_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  type.measure = "class",
  alpha = 0
)

cat("\nMultinomial Logistic Regression model trained successfully.\n")
cat("Best lambda selected:\n")
print(lr_model$lambda.min)


# Make predictions
lr_pred <- predict(lr_model, newx = x_test, s = "lambda.min", type = "class")
lr_pred <- factor(as.vector(lr_pred), levels = levels(y_test))


# Evaluate model
lr_cm <- confusionMatrix(lr_pred, y_test)

cat("\nLogistic Regression Confusion Matrix:\n")
print(lr_cm)


# Visualize confusion matrix heatmap
cm_table <- as.data.frame(lr_cm$table)

cm_plot <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Logistic Regression Confusion Matrix",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Count"
  ) +
  theme_minimal()

print(cm_plot)

ggsave(
  "Outputs/figures/logistic_regression_confusion_matrix.png",
  plot = cm_plot,
  width = 6,
  height = 5,
  dpi = 300
)


# Extract overall accuracy
accuracy_results <- data.frame(
  Model = "Logistic Regression",
  Accuracy = as.numeric(lr_cm$overall["Accuracy"]),
  Kappa = as.numeric(lr_cm$overall["Kappa"])
)

print(accuracy_results)


# Extract class-level metrics
class_metrics <- as.data.frame(lr_cm$byClass)
class_metrics$class <- rownames(class_metrics)
class_metrics$model <- "Logistic Regression"

print(class_metrics)


# Save outputs
write_csv(accuracy_results, "Outputs/tables/logistic_regression_accuracy.csv")
write_csv(class_metrics, "Outputs/tables/logistic_regression_class_metrics.csv")

saveRDS(lr_model, "Data/Processed/logistic_regression_model_glmnet.rds")

cat("\nLogistic Regression results saved successfully.\n")