# =====================================================
# 06_model_comparison_and_summary.R
# Purpose: Compare Naive Bayes and Logistic Regression
# =====================================================

library(readr)
library(dplyr)
library(ggplot2)

# -----------------------------
# Load model accuracy results
# -----------------------------
nb_accuracy <- read_csv("Outputs/tables/naive_bayes_accuracy.csv", show_col_types = FALSE)
lr_accuracy <- read_csv("Outputs/tables/logistic_regression_accuracy.csv", show_col_types = FALSE)

# Combine results
model_comparison <- bind_rows(nb_accuracy, lr_accuracy)

cat("Combined model comparison table:\n")
print(model_comparison)

# Save combined table
write_csv(model_comparison, "Outputs/tables/model_comparison_summary.csv")

# -----------------------------
# Plot model accuracy comparison
# -----------------------------
comparison_plot <- ggplot(model_comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = round(Accuracy, 4)), vjust = -0.5) +
  labs(
    title = "Model Accuracy Comparison",
    x = "Model",
    y = "Accuracy"
  ) +
  theme_minimal()

print(comparison_plot)

ggsave(
  "Outputs/figures/model_accuracy_comparison.png",
  plot = comparison_plot,
  width = 8,
  height = 5,
  dpi = 300
)

cat("\nModel comparison outputs saved successfully.\n")
cat("- Outputs/tables/model_comparison_summary.csv\n")
cat("- Outputs/figures/model_accuracy_comparison.png\n")