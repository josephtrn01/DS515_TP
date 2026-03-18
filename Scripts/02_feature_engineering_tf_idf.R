# Purpose: Create train/test split and build TF-IDF features

library(readr)
library(dplyr)
library(caret)
library(quanteda)
library(ggplot2)


# Load cleaned dataset
reviews_clean <- read_csv("Data/Processed/cleaned_reviews.csv", show_col_types = FALSE)

cat("Cleaned dataset loaded successfully.\n")
cat("Dataset dimensions:\n")
print(dim(reviews_clean))


# Keep only valid rows
reviews_clean <- reviews_clean %>%
  filter(!is.na(sentiment), !is.na(review_text), review_text != "")

reviews_clean$sentiment <- as.factor(reviews_clean$sentiment)

cat("\nSentiment distribution before split:\n")
print(table(reviews_clean$sentiment))

sentiment_plot_before_split <- ggplot(reviews_clean, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  theme_minimal() +
  labs(
    title = "Sentiment Distribution Before Train-Test Split",
    x = "Sentiment Category",
    y = "Number of Reviews"
  )

print(sentiment_plot_before_split)

ggsave(
  "Outputs/figures/sentiment_distribution_before_split.png",
  sentiment_plot_before_split,
  width = 8,
  height = 5
)


# Train-test split
set.seed(123)

train_index <- createDataPartition(reviews_clean$sentiment, p = 0.80, list = FALSE)

train_data <- reviews_clean[train_index, ]
test_data  <- reviews_clean[-train_index, ]

cat("\nTrain set dimensions:\n")
print(dim(train_data))

cat("\nTest set dimensions:\n")
print(dim(test_data))

cat("\nTrain sentiment distribution:\n")
print(table(train_data$sentiment))

cat("\nTest sentiment distribution:\n")
print(table(test_data$sentiment))


# Create corpora
train_corpus <- corpus(train_data, text_field = "review_text")
test_corpus  <- corpus(test_data, text_field = "review_text")


# Tokenization
train_tokens <- tokens(
  train_corpus,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE
)

test_tokens <- tokens(
  test_corpus,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE
)


# Remove stopwords
train_tokens <- tokens_remove(train_tokens, stopwords("en"))
test_tokens  <- tokens_remove(test_tokens, stopwords("en"))


# Create document-feature matrices
train_dfm <- dfm(train_tokens)
test_dfm  <- dfm(test_tokens)

# Match test features to train vocabulary
test_dfm <- dfm_match(test_dfm, features = featnames(train_dfm))

cat("\nTrain DFM dimensions:\n")
print(dim(train_dfm))

cat("\nTest DFM dimensions:\n")
print(dim(test_dfm))


# Apply TF-IDF weighting
train_tfidf <- dfm_tfidf(train_dfm)
test_tfidf  <- dfm_tfidf(test_dfm)

cat("\nTrain TF-IDF dimensions:\n")
print(dim(train_tfidf))

cat("\nTest TF-IDF dimensions:\n")
print(dim(test_tfidf))

# -----------------------------
# Save outputs
# -----------------------------
saveRDS(train_tfidf, "Data/Processed/train_tfidf.rds")
saveRDS(test_tfidf, "Data/Processed/test_tfidf.rds")
saveRDS(train_data$sentiment, "Data/Processed/y_train.rds")
saveRDS(test_data$sentiment, "Data/Processed/y_test.rds")

write_csv(train_data, "Data/Processed/train_data.csv")
write_csv(test_data, "Data/Processed/test_data.csv")

cat("\nFiles saved successfully:\n")
cat("- Data/Processed/train_tfidf.rds\n")
cat("- Data/Processed/test_tfidf.rds\n")
cat("- Data/Processed/y_train.rds\n")
cat("- Data/Processed/y_test.rds\n")
cat("- Data/Processed/train_data.csv\n")
cat("- Data/Processed/test_data.csv\n")
