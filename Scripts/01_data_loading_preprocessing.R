# Purpose: Load dataset and perform initial preprocessing
# Load required libraries
library(readr)
library(dplyr)
library(stringr)
library(janitor)
library(ggplot2)

# Load dataset
file_path <- "Data/Raw/Amazon_Reviews.csv"

reviews <- read_csv(file_path, show_col_types = FALSE)

cat("Dataset loaded successfully\n")
cat("Dataset dimensions:\n")
print(dim(reviews))

cat("\nColumn names:\n")
print(colnames(reviews))

# Clean column names
reviews <- clean_names(reviews)

cat("\nColumn names:\n")
print(colnames(reviews))

# Removing duplicates
reviews_clean <- reviews %>%
  distinct()


# Removing missing values
reviews_clean <- reviews_clean %>%
  filter(!is.na(`review_text`), !is.na(rating))


# Convert rating text to numeric
reviews_clean <- reviews_clean %>%
  mutate(
    rating = as.numeric(str_extract(rating, "\\d"))
  )

# Create sentiment labels
reviews_clean <- reviews_clean %>%
  mutate(
    sentiment = case_when(
      rating %in% c(1, 2) ~ "negative",
      rating == 3 ~ "neutral",
      rating %in% c(4, 5) ~ "positive",
      TRUE ~ NA_character_
    )
  )

reviews_clean$sentiment <- as.factor(reviews_clean$sentiment)



# Basic text cleaning
reviews_clean <- reviews_clean %>%
  mutate(
    review_text = str_to_lower(review_text),
    review_text = str_replace_all(review_text, "[^a-z\\s]", " "),
    review_text = str_squish(review_text)
  )

table(reviews_clean$rating)


# Sentiment distribution check
cat("\nSentiment distribution:\n")
print(table(reviews_clean$sentiment))

# Bar plot of Sentiment Distribution
ggplot(reviews_clean, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  theme_minimal() +
  labs(
    title = "Sentiment Distribution of Amazon Reviews",
    x = "Sentiment Category",
    y = "Number of Reviews"
  )

# Save bar plot
ggsave(
  "Outputs/figures/sentiment_distribution.png",
  width = 8,
  height = 5
)

# Save cleaned dataset
write_csv(
  reviews_clean,
  "Data/Processed/cleaned_reviews.csv"
)

cat("\nCleaned dataset saved to data/processed/cleaned_reviews.csv\n")
