# =====================================================
# 05_lda_topic_modeling.R
# Purpose: Perform LDA topic modeling on review text
# =====================================================

library(readr)
library(dplyr)
library(ggplot2)
library(tidytext)
library(topicmodels)
library(tm)

# -----------------------------
# Load cleaned dataset
# -----------------------------
reviews_clean <- read_csv("Data/Processed/cleaned_reviews.csv", show_col_types = FALSE)

cat("Cleaned dataset loaded successfully.\n")
cat("Dataset dimensions:\n")
print(dim(reviews_clean))

# -----------------------------
# Keep only valid text rows
# -----------------------------
reviews_clean <- reviews_clean %>%
  filter(!is.na(review_text), review_text != "")

# Optional: sample to reduce memory pressure
# You can increase this if your machine is fine
set.seed(42)
reviews_sample <- reviews_clean %>%
  sample_n(min(5000, nrow(reviews_clean)))

cat("\nSampled dataset dimensions for LDA:\n")
print(dim(reviews_sample))

# -----------------------------
# Create document IDs
# -----------------------------
reviews_sample <- reviews_sample %>%
  mutate(doc_id = row_number())

# -----------------------------
# Tokenize text into tidy format
# -----------------------------
tidy_reviews <- reviews_sample %>%
  select(doc_id, review_text) %>%
  unnest_tokens(word, review_text)

# -----------------------------
# Remove stopwords
# -----------------------------
data("stop_words")

tidy_reviews <- tidy_reviews %>%
  anti_join(stop_words, by = "word") %>%
  filter(grepl("^[a-z]+$", word))

cat("\nTokenized and cleaned word rows:\n")
print(dim(tidy_reviews))

# -----------------------------
# Count word frequencies per document
# -----------------------------
review_word_counts <- tidy_reviews %>%
  count(doc_id, word, sort = TRUE)

# -----------------------------
# Cast to DocumentTermMatrix
# -----------------------------
dtm <- review_word_counts %>%
  cast_dtm(document = doc_id, term = word, value = n)

cat("\nDocument-Term Matrix dimensions:\n")
print(dim(dtm))

# -----------------------------
# Fit LDA model
# -----------------------------
# Choose number of topics
k <- 4

set.seed(42)
lda_model <- LDA(dtm, k = k, control = list(seed = 42))

cat("\nLDA model trained successfully.\n")
cat("Number of topics:\n")
print(k)

# -----------------------------
# Extract top terms per topic
# -----------------------------
lda_terms <- tidy(lda_model, matrix = "beta")

top_terms <- lda_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, desc(beta))

cat("\nTop terms per topic:\n")
print(top_terms)

# -----------------------------
# Save top terms table
# -----------------------------
write_csv(top_terms, "Outputs/tables/lda_top_terms.csv")

# -----------------------------
# Create topic labels for plotting
# -----------------------------
top_terms_plot_data <- top_terms %>%
  mutate(term = reorder_within(term, beta, topic))

# -----------------------------
# Plot top terms per topic
# -----------------------------
lda_plot <- ggplot(top_terms_plot_data, aes(x = term, y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(
    title = "Top Terms by LDA Topic",
    x = "Term",
    y = "Beta (Topic-Word Probability)"
  ) +
  theme_minimal()

print(lda_plot)

ggsave(
  "Outputs/figures/lda_topics.png",
  plot = lda_plot,
  width = 10,
  height = 6,
  dpi = 300
)

# -----------------------------
# Save model
# -----------------------------
saveRDS(lda_model, "Data/Processed/lda_model.rds")

cat("\nLDA outputs saved successfully.\n")
cat("- Outputs/tables/lda_top_terms.csv\n")
cat("- Outputs/figures/lda_topics.png\n")
cat("- Data/Processed/lda_model.rds\n")