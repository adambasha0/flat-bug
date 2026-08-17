#!/usr/bin/env Rscript
# Simplified AP curve calculation for FlatBug evaluation
# Usage: Rscript compute_ap.R <combined_results.csv>
#
# IMPORTANT: TP/FP/FN classification must be re-computed at each IoU threshold.
# A match with IoU < threshold should count as BOTH FP (bad pred) AND FN (missed GT).

library("data.table", warn.conflicts=FALSE)
library("dplyr", warn.conflicts=FALSE)
library("tidyr", warn.conflicts=FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  input_file <- "eval_results/combined_results.csv"
} else {
  input_file <- args[1]
}

cat("Loading data from:", input_file, "\n")

# Read raw data (don't classify TP/FP/FN yet - depends on IoU threshold)
dt_raw <- fread(input_file, sep = ";") %>%
  as_tibble() %>%
  mutate(
    area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size = sqrt(area),
    # Match type based on indices (not IoU threshold yet)
    match_type = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",   # GT with no prediction
      idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred", # Prediction with no GT  
      idx_1 != -1 & idx_2 != -1 ~ "matched"         # Matched pair
    ),
    conf = ifelse(is.na(conf2), 0, conf2)
  ) %>%
  filter(size >= 32)  # Standard cutoff as in paper

# Function to classify TP/FP/FN at a given IoU threshold
# For matches with IoU < threshold, we need to split into FP + FN rows
classify_at_threshold <- function(data, iou_thresh) {
  # Unmatched GT -> FN
  fn_unmatched <- data %>%
    filter(match_type == "unmatched_gt") %>%
    mutate(result = "FN")
  
  # Unmatched pred -> FP
  fp_unmatched <- data %>%
    filter(match_type == "unmatched_pred") %>%
    mutate(result = "FP")
  
  # Matched pairs with IoU >= threshold -> TP
  tp_matched <- data %>%
    filter(match_type == "matched" & IoU >= iou_thresh) %>%
    mutate(result = "TP")
  
  # Matched pairs with IoU < threshold -> split into FP (pred) and FN (GT)
  bad_matches <- data %>%
    filter(match_type == "matched" & IoU < iou_thresh)
  
  # Create FP rows for bad match predictions
  fp_bad_match <- bad_matches %>%
    mutate(result = "FP")
  
  # Create FN rows for bad match ground truths
  fn_bad_match <- bad_matches %>%
    mutate(result = "FN", conf = 0)  # FN has no confidence
  
  # Combine all
  bind_rows(fn_unmatched, fp_unmatched, tp_matched, fp_bad_match, fn_bad_match)
}

# Show stats at IoU=0.5 threshold
dt <- classify_at_threshold(dt_raw, 0.5)

cat("Total rows after filtering (size >= 32):", nrow(dt_raw), "\n")
cat("At IoU threshold 0.5:\n")
cat("  TP:", sum(dt$result == "TP"), "FP:", sum(dt$result == "FP"), "FN:", sum(dt$result == "FN"), "\n")
cat("  (Note: matches with 0.2 <= IoU < 0.5 count as both FP and FN)\n\n")

# Function to compute area under PR curve (AP)
integrate_curve <- function(x, y) {
  if (length(x) != length(y)) stop("x and y must have the same length")
  if (length(x) == 0) return(0)
  df <- data.frame(x = x, y = y)
  df <- df[order(df$x), ]
  if (any(duplicated(df$x))) {
    df <- aggregate(y ~ x, data = df, FUN = max)
    df <- df[order(df$x), ]
  }
  if (nrow(df) < 2) return(0)
  dx <- diff(df$x)
  avg_y <- (head(df$y, -1) + tail(df$y, -1)) / 2
  area <- sum(dx * avg_y)
  return(area)
}

# Function to compute AP at a given IoU threshold
compute_ap_at_threshold <- function(data, iou_thresh) {
  # Classify TP/FP/FN at this threshold
  dt_thresh <- classify_at_threshold(data, iou_thresh)
  
  n_tp <- sum(dt_thresh$result == "TP")
  n_fp <- sum(dt_thresh$result == "FP")
  n_fn <- sum(dt_thresh$result == "FN")
  
  if (n_tp == 0) return(0)
  
  # Compute PR curve
  PR_curve <- dt_thresh %>%
    arrange(desc(conf)) %>%
    mutate(
      tp_cumsum = cumsum(result == "TP"),
      fp_cumsum = cumsum(result == "FP"),
      recall_curve = tp_cumsum / (n_tp + n_fn),
      precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
    )
  
  integrate_curve(PR_curve$recall_curve, PR_curve$precision_curve)
}

# Compute AP50
AP50 <- compute_ap_at_threshold(dt_raw, 0.5)

cat("=== Results at IoU=0.5 ===\n")
cat("AP50:", round(AP50, 4), "\n\n")

# Compute AP at multiple IoU thresholds (0.50 to 0.95)
cat("Computing AP at multiple IoU thresholds...\n")
iou_thresholds <- seq(0.50, 0.95, by = 0.01)

AP_results <- sapply(iou_thresholds, function(thresh) {
  compute_ap_at_threshold(dt_raw, thresh)
})

names(AP_results) <- paste0("IoU=", iou_thresholds)

cat("\n=== AP at Each IoU Threshold ===\n")
for (i in seq_along(iou_thresholds)) {
  cat(sprintf("AP@%.2f: %.4f\n", iou_thresholds[i], AP_results[i]))
}

AP50_95 <- mean(AP_results)

cat("\n=== SUMMARY ===\n")
cat("AP50 (IoU=0.50):", round(AP_results[1], 4), "\n")
cat("AP50-95 (mAP):  ", round(AP50_95, 4), "\n")

# Overall precision/recall at IoU=0.5
dt_50 <- classify_at_threshold(dt_raw, 0.5)
n_tp <- sum(dt_50$result == "TP")
n_fp <- sum(dt_50$result == "FP")
n_fn <- sum(dt_50$result == "FN")

precision <- n_tp / (n_tp + n_fp)
recall <- n_tp / (n_tp + n_fn)
f1 <- 2 * precision * recall / (precision + recall)

cat("\n=== Simple Metrics at IoU=0.5 ===\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:   ", round(recall, 4), "\n")
cat("F1 Score: ", round(f1, 4), "\n")
