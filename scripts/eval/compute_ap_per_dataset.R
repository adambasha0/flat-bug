#!/usr/bin/env Rscript
# Compute AP50 and AP50-95 per dataset for FlatBug evaluation
# Usage: Rscript compute_ap_per_dataset.R <combined_results.csv> [output_dir]
#
# IMPORTANT: TP/FP/FN classification must be re-computed at each IoU threshold.
# A match with IoU < threshold should count as BOTH FP (bad pred) AND FN (missed GT).

library("data.table", warn.conflicts=FALSE)
library("dplyr", warn.conflicts=FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  input_file <- "eval_results/combined_results.csv"
} else {
  input_file <- args[1]
}

if (length(args) >= 2) {
  output_dir <- args[2]
} else {
  output_dir <- dirname(input_file)
}

# Create output directory if needed
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Read raw data (don't classify TP/FP/FN yet - depends on IoU threshold)
dt <- fread(input_file, sep = ";") %>%
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
    conf = ifelse(is.na(conf2), 0, conf2),
    # Extract dataset name from image (first part before underscore, handling special cases)
    dataset = sub("^([^_]+(?:-[^_]+)*)_.*$", "\\1", image)
  ) %>%
  filter(size >= 32)  # Standard cutoff as in paper

# Function to classify TP/FP/FN at a given IoU threshold
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

# Function to compute AP50 for a dataset subset
compute_ap50 <- function(data) {
  compute_ap_at_threshold(data, 0.5)
}

# Function to compute AP50-95 for a dataset subset
compute_ap50_95 <- function(data) {
  if (nrow(data) == 0) return(0)
  
  iou_thresholds <- seq(0.50, 0.95, by = 0.05)
  AP_results <- sapply(iou_thresholds, function(thresh) {
    compute_ap_at_threshold(data, thresh)
  })
  
  return(mean(AP_results))
}

# Get unique datasets
datasets <- sort(unique(dt$dataset))

# Compute metrics per dataset
results <- data.frame(
  dataset = character(),
  AP50 = numeric(),
  AP50_95 = numeric(),
  n_images = integer(),
  n_TP = integer(),
  n_FP = integer(),
  n_FN = integer(),
  stringsAsFactors = FALSE
)

cat("\n===== Per-Dataset AP Metrics =====\n\n")
cat(sprintf("%-25s %8s %10s %8s %6s %6s %6s\n", "Dataset", "AP50", "AP50-95", "Images", "TP", "FP", "FN"))
cat(paste(rep("-", 80), collapse = ""), "\n")

for (ds in datasets) {
  ds_data <- dt %>% filter(dataset == ds)
  
  ap50 <- compute_ap50(ds_data)
  ap50_95 <- compute_ap50_95(ds_data)
  n_images <- length(unique(ds_data$image))
  
  # Get TP/FP/FN at IoU=0.5 threshold
  ds_classified <- classify_at_threshold(ds_data, 0.5)
  n_tp <- sum(ds_classified$result == "TP")
  n_fp <- sum(ds_classified$result == "FP")
  n_fn <- sum(ds_classified$result == "FN")
  
  results <- rbind(results, data.frame(
    dataset = ds,
    AP50 = ap50,
    AP50_95 = ap50_95,
    n_images = n_images,
    n_TP = n_tp,
    n_FP = n_fp,
    n_FN = n_fn
  ))
  
  cat(sprintf("%-25s %8.4f %10.4f %8d %6d %6d %6d\n", ds, ap50, ap50_95, n_images, n_tp, n_fp, n_fn))
}

cat(paste(rep("-", 80), collapse = ""), "\n")

# Compute overall metrics
overall_ap50 <- compute_ap50(dt)
overall_ap50_95 <- compute_ap50_95(dt)
n_images_total <- length(unique(dt$image))

# Get TP/FP/FN at IoU=0.5 threshold for overall
dt_classified <- classify_at_threshold(dt, 0.5)
n_tp_total <- sum(dt_classified$result == "TP")
n_fp_total <- sum(dt_classified$result == "FP")
n_fn_total <- sum(dt_classified$result == "FN")

cat(sprintf("%-25s %8.4f %10.4f %8d %6d %6d %6d\n", "OVERALL", overall_ap50, overall_ap50_95, n_images_total, n_tp_total, n_fp_total, n_fn_total))

# Save results to CSV
output_csv <- file.path(output_dir, "ap_per_dataset.csv")
write.csv(results, output_csv, row.names = FALSE)
cat("\n\nResults saved to:", output_csv, "\n")

# Also print in compact format for easy copy-paste
cat("\n===== Compact Format =====\n")
for (i in 1:nrow(results)) {
  cat(sprintf("%s: AP50=%.4f, AP50-95=%.4f\n", results$dataset[i], results$AP50[i], results$AP50_95[i]))
}
