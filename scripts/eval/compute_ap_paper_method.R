#!/usr/bin/env Rscript
# AP calculation using the SAME method as the paper (ap_curve.R)
# This replicates the paper's approach for comparison

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
cat("Using PAPER METHOD (same as scripts/manuscript/statistics/ap_curve.R)\n\n")

# Read data - EXACTLY like the paper does
dt <- fread(input_file, sep = ";") %>%
  as_tibble() %>%
  mutate(
    area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size = sqrt(area),
    # Paper method: classify based on idx only, NOT IoU threshold
    result = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "FN",
      idx_1 == -1 & idx_2 != -1 ~ "FP",
      idx_1 != -1 & idx_2 != -1 ~ "TP"  # ALL matches are TP!
    ),
    conf = replace_na(conf2, 0)
  ) %>%
  filter(size >= 32)

cat("Total rows after filtering (size >= 32):", nrow(dt), "\n")
cat("TP:", sum(dt$result == "TP"), "FP:", sum(dt$result == "FP"), "FN:", sum(dt$result == "FN"), "\n")
cat("(Note: Paper method counts ALL matches as TP regardless of IoU)\n\n")

# Function to compute area under PR curve (AP)
integrate_curve <- function(x, y) {
  if (length(x) != length(y)) stop("x and y must have the same length")
  df <- data.frame(x = x, y = y)
  df <- df[order(df$x), ]
  if (any(duplicated(df$x))) {
    df <- aggregate(y ~ x, data = df, FUN = max)
    df <- df[order(df$x), ]
  }
  dx <- diff(df$x)
  avg_y <- (head(df$y, -1) + tail(df$y, -1)) / 2
  area <- sum(dx * avg_y)
  return(area)
}

# Compute AP at multiple IoU thresholds - PAPER METHOD
# Filter: keep only rows with IoU >= threshold OR result == "FN"
# This removes low-IoU TPs but does NOT create new FN/FP entries
iou_thresholds <- seq(0.50, 0.95, by = 0.01)

AP_results <- sapply(iou_thresholds, function(thresh) {
  filtered <- dt %>%
    filter(IoU >= thresh | result == "FN") %>%  # Paper's filter
    arrange(desc(conf)) %>%
    mutate(
      recall_curve = cumsum(result == "TP") / sum(result != "FP"),
      precision_curve = cumsum(result == "TP") / cumsum(result != "FN")
    )
  
  if (nrow(filtered) > 0) {
    ap <- integrate_curve(filtered$recall_curve, filtered$precision_curve)
  } else {
    ap <- 0
  }
  return(ap)
})

names(AP_results) <- paste0("IoU=", iou_thresholds)

cat("=== AP at Each IoU Threshold (Paper Method) ===\n")
for (i in seq(1, length(iou_thresholds), by = 5)) {
  cat(sprintf("AP@%.2f: %.4f\n", iou_thresholds[i], AP_results[i]))
}

AP50 <- AP_results[1]
AP50_95 <- mean(AP_results)

cat("\n=== SUMMARY (Paper Method) ===\n")
cat("AP50 (IoU=0.50):", round(AP50, 4), "\n")
cat("AP50-95 (mAP):  ", round(AP50_95, 4), "\n")

# Compare with paper's reported values
cat("\n=== Comparison with Paper Figure S3 ===\n")
cat("Paper reports for Model L: AP50=0.937, AP50-95=0.872\n")
cat("Our reproduction:          AP50=", round(AP50, 3), ", AP50-95=", round(AP50_95, 3), "\n")
