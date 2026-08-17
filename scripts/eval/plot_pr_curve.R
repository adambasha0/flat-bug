#!/usr/bin/env Rscript
# Precision-Recall curve plotting for FlatBug evaluation
# Analogous to compute_ap.R but generates plot with formatted axes
# Usage: Rscript plot_pr_curve.R <combined_results.csv> [output_dir]

library("data.table", warn.conflicts=FALSE)
library("dplyr", warn.conflicts=FALSE)
library("ggplot2", warn.conflicts=FALSE)
suppressPackageStartupMessages(library("scales"))

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

# Read data
dt <- fread(input_file, sep = ";") %>%
  as_tibble() %>%
  mutate(
    area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size = sqrt(area),
    result = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "FN",
      idx_1 == -1 & idx_2 != -1 ~ "FP",
      idx_1 != -1 & idx_2 != -1 ~ "TP"
    ),
    conf = ifelse(is.na(conf2), 0, conf2)
  ) %>%
  filter(size >= 32)  # Standard cutoff as in paper

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

# Compute PR curve at IoU=0.5 (using the pre-matched results from fb_evaluate)
PR_curve <- dt %>%
  arrange(desc(conf)) %>%
  mutate(
    recall_curve = cumsum(result == "TP") / sum(result != "FP"),
    precision_curve = cumsum(result == "TP") / cumsum(result != "FN")
  )

AP50 <- integrate_curve(PR_curve$recall_curve, PR_curve$precision_curve)

# Compute AP at multiple IoU thresholds (0.50 to 0.95)
iou_thresholds <- seq(0.50, 0.95, by = 0.05)

AP_results <- sapply(iou_thresholds, function(thresh) {
  filtered <- dt %>%
    filter(IoU >= thresh | result == "FN") %>%
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

AP50_95 <- mean(AP_results)

# Overall precision/recall at IoU=0.5
precision <- sum(dt$result == "TP") / sum(dt$result != "FN")
recall <- sum(dt$result == "TP") / sum(dt$result != "FP")
f1 <- 2 * precision * recall / (precision + recall)

# Prepare data for plotting - add starting point (0, 1)
plot_data <- bind_rows(
  tibble(recall_curve = 0, precision_curve = 1),
  PR_curve %>% select(recall_curve, precision_curve)
)

# Create the plot
# Create the plot with exact borders
p <- ggplot(plot_data, aes(x = recall_curve, y = precision_curve)) +
  geom_line(color = "#1f77b4", linewidth = 1) +
  
  # X-axis: 0 to 100% with no padding at borders
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0, 1, by = 0.1),
    limits = c(0, 1),
    expand = expansion(0, 0)  # Removes space between axis and data
  ) +
  
  # Y-axis: 93% to 100% with no padding at borders
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0.93, 1, by = 0.01),
    limits = c(0.93, 1),
    expand = expansion(0, 0)  # Removes space between axis and data
  ) +
  
  labs(
    x = "Recall",
    y = "Precision",
    title = "Precision-Recall Curve (IoU=0.50)"
  ) +
  
  # Adjust annotation position to fit the new zoomed area
  annotate(
    "label",
    x = 0.05, y = 0.94, 
    label = sprintf("AP50: %.4f\nAP50-95: %.4f", AP50, AP50_95),
    hjust = 0, vjust = 0,
    size = 3.5,
    fill = "white",
    label.size = 0.5
  ) +
  
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", linetype = "dashed"),
    plot.title = element_text(hjust = 0.5)
  )

# Save plots
png_path <- file.path(output_dir, "pr_curve_r.png")
pdf_path <- file.path(output_dir, "pr_curve_r.pdf")

ggsave(png_path, p, width = 6, height = 6, dpi = 200)
ggsave(pdf_path, p, width = 6, height = 6)

# Print outputs
cat("Saved PR images:", png_path, "and", pdf_path, "\n")
cat("AP50:", round(AP50, 4), "\n")
cat("AP50-95:", round(AP50_95, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1:", round(f1, 4), "\n")
