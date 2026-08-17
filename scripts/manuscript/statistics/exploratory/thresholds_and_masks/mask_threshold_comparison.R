## Mask threshold comparison for SAM3 (score threshold 0.20)
## Produces:
##   figures/sam3_mask_threshold_pr_curves.png   – overlay PR curves (mask IoU)
##   figures/sam3_mask_threshold_ap_table.csv    – AP50 / AP50-95 for all thresholds

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path_arg <- args[grepl(file_arg, args)]
script_dir <- if (length(script_path_arg) > 0) {
  dirname(normalizePath(sub(file_arg, "", script_path_arg[1])))
} else {
  getwd()
}

oldwd <- getwd()
setwd(script_dir)
on.exit(setwd(oldwd), add = TRUE)

source(file.path("helpers", "flatbug_init.R"))

# --------------------------------------------------------------------------- #
# Helpers (copied from ap_curve_corrected_sam3_005.R)
# --------------------------------------------------------------------------- #
parse_bbox <- function(bbox_str) {
  if (is.na(bbox_str) || bbox_str == "") return(c(NA, NA, NA, NA))
  tryCatch({
    nums <- as.numeric(strsplit(gsub("\\[|\\]", "", bbox_str), ",\\s*")[[1]])
    if (length(nums) == 4) nums else c(NA, NA, NA, NA)
  }, error = function(e) c(NA, NA, NA, NA))
}

compute_bbox_iou <- function(bbox1, bbox2) {
  if (any(is.na(bbox1)) || any(is.na(bbox2))) return(NA_real_)
  x1 <- max(bbox1[1], bbox2[1]); y1 <- max(bbox1[2], bbox2[2])
  x2 <- min(bbox1[3], bbox2[3]); y2 <- min(bbox1[4], bbox2[4])
  if (x2 <= x1 || y2 <= y1) return(0.0)
  intersection <- (x2 - x1) * (y2 - y1)
  area1 <- (bbox1[3] - bbox1[1]) * (bbox1[4] - bbox1[2])
  area2 <- (bbox2[3] - bbox2[1]) * (bbox2[4] - bbox2[2])
  union <- area1 + area2 - intersection
  if (union <= 0) return(0.0)
  intersection / union
}

integrate_curve <- function(x, y) {
  df <- data.frame(x = x, y = y)
  df <- df[order(df$x), ]
  if (any(duplicated(df$x))) {
    warning("Duplicate x values detected. Aggregating by taking the maximum y for each unique x.")
    df <- aggregate(y ~ x, data = df, FUN = max)
    df <- df[order(df$x), ]
  }
  dx <- diff(df$x)
  avg_y <- (head(df$y, -1) + tail(df$y, -1)) / 2
  sum(dx * avg_y)
}

classify_at_threshold <- function(data, iou_thresh, iou_col = "IoU") {
  fn_unmatched <- data %>% filter(match_type == "unmatched_gt")  %>% mutate(result = "FN")
  fp_unmatched <- data %>% filter(match_type == "unmatched_pred") %>% mutate(result = "FP")
  tp_matched   <- data %>% filter(match_type == "matched" & .data[[iou_col]] >= iou_thresh) %>% mutate(result = "TP")
  bad_matches  <- data %>% filter(match_type == "matched" & .data[[iou_col]] < iou_thresh)
  fp_bad <- bad_matches %>% mutate(result = "FP")
  fn_bad <- bad_matches %>% mutate(result = "FN", conf = 0)
  bind_rows(fn_unmatched, fp_unmatched, tp_matched, fp_bad, fn_bad)
}

compute_ap50 <- function(data, iou_col = "IoU") {
  dt <- classify_at_threshold(data, 0.5, iou_col)
  n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
  if (n_tp == 0) return(0)
  PR <- dt %>% arrange(desc(conf)) %>%
    mutate(tp_cumsum = cumsum(result == "TP"),
           fp_cumsum = cumsum(result == "FP"),
           recall_curve    = tp_cumsum / (n_tp + n_fn),
           precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum))
  integrate_curve(PR$recall_curve, PR$precision_curve)
}

compute_ap5095 <- function(data, iou_col = "IoU") {
  mean(sapply(seq(0.5, 0.95, 0.05), function(t) {
    dt <- classify_at_threshold(data, t, iou_col)
    n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
    if (n_tp == 0) return(0)
    PR <- dt %>% arrange(desc(conf)) %>%
      mutate(tp_cumsum = cumsum(result == "TP"),
             fp_cumsum = cumsum(result == "FP"),
             recall_curve    = tp_cumsum / (n_tp + n_fn),
             precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum))
    integrate_curve(PR$recall_curve, PR$precision_curve)
  }))
}

load_csv <- function(path) {
  data.table::fread(path) %>%
    as_tibble() %>%
    mutate(model = "SAM3") %>%
    rowwise() %>%
    mutate(IoU_bb = {
      if (idx_1 != -1 & idx_2 != -1) compute_bbox_iou(parse_bbox(bbox_1), parse_bbox(bbox_2))
      else NA_real_
    }) %>%
    ungroup() %>%
    mutate(
      area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
      size = sqrt(area),
      match_type = case_when(
        idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",
        idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred",
        idx_1 != -1 & idx_2 != -1 ~ "matched"
      ),
      conf = replace_na(conf2, 0.)
    ) %>%
    filter(size >= 32)
}

# --------------------------------------------------------------------------- #
# Dataset definitions – mask threshold, file path, display label
# --------------------------------------------------------------------------- #
datasets <- tribble(
  ~mask_thresh, ~csv_path,
  0.50, "data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_5_combined_results.csv",
  0.35, "data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_035_combined_results.csv",
  0.25, "data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_025_combined_results.csv",
  0.10, "data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_010_combined_results.csv",
  0.005,"data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_005_combined_results.csv",
  5e-4, "data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_0005_combined_results.csv",
  1e-4, "data/thesis_experiments_using_fb_eval_refactored/mask_threshold_experiment/sam3_prediction_results-score_20_mask_0001_combined_results.csv"
)

# --------------------------------------------------------------------------- #
# Compute AP50 / AP50-95 for each dataset
# --------------------------------------------------------------------------- #
results <- datasets %>%
  mutate(
    data     = map(csv_path, load_csv),
    ap50_mask = map_dbl(data, ~ compute_ap50(.x, "IoU")),
    ap50_bb   = map_dbl(data, ~ compute_ap50(.x, "IoU_bb")),
    ap5095_mask = map_dbl(data, ~ compute_ap5095(.x, "IoU")),
    ap5095_bb   = map_dbl(data, ~ compute_ap5095(.x, "IoU_bb"))
  )

# Print table
cat("\n========== MASK THRESHOLD AP RESULTS ==========\n")
cat(sprintf("%-12s  %8s  %8s  %10s  %10s\n",
            "MaskThresh", "AP50_msk", "AP50_bb", "AP5095_msk", "AP5095_bb"))
cat(strrep("-", 56), "\n")
for (i in seq_len(nrow(results))) {
  cat(sprintf("%-12s  %8.4f  %8.4f  %10.4f  %10.4f\n",
              results$mask_thresh[i],
              results$ap50_mask[i], results$ap50_bb[i],
              results$ap5095_mask[i], results$ap5095_bb[i]))
}
cat(strrep("=", 56), "\n\n")

# Save CSV table
write.csv(
  results %>% select(mask_thresh, ap50_mask, ap50_bb, ap5095_mask, ap5095_bb),
  "figures/graphs_thesis_experiments_using_fb_eval_refactored/sam3_mask_threshold_ap_table.csv",
  row.names = FALSE
)

# --------------------------------------------------------------------------- #
# Build overlay PR curves (mask IoU only)
# --------------------------------------------------------------------------- #
pr_all <- results %>%
  mutate(
    pr_curve = map2(data, mask_thresh, function(d, thr) {
      dt <- classify_at_threshold(d, 0.5, "IoU")
      n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
      if (n_tp == 0) return(tibble(recall_curve = numeric(), precision_curve = numeric()))
      dt %>% arrange(desc(conf)) %>%
        mutate(tp_cumsum = cumsum(result == "TP"),
               fp_cumsum = cumsum(result == "FP"),
               recall_curve    = tp_cumsum / (n_tp + n_fn),
               precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)) %>%
        select(recall_curve, precision_curve)
    })
  ) %>%
  select(mask_thresh, ap50_mask, pr_curve) %>%
  unnest(pr_curve) %>%
  mutate(
    label = factor(
      sprintf("mask = %.4f  |  AP50: %.2f%%", mask_thresh, ap50_mask * 100),
      levels = sprintf("mask = %.4f  |  AP50: %.2f%%",
                       results$mask_thresh, results$ap50_mask * 100)
    )
  )

# Colour palette: 7 colours spanning a gradient from dark-blue to red
pal <- c("#08306b","#2171b5","#4dac26","#fd8d3c","#de2d26","#a50f15","#67000d")

p <- pr_all %>%
  ggplot(aes(recall_curve, precision_curve, colour = label)) +
  geom_line(linewidth = 0.85) +
  scale_x_continuous(
    labels = scales::label_percent(.1, drop0trailing = TRUE),
    expand = expansion(), limits = 0:1, n.breaks = 11
  ) +
  scale_y_continuous(
    labels = scales::label_percent(.1, drop0trailing = TRUE),
    expand = expansion(),
    limits = c(0.35, 1),
    breaks = seq(0.35, 1, by = 0.1)
  ) +
  scale_colour_manual(values = pal) +
  labs(
    x      = "Recall",
    y      = "Precision",
    colour = "Mask threshold"
  ) +
  guides(colour = guide_legend(keywidth = unit(1.5, "lines"),
                                keyheight = unit(0.9, "lines"),
                                ncol = 2)) +
  theme(
    legend.text             = element_text(family = "Courier New", size = 9),
    legend.position         = "inside",
    legend.position.inside  = c(0.999, 0.01),
    legend.justification    = c(1, 0),
    legend.background       = element_rect(fill = alpha("white", 0.85), colour = NA),
    legend.key              = element_rect(fill = "transparent"),
    legend.margin           = margin(4, 6, 4, 6),
    plot.margin             = margin(0.5, 1, 0, 0.25, "lines"),
    axis.text               = element_text(size = 17),
    axis.title              = element_text(size = 15),
    plot.title              = element_blank()
  )

ggsave(
  "figures/graphs_thesis_experiments_using_fb_eval_refactored/sam3_mask_threshold_pr_curves_final.png",
  p,
  device = "png",
  width  = 5, height = 4,
  scale  = 2.5,
  dpi    = 300
)

cat("Saved: figures/graphs_thesis_experiments_using_fb_eval_refactored/sam3_mask_threshold_pr_curves_final.png\n")
cat("Saved: figures/graphs_thesis_experiments_using_fb_eval_refactored/sam3_mask_threshold_ap_table.csv\n")
