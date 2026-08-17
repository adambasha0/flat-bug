## PR curve for SAM3 fine-tuned model (checkpoint 18, score threshold 0.005, mask threshold 0.50)
## Produces:
##   figures/sam3_lora_checkpoint10_score005_mask_vs_bbox.png  – combined mask vs bbox PR curve

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
# Helpers
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
    df <- aggregate(y ~ x, data = df, FUN = max)
    df <- df[order(df$x), ]
  }
  dx <- diff(df$x)
  avg_y <- (head(df$y, -1) + tail(df$y, -1)) / 2
  sum(dx * avg_y)
}

classify_at_threshold <- function(data, iou_thresh, iou_col = "IoU") {
  fn_unmatched <- data %>% filter(match_type == "unmatched_gt")   %>% mutate(result = "FN")
  fp_unmatched <- data %>% filter(match_type == "unmatched_pred") %>% mutate(result = "FP")
  tp_matched   <- data %>% filter(match_type == "matched" & .data[[iou_col]] >= iou_thresh) %>% mutate(result = "TP")
  bad_matches  <- data %>% filter(match_type == "matched" & .data[[iou_col]] <  iou_thresh)
  fp_bad <- bad_matches %>% mutate(result = "FP")
  fn_bad <- bad_matches %>% mutate(result = "FN", conf = 0)
  bind_rows(fn_unmatched, fp_unmatched, tp_matched, fp_bad, fn_bad)
}

format_ap_label <- function(name, ap50, ap5095, width = 22) {
  name_pad <- stringr::str_pad(paste0(name, ":"), width = width, side = "right")
  ap50_str <- paste0(format(round(ap50 * 100, 2), nsmall = 2), "%")
  ap95_str <- paste0(format(round(ap5095 * 100, 2), nsmall = 2), "%")
  str_c(name_pad, "AP50: ", ap50_str, "   |   AP50-95: ", ap95_str)
}

# --------------------------------------------------------------------------- #
# Load fine-tuned CSV (score 0.005, mask 0.50)
# --------------------------------------------------------------------------- #
ft_data <- data.table::fread(
  "data/finetuned/lora/fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_with_bb.csv"
) %>%
  as_tibble() %>%
  mutate(model = "SAM3-LoRA") %>%
  rowwise() %>%
  mutate(
    IoU_bb = {
      if (idx_1 != -1 & idx_2 != -1) compute_bbox_iou(parse_bbox(bbox_1), parse_bbox(bbox_2))
      else NA_real_
    }
  ) %>%
  ungroup() %>%
  mutate(
    area      = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size      = sqrt(area),
    match_type = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",
      idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred",
      idx_1 != -1 & idx_2 != -1 ~ "matched"
    ),
    conf = replace_na(conf2, 0.)
  ) %>%
  filter(size >= 32)

# --------------------------------------------------------------------------- #
# PR curves at IoU=0.50
# --------------------------------------------------------------------------- #
PR_curve_mask <- ft_data %>%
  group_modify(~ classify_at_threshold(.x, 0.5, "IoU")) %>%
  arrange(desc(conf)) %>%
  mutate(
    n_tp = sum(result == "TP"),
    n_fn = sum(result == "FN"),
    tp_cumsum = cumsum(result == "TP"),
    fp_cumsum = cumsum(result == "FP"),
    recall_curve    = tp_cumsum / (n_tp + n_fn),
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum),
    iou_type = "IoU (mask)"
  ) %>%
  ungroup()

PR_curve_bb <- ft_data %>%
  group_modify(~ classify_at_threshold(.x, 0.5, "IoU_bb")) %>%
  arrange(desc(conf)) %>%
  mutate(
    n_tp = sum(result == "TP"),
    n_fn = sum(result == "FN"),
    tp_cumsum = cumsum(result == "TP"),
    fp_cumsum = cumsum(result == "FP"),
    recall_curve    = tp_cumsum / (n_tp + n_fn),
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum),
    iou_type = "IoU (bounding box)"
  ) %>%
  ungroup()

# --------------------------------------------------------------------------- #
# Compute AP50 and AP50-95
# --------------------------------------------------------------------------- #
compute_ap_at_threshold <- function(data, iou_thresh, iou_col = "IoU") {
  dt <- classify_at_threshold(data, iou_thresh, iou_col)
  n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
  if (n_tp == 0) return(0)
  PR <- dt %>%
    arrange(desc(conf)) %>%
    mutate(
      tp_cumsum = cumsum(result == "TP"),
      fp_cumsum = cumsum(result == "FP"),
      recall_curve    = tp_cumsum / (n_tp + n_fn),
      precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
    )
  integrate_curve(PR$recall_curve, PR$precision_curve)
}

ap50_mask   <- compute_ap_at_threshold(ft_data, 0.5,  "IoU")
ap50_bb     <- compute_ap_at_threshold(ft_data, 0.5,  "IoU_bb")
ap5095_mask <- mean(sapply(seq(0.5, 0.95, 0.05),
                           function(t) compute_ap_at_threshold(ft_data, t, "IoU")))
ap5095_bb   <- mean(sapply(seq(0.5, 0.95, 0.05),
                           function(t) compute_ap_at_threshold(ft_data, t, "IoU_bb")))

cat("\n========== SAM3 LoRA (checkpoint 10, score 0.005, mask 0.50) ==========\n")
cat("IoU (mask):\n")
cat("  AP50:    ", format(round(ap50_mask  * 100, 2), nsmall = 2), "%\n")
cat("  AP50-95: ", format(round(ap5095_mask * 100, 2), nsmall = 2), "%\n\n")
cat("IoU (bounding box):\n")
cat("  AP50:    ", format(round(ap50_bb    * 100, 2), nsmall = 2), "%\n")
cat("  AP50-95: ", format(round(ap5095_bb  * 100, 2), nsmall = 2), "%\n")
cat("=============================================================================\n\n")

AP_results <- tibble(
  iou_type = c("IoU (mask)", "IoU (bounding box)"),
  AP50     = c(ap50_mask, ap50_bb),
  AP50_95  = c(ap5095_mask, ap5095_bb)
)

# --------------------------------------------------------------------------- #
# Combined PR plot
# --------------------------------------------------------------------------- #
PR_curve_combined <- bind_rows(
  PR_curve_mask %>% select(recall_curve, precision_curve, iou_type),
  PR_curve_bb   %>% select(recall_curve, precision_curve, iou_type)
)

PR_plt <- PR_curve_combined %>%
  left_join(
    AP_results %>%
      mutate(label = format_ap_label(iou_type, AP50, AP50_95) %>% factor(., .)) %>%
      select(iou_type, label),
    by = "iou_type"
  ) %>%
  ggplot(aes(recall_curve, precision_curve, color = label)) +
  geom_line(linewidth = 1) +
  scale_x_continuous(
    labels = scales::label_percent(.1, drop0trailing = TRUE),
    expand = expansion(), limits = 0:1, n.breaks = 11
  ) +
  scale_y_continuous(
    labels = scales::label_percent(.1, drop0trailing = TRUE),
    expand = expansion(), limits = 0:1, n.breaks = 11
  ) +
  scale_color_manual(values = c("#E69F00", "#56B4E9")) +
  labs(x = "Recall", y = "Precision", color = NULL) +
  guides(color = guide_legend(keywidth = unit(1.5, "lines"), keyheight = unit(1, "lines"))) +
  theme(
    legend.text             = element_text(family = "Courier New", size = 13),
    legend.position         = "inside",
    legend.position.inside  = c(0.5, 0.01),
    legend.justification    = c(0.5, 0),
    legend.background       = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.key              = element_rect(fill = "transparent"),
    legend.spacing.x        = unit(0.2, "lines"),
    legend.margin           = margin(4, 6, 4, 6),
    plot.margin             = margin(0.5, 1, 0, 0.25, "lines"),
    axis.text               = element_text(size = 14),
    axis.title              = element_text(size = 15)
  )

ggsave(
  "figures/sam3_lora_checkpoint10_score005_mask_vs_bbox.png",
  PR_plt,
  device = "png",
  width = 4, height = 4,
  scale = 2.75,
  dpi = 300
)

cat("Saved: figures/sam3_lora_checkpoint10_score005_mask_vs_bbox.png\n")
