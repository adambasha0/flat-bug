## PR curve for SAM3 base model (score 0.005) — saves PNG for thesis
## Uses same font/scale settings as ap_curve_fb_score001_thesis.R for side-by-side consistency.

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

# ── helper functions ──────────────────────────────────────────────────────────
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
  if (length(x) != length(y)) stop("x and y must have the same length")
  df <- data.frame(x = x, y = y)
  df <- df[order(df$x), ]
  if (any(duplicated(df$x))) {
    df <- aggregate(y ~ x, data = df, FUN = max)
    df <- df[order(df$x), ]
  }
  sum(diff(df$x) * (head(df$y, -1) + tail(df$y, -1)) / 2)
}

classify_at_threshold <- function(data, iou_thresh, iou_col = "IoU") {
  fn_unmatched <- data %>% filter(match_type == "unmatched_gt") %>% mutate(result = "FN")
  fp_unmatched <- data %>% filter(match_type == "unmatched_pred") %>% mutate(result = "FP")
  tp_matched   <- data %>% filter(match_type == "matched" & .data[[iou_col]] >= iou_thresh) %>%
    mutate(result = "TP")
  bad_matches  <- data %>% filter(match_type == "matched" & .data[[iou_col]] < iou_thresh)
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

# ── Load data ─────────────────────────────────────────────────────────────────
# IoU_bb is already computed in this CSV
bb_data <- data.table::fread("coco_instances/sam_ep18_mask_05_score_005/sam3_ep18_mask_05_score_005_using_refactored_old_greedy_with_bb.csv") %>%
  as_tibble() %>%
  mutate(model = "SAM3") %>%
  mutate(
    area      = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size      = sqrt(area),
    dataset   = str_extract(image, "^[^_]+"),
    short     = short_name(dataset),
    match_type = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",
      idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred",
      idx_1 != -1 & idx_2 != -1 ~ "matched"
    ),
    conf = replace_na(conf2, 0.)
  ) %>%
  filter(size >= 32) %>%
  select(!c(conf1, conf2))

# ── PR curves ─────────────────────────────────────────────────────────────────
make_pr_curve <- function(data, iou_col, label) {
  data %>%
    group_modify(~ classify_at_threshold(.x, 0.5, iou_col)) %>%
    arrange(desc(conf)) %>%
    mutate(
      n_tp = sum(result == "TP"),
      n_fn = sum(result == "FN"),
      tp_cumsum = cumsum(result == "TP"),
      fp_cumsum = cumsum(result == "FP"),
      recall_curve    = tp_cumsum / (n_tp + n_fn),
      precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum),
      iou_type = label
    ) %>%
    ungroup()
}

PR_curve_mask <- make_pr_curve(bb_data, "IoU",    "IoU (mask)")
PR_curve_bb   <- make_pr_curve(bb_data, "IoU_bb", "IoU (bounding box)")

# ── AP computation ────────────────────────────────────────────────────────────
compute_ap_curve <- function(data, iou_col, iou_type_label) {
  tibble(IoU_thresh = seq(0.5, 0.95, 0.01)) %>%
    mutate(result = map(IoU_thresh, function(t) {
      data %>%
        group_modify(~ {
          dt <- classify_at_threshold(.x, t, iou_col)
          n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
          if (n_tp == 0) return(tibble(AP = 0, P = 0, R = 0))
          PR <- dt %>%
            arrange(desc(conf)) %>%
            mutate(
              tp_cumsum = cumsum(result == "TP"),
              fp_cumsum = cumsum(result == "FP"),
              recall_curve    = tp_cumsum / (n_tp + n_fn),
              precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
            )
          tibble(AP = integrate_curve(PR$recall_curve, PR$precision_curve),
                 w = which.min(abs(PR$precision_curve - 0.975)),
                 P = PR$precision_curve[w],
                 R = PR$recall_curve[w]) %>% select(!w)
        }) %>% ungroup()
    }, .progress = if (show_progress()) paste("AP Curve", iou_type_label) else FALSE)) %>%
    unnest(result) %>%
    rename(IoU = IoU_thresh) %>%
    mutate(iou_type = iou_type_label)
}

AP_curve_mask <- compute_ap_curve(bb_data, "IoU",    "IoU (mask)")
AP_curve_bb   <- compute_ap_curve(bb_data, "IoU_bb", "IoU (bounding box)")
AP_results <- bind_rows(
  AP_curve_mask %>% summarize(model = "SAM3", iou_type = "IoU (mask)",
                               AP50 = AP[IoU == 0.5], AP50_95 = mean(AP[between(IoU, 0.5, 0.95)])),
  AP_curve_bb   %>% summarize(model = "SAM3", iou_type = "IoU (bounding box)",
                               AP50 = AP[IoU == 0.5], AP50_95 = mean(AP[between(IoU, 0.5, 0.95)]))
)

cat("\n========== AP RESULTS — SAM3 base model (score 0.005) ==========\n")
cat(sprintf("  AP50:     %6.2f %%\n",  AP_results$AP50[AP_results$iou_type == "IoU (mask)"]    * 100))
cat(sprintf("  AP50-95:  %6.2f %%\n",  AP_results$AP50_95[AP_results$iou_type == "IoU (mask)"] * 100))
cat(sprintf("  AP50:     %6.2f %%\n",  AP_results$AP50[AP_results$iou_type == "IoU (bounding box)"]    * 100))
cat(sprintf("  AP50-95:  %6.2f %%\n",  AP_results$AP50_95[AP_results$iou_type == "IoU (bounding box)"] * 100))
cat("================================================================\n\n")

# ── Combined PNG (y-axis 0–100%: SAM3 performs poorly so full axis is needed) ─
PR_combined <- bind_rows(
  PR_curve_mask %>% select(recall_curve, precision_curve, iou_type),
  PR_curve_bb   %>% select(recall_curve, precision_curve, iou_type)
) %>%
  left_join(
    AP_results %>%
      mutate(label = format_ap_label(iou_type, AP50, AP50_95) %>% factor(., .)) %>%
      select(iou_type, label)
  )

PR_plt <- PR_combined %>%
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
    legend.text            = element_text(family = "Courier New", size = 11),
    legend.position        = "inside",
    legend.position.inside = c(0.50, 0.98),
    legend.justification   = c(0.5, 1),
    legend.background      = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.key             = element_rect(fill = "transparent"),
    legend.spacing.x       = unit(0.2, "lines"),
    legend.margin          = margin(4, 6, 4, 6),
    plot.margin            = margin(0.5, 1, 0, 0.25, "lines"),
    axis.text              = element_text(size = 13),
    axis.title             = element_text(size = 14)
  )

dir.create("figures/graphs_thesis_experiments_using_fb_eval_refactored", showWarnings = FALSE, recursive = TRUE)
ggsave(
  "figures/graphs_thesis_experiments_using_fb_eval_refactored/pr_curve_ap_sam3_score_005_mask_05_mask_vs_bbox-old_greedy.png",
  PR_plt,
  device = "png",
  width = 4, height = 4,
  scale = 2.75,
  dpi = 300
)
cat("Saved: figures/graphs_thesis_experiments_using_fb_eval_refactored/pr_curve_ap_sam3_score_005_mask_05_mask_vs_bbox-old_greedy.png\n")
