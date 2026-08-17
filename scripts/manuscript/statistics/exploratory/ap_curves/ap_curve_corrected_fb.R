## Resolve script directory so the script can be run from the repo root
args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path_arg <- args[grepl(file_arg, args)]
script_dir <- if (length(script_path_arg) > 0) {
  dirname(normalizePath(sub(file_arg, "", script_path_arg[1])))
} else {
  # fallback when running interactively or from RStudio: use working dir
  getwd()
}

# Temporarily set working directory so relative `source()` calls inside
# `flatbug_init.R` (which expect `helpers/` relative to the script dir)
# resolve correctly when running Rscript from the repo root.
oldwd <- getwd()
setwd(script_dir)
on.exit(setwd(oldwd), add = TRUE)

source(file.path("helpers", "flatbug_init.R"))

# Function to parse bbox string "[x1, y1, x2, y2]" to numeric vector
parse_bbox <- function(bbox_str) {
  if (is.na(bbox_str) || bbox_str == "") return(c(NA, NA, NA, NA))
  tryCatch({
    nums <- as.numeric(strsplit(gsub("\\[|\\]", "", bbox_str), ",\\s*")[[1]])
    if (length(nums) == 4) nums else c(NA, NA, NA, NA)
  }, error = function(e) c(NA, NA, NA, NA))
}

# Function to compute IoU between two bounding boxes
compute_bbox_iou <- function(bbox1, bbox2) {
  if (any(is.na(bbox1)) || any(is.na(bbox2))) return(NA_real_)
  
  x1 <- max(bbox1[1], bbox2[1])
  y1 <- max(bbox1[2], bbox2[2])
  x2 <- min(bbox1[3], bbox2[3])
  y2 <- min(bbox1[4], bbox2[4])
  
  if (x2 <= x1 || y2 <= y1) return(0.0)
  
  intersection <- (x2 - x1) * (y2 - y1)
  area1 <- (bbox1[3] - bbox1[1]) * (bbox1[4] - bbox1[2])
  area2 <- (bbox2[3] - bbox2[1]) * (bbox2[4] - bbox2[2])
  union <- area1 + area2 - intersection
  
  if (union <= 0) return(0.0)
  intersection / union
}


integrate_curve <- function(x, y) {
  # Ensure x and y have the same length
  if (length(x) != length(y)) {
    stop("x and y must have the same length")
  }
  
  # Create a data frame of the points
  df <- data.frame(x = x, y = y)
  
  # Sort by x in increasing order
  df <- df[order(df$x), ]
  
  # Check for duplicate x values.
  # For precision-recall curves, it's common to take the maximum y for each x.
  if(any(duplicated(df$x))) {
    warning("Duplicate x values detected. Aggregating by taking the maximum y for each unique x.")
    df <- aggregate(y ~ x, data = df, FUN = max)
    df <- df[order(df$x), ]
  }
  
  # Apply the trapezoidal rule:
  # area = sum( (x[i+1]-x[i]) * (y[i] + y[i+1]) / 2 )
  dx <- diff(df$x)
  avg_y <- (head(df$y, -1) + tail(df$y, -1)) / 2
  area <- sum(dx * avg_y)
  
  return(area)
}

# Function to classify TP/FP/FN at a given IoU threshold
# A match with IoU < threshold should count as BOTH FP (bad pred) AND FN (missed GT)
# iou_col: which IoU column to use ("IoU" for mask, "IoU_bb" for bbox)
classify_at_threshold <- function(data, iou_thresh, iou_col = "IoU") {
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
    filter(match_type == "matched" & .data[[iou_col]] >= iou_thresh) %>%
    mutate(result = "TP")
  
  # Matched pairs with IoU < threshold -> split into FP (pred) and FN (GT)
  bad_matches <- data %>%
    filter(match_type == "matched" & .data[[iou_col]] < iou_thresh)
  
  # Create FP rows for bad match predictions
  fp_bad_match <- bad_matches %>%
    mutate(result = "FP")
  
  # Create FN rows for bad match ground truths  
  fn_bad_match <- bad_matches %>%
    mutate(result = "FN", conf = 0)  # FN has no confidence
  
  # Combine all
  bind_rows(fn_unmatched, fp_unmatched, tp_matched, fp_bad_match, fn_bad_match)
}

# Function to compute AP at a given IoU threshold
compute_ap_at_threshold <- function(data, iou_thresh, iou_col = "IoU") {
  # Classify TP/FP/FN at this threshold
  dt_thresh <- classify_at_threshold(data, iou_thresh, iou_col)
  
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

# Format legend labels with aligned names and percentage AP values
format_ap_label <- function(name, ap50, ap5095, width = 22) {
  name_pad <- stringr::str_pad(paste0(name, ":"), width = width, side = "right")
  ap50_str <- paste0(format(round(ap50 * 100, 2), nsmall = 2), "%")
  ap95_str <- paste0(format(round(ap5095 * 100, 2), nsmall = 2), "%")
  str_c(name_pad, "AP50: ", ap50_str, "   |   AP50-95: ", ap95_str)
}

bb_data <- data.table::fread("data/combined_results_fb_score_001_with_bb.csv") %>%
  as_tibble() %>%
  mutate(model = "Flatbug") %>%
  rowwise() %>%
  mutate(
    # Compute IoU_bb from bounding boxes
    IoU_bb = {
      if (idx_1 != -1 & idx_2 != -1) {
        bbox1 <- parse_bbox(bbox_1)
        bbox2 <- parse_bbox(bbox_2)
        compute_bbox_iou(bbox1, bbox2)
      } else {
        NA_real_
      }
    }
  ) %>%
  ungroup() %>%
  mutate(
    area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size = sqrt(area),
    dataset = str_extract(image, "^[^_]+"),
    short = short_name(dataset),
    # Match type based on indices (not IoU threshold yet)
    match_type = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",   # GT with no prediction
      idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred", # Prediction with no GT  
      idx_1 != -1 & idx_2 != -1 ~ "matched"         # Matched pair
    ),
    conf = replace_na(conf2, 0.)
  ) %>% 
  filter(size >= 32) %>% 
  select(!c(conf1, conf2)) 

# For plotting PR curve at IoU=0.5, use corrected classification
# Compute for both IoU (mask) and IoU_bb (bbox)
PR_curve_mask <- bb_data %>% 
  group_modify(~ classify_at_threshold(.x, 0.5, "IoU")) %>%
  arrange(desc(conf)) %>% 
  mutate(
    n_tp = sum(result == "TP"),
    n_fn = sum(result == "FN"),
    tp_cumsum = cumsum(result == "TP"),
    fp_cumsum = cumsum(result == "FP"),
    recall_curve = tp_cumsum / (n_tp + n_fn),
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum),
    iou_type = "IoU (mask)"
  ) %>% 
  ungroup()

PR_curve_bb <- bb_data %>% 
  group_modify(~ classify_at_threshold(.x, 0.5, "IoU_bb")) %>%
  arrange(desc(conf)) %>% 
  mutate(
    n_tp = sum(result == "TP"),
    n_fn = sum(result == "FN"),
    tp_cumsum = cumsum(result == "TP"),
    fp_cumsum = cumsum(result == "FP"),
    recall_curve = tp_cumsum / (n_tp + n_fn),
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum),
    iou_type = "IoU (bounding box)"
  ) %>% 
  ungroup()

# Helper function to compute AP curve for a given IoU column
compute_ap_curve <- function(data, iou_col, iou_type_label) {
  tibble(
    IoU_thresh = seq(0.5, 0.95, 0.01)
  ) %>% 
    mutate(
      result = map(IoU_thresh, function(t) {
        data %>% 
          group_modify(~ {
            dt_thresh <- classify_at_threshold(.x, t, iou_col)
            n_tp <- sum(dt_thresh$result == "TP")
            n_fn <- sum(dt_thresh$result == "FN")
            
            if (n_tp == 0) {
              return(tibble(AP = 0, P = 0, R = 0))
            }
            
            PR <- dt_thresh %>%
              arrange(desc(conf)) %>%
              mutate(
                tp_cumsum = cumsum(result == "TP"),
                fp_cumsum = cumsum(result == "FP"),
                recall_curve = tp_cumsum / (n_tp + n_fn),
                precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
              )
            
            tibble(
              AP = integrate_curve(PR$recall_curve, PR$precision_curve),
              w = which.min(abs(PR$precision_curve - 0.975)),
              P = PR$precision_curve[w],
              R = PR$recall_curve[w]
            ) %>% select(!w)
          }) %>%
          ungroup()
      }, .progress = if (show_progress()) paste("Computing AP Curve", iou_type_label) else FALSE)
    ) %>% 
    unnest(result) %>%
    rename(IoU = IoU_thresh) %>%
    mutate(iou_type = iou_type_label)
}

# Compute AP curves for both IoU types
AP_curve_mask <- compute_ap_curve(bb_data, "IoU", "IoU (mask)")
AP_curve_bb <- compute_ap_curve(bb_data, "IoU_bb", "IoU (bounding box)")
AP_curve <- bind_rows(AP_curve_mask, AP_curve_bb)

AP_results_mask <- AP_curve_mask %>% 
  summarize(
    model = bb_data$model[1],
    iou_type = "IoU (mask)",
    AP50 = AP[IoU == 0.5],
    AP50_95 = mean(AP[between(IoU, 0.5, 0.95)])
  )

AP_results_bb <- AP_curve_bb %>% 
  summarize(
    model = bb_data$model[1],
    iou_type = "IoU (bounding box)",
    AP50 = AP[IoU == 0.5],
    AP50_95 = mean(AP[between(IoU, 0.5, 0.95)])
  )

AP_results <- bind_rows(AP_results_mask, AP_results_bb)

# Print comparison
cat("\n========== AP COMPARISON ==========\n")
cat("Model:", bb_data$model[1], "\n\n")
cat("IoU (mask):\n")
cat("  AP50:     ", format(round(AP_results_mask$AP50, 4), nsmall=4), "\n")
cat("  AP50-95:  ", format(round(AP_results_mask$AP50_95, 4), nsmall=4), "\n\n")
cat("IoU_bb (bbox):\n")
cat("  AP50:     ", format(round(AP_results_bb$AP50, 4), nsmall=4), "\n")
cat("  AP50-95:  ", format(round(AP_results_bb$AP50_95, 4), nsmall=4), "\n")
cat("====================================\n\n")

# Plot for IoU (mask) - original plot
PR_plt <- PR_curve_mask %>% 
  left_join(
    AP_results_mask %>% 
      mutate(
        label = format_ap_label("IoU (mask)", AP50, AP50_95) %>%
          factor(., .)
      ) %>% 
      select(model, label) 
  ) %>% 
  ggplot(aes(recall_curve, precision_curve, color = label)) +
  geom_line(
    linewidth = 1
  ) +
  scale_x_continuous(
    labels = scales::label_percent(.1, drop0trailing=T), 
    expand = expansion(),
    limits = 0:1,
    n.breaks = 11
  ) +
  scale_y_continuous(
    labels = scales::label_percent(.1, drop0trailing=T), 
    expand = expansion(),
    limits = c(0.60, 1),
    n.breaks = 11
  ) +
  scale_color_flatbug() +
  labs(
    x = "Recall",
    y = "Precision",
    color = NULL
  ) +
  guides(
    color = guide_legend(
      keywidth = unit(1.5, "lines"),
      keyheight = unit(1, "lines")
    )
  ) +
  theme(
    legend.text = element_text(family = "Courier New", size = 9),
    legend.position = "inside",
    legend.position.inside = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.key = element_rect(fill = "transparent"),
    legend.spacing.x = unit(0.2, "lines"),
    legend.margin = margin(4, 6, 4, 6),
    plot.margin = margin(0.5, 1, 0, 0.25, "lines")
  )

ggsave(
  "figures/pr_curve_ap_fb_score_001_mask.pdf", 
  PR_plt,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 2.75,
  antialias = "subpixel"
)

# Plot for IoU_bb (bbox)
PR_plt_bb <- PR_curve_bb %>% 
  left_join(
    AP_results_bb %>% 
      mutate(
        label = format_ap_label("IoU (bounding box)", AP50, AP50_95) %>%
          factor(., .)
      ) %>% 
      select(model, label) 
  ) %>% 
  ggplot(aes(recall_curve, precision_curve, color = label)) +
  geom_line(
    linewidth = 1
  ) +
  scale_x_continuous(
    labels = scales::label_percent(.1, drop0trailing=T), 
    expand = expansion(),
    limits = 0:1,
    n.breaks = 11
  ) +
  scale_y_continuous(
    labels = scales::label_percent(.1, drop0trailing=T), 
    expand = expansion(),
    limits = c(0.60, 1),
    n.breaks = 11
  ) +
  scale_color_flatbug() +
  labs(
    x = "Recall",
    y = "Precision",
    color = NULL
  ) +
  guides(
    color = guide_legend(
      keywidth = unit(1.5, "lines"),
      keyheight = unit(1, "lines")
    )
  ) +
  theme(
    legend.text = element_text(family = "Courier New", size = 9),
    legend.position = "inside",
    legend.position.inside = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.key = element_rect(fill = "transparent"),
    legend.spacing.x = unit(0.2, "lines"),
    legend.margin = margin(4, 6, 4, 6),
    plot.margin = margin(0.5, 1, 0, 0.25, "lines")
  )

ggsave(
  "figures/pr_curve_ap_fb_score_001_bb.pdf", 
  PR_plt_bb,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 2.75,
  antialias = "subpixel"
)

# Combined comparison plot
PR_curve_combined <- bind_rows(
  PR_curve_mask %>% select(recall_curve, precision_curve, iou_type),
  PR_curve_bb %>% select(recall_curve, precision_curve, iou_type)
)

PR_plt_combined <- PR_curve_combined %>%
  left_join(
    AP_results %>%
      mutate(
        label = format_ap_label(iou_type, AP50, AP50_95) %>%
          factor(., .)
      ) %>%
      select(iou_type, label)
  ) %>%
  ggplot(aes(recall_curve, precision_curve, color = label)) +
  geom_line(
    linewidth = 1
  ) +
  scale_x_continuous(
    labels = scales::label_percent(.1, drop0trailing=T),
    expand = expansion(),
    limits = 0:1,
    n.breaks = 11
  ) +
  scale_y_continuous(
    labels = scales::label_percent(.1, drop0trailing=T),
    expand = expansion(),
    limits = c(0.60, 1),
    n.breaks = 11
  ) +
  scale_color_manual(values = c("#E69F00", "#56B4E9")) +
  labs(
    x = "Recall",
    y = "Precision",
    color = NULL
  ) +
  guides(
    color = guide_legend(
      keywidth = unit(1.5, "lines"),
      keyheight = unit(1, "lines")
    )
  ) +
  theme(
    legend.text = element_text(family = "Courier New", size = 9),
    legend.position = "inside",
    legend.position.inside = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.key = element_rect(fill = "transparent"),
    legend.spacing.x = unit(0.2, "lines"),
    legend.margin = margin(4, 6, 4, 6),
    plot.margin = margin(0.5, 1, 0, 0.25, "lines")
  )

ggsave(
  "figures/pr_curve_ap_fb_mask_vs_bbox_score_001.pdf",
  PR_plt_combined,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 2.75,
  antialias = "subpixel"
)


