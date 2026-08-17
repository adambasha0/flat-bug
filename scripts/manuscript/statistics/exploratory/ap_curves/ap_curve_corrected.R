source("helpers/flatbug_init.R")

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

bb_data <- data.table::fread("data/combined_results_sam3_corrected.csv") %>%
  as_tibble() %>%
  mutate(model = "SAM3") %>%
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
PR_curve <- bb_data %>% 
  group_modify(~ classify_at_threshold(.x, 0.5)) %>%
  arrange(desc(conf)) %>% 
  mutate(
    n_tp = sum(result == "TP"),
    n_fn = sum(result == "FN"),
    tp_cumsum = cumsum(result == "TP"),
    fp_cumsum = cumsum(result == "FP"),
    recall_curve = tp_cumsum / (n_tp + n_fn),
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
  ) %>% 
  ungroup 

AP_curve <- tibble(
  IoU_thresh = seq(0.5, 0.95, 0.01)
) %>% 
  mutate(
    result = map(IoU_thresh, function(t) {
      bb_data %>% 
        group_modify(~ {
          dt_thresh <- classify_at_threshold(.x, t)
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
    }, .progress = if (show_progress()) "Computing AP Curve" else F)
  ) %>% 
  unnest(result) %>%
  rename(IoU = IoU_thresh)

AP_results <- AP_curve %>% 
  summarize(
    model = bb_data$model[1],
    AP50 = AP[IoU == 0.5],
    AP50_95 = mean(AP[between(IoU, 0.5, 0.95)])
  )

PR_plt <- PR_curve %>% 
  left_join(
    AP_results %>% 
      mutate(
        label = str_c(model, "\n", "AP50: ", format(round(AP50, 3), nsmall=3), "  AP50-95: ", format(round(AP50_95, 3), nsmall=3)) %>% 
          factor(., .)
      ) %>% 
      select(model, label) 
  ) %>% 
  ggplot(aes(recall_curve, precision_curve, color = label)) +
  geom_line(
    key_glyph = draw_key_point,
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
    limits = c(0.5, 1),
    n.breaks = 11
  ) +
  scale_color_flatbug() +
  labs(
    x = "Recall",
    y = "Precision",
    color = "Model"
  ) +
  guides(
    color = guide_legend(
      override.aes = list(
        shape = 16,
        size = 6
      )
    )
  ) +
  theme(
    legend.text = element_text(family = "Courier New"),
    legend.position = "inside",
    legend.position.inside = c(0.95, 0.95),
    legend.justification = c(1, 1),
    plot.margin = margin(0.5, 1, 0, 0.25, "lines")
  )

ggsave(
  "figures/pr_curve_ap.pdf", 
  PR_plt,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 2.75,
  antialias = "subpixel"
)


