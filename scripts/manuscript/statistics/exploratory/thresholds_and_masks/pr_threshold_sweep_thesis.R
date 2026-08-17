## PR Threshold Sweep — thesis-consistent R version
##
## Replaces pr_threshold_sweep.py with methodology identical to
## ap_curve_sam3_005_thesis.R:
##   • Ranked-list PR curve (sort by confidence, cumulate TP/FP)
##   • Trapezoidal AP integration (gives AP@50 ≈ 90 % for SAM3-FT ckpt-18,
##     consistent with the thesis, NOT the ~94 % from the Python script which
##     uses COCO 101-point interpolation on a threshold-sweep curve)
##   • AP@50:95 averaged over 10 IoU thresholds: 0.50, 0.55, …, 0.95
##     (COCO-standard step size; thesis used 0.01 steps but difference is < 0.5 %)
##
## Part A  — LoRA v3 ckpt-6 vs SAM3-FT ckpt-18   (both score≥0.005)
## Part B  — winner (best F1) vs Flatbug
## Part C  — sanity check: ft_ckpt18 file score≥0.005 filtered to conf≥0.35
##            vs ft_ckpt18 file score≥0.35 (both curves now always visible)
## Overview— 4-panel combined view
##
## Outputs → figures/graphs_thesis_experiments_using_fb_eval_refactored/
##   pr_threshold_sweep_part_A.png
##   pr_threshold_sweep_part_B.png
##   pr_threshold_sweep_part_C.png
##   pr_threshold_sweep_overview.png

args         <- commandArgs(trailingOnly = FALSE)
file_arg     <- "--file="
script_path  <- args[grepl(file_arg, args)]
script_dir   <- if (length(script_path) > 0) {
                  dirname(normalizePath(sub(file_arg, "", script_path[1])))
                } else {
                  getwd()
                }
oldwd <- getwd()
setwd(script_dir)
on.exit(setwd(oldwd), add = TRUE)

source(file.path("helpers", "flatbug_init.R"))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR <- "data/thesis_experiments_using_fb_eval_refactored"
FIG_DIR  <- "figures/graphs_thesis_experiments_using_fb_eval_refactored"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

FILES <- list(
  lora_v3_ckpt6  = file.path(DATA_DIR, "fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_combined_results_with_bb.csv"),
  ft_ckpt18_s005 = file.path(DATA_DIR, "sam3_ft_round4_ep18_predictions_mask_05_score_005_combined_results_with_bb.csv"),
  flatbug        = file.path(DATA_DIR, "fb_score_005_using_fb_eval_greedy_corrected_with_bb.csv"),
  ft_ckpt18_s035 = file.path(DATA_DIR, "sam3_ft_round3_ep18_predictions_mask_5_score_35_combined_results_with_bb.csv")
)

LABELS <- c(
  lora_v3_ckpt6  = "LoRA v3 ckpt-6 (score\u22650.005)",
  ft_ckpt18_s005 = "SAM3-FT ckpt-18 (score\u22650.005)",
  flatbug        = "Flatbug (score\u22650.005)",
  ft_ckpt18_s035 = "SAM3-FT ckpt-18 (score\u22650.35)"
)

COLORS <- c(
  lora_v3_ckpt6  = "#E69F00",
  ft_ckpt18_s005 = "#0072B2",
  flatbug        = "#009E73",
  ft_ckpt18_s035 = "#CC79A7"
)

# ── IoU thresholds for AP50:95 (COCO 0.05 step — 10 values) ───────────────────
IOV_THRESHOLDS_50_95 <- seq(0.50, 0.95, 0.05)   # 0.50, 0.55, …, 0.95

# ── Data loading ───────────────────────────────────────────────────────────────
load_model_data <- function(path, model_label) {
  cat(sprintf("  Loading %-50s", basename(path)))
  df <- data.table::fread(path) %>%
    as_tibble() %>%
    mutate(
      area       = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
      size       = sqrt(area),
      match_type = case_when(
        idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",
        idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred",
        idx_1 != -1 & idx_2 != -1 ~ "matched"
      ),
      conf = replace_na(conf2, 0.),
      model = model_label
    ) %>%
    filter(size >= 32) %>%
    select(!any_of(c("conf1", "conf2")))
  cat(sprintf("  %d rows\n", nrow(df)))
  df
}

# ── Helper: filter to predictions with conf >= min_conf, keeping ALL GT rows ──
# Used for Part C to simulate "inference with higher score threshold".
filter_by_conf <- function(data, min_conf) {
  data %>%
    filter(
      match_type == "unmatched_gt" |
      (match_type %in% c("matched", "unmatched_pred") & conf >= min_conf)
    )
}

# ── Core AP helpers (identical to ap_curve_sam3_005_thesis.R) ─────────────────
integrate_curve <- function(x, y) {
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
  tp_matched   <- data %>% filter(match_type == "matched", .data[[iou_col]] >= iou_thresh) %>%
                    mutate(result = "TP")
  bad          <- data %>% filter(match_type == "matched", .data[[iou_col]] < iou_thresh)
  bind_rows(fn_unmatched, fp_unmatched, tp_matched,
            bad %>% mutate(result = "FP"),
            bad %>% mutate(result = "FN", conf = 0))
}

# Build ranked-list PR curve (sort by conf, cumulate TP/FP) at IoU@0.5
make_pr_curve <- function(data, iou_col, label) {
  classify_at_threshold(data, 0.5, iou_col) %>%
    arrange(desc(conf)) %>%
    mutate(
      n_tp  = sum(result == "TP"),
      n_fn  = sum(result == "FN"),
      tp_cs = cumsum(result == "TP"),
      fp_cs = cumsum(result == "FP"),
      recall    = tp_cs / (n_tp + n_fn),
      precision = tp_cs / (tp_cs + fp_cs),
      iou_label = label
    )
}

# AP at a single IoU threshold using the ranked-list curve
ap_at_iou <- function(data, iou_thresh, iou_col = "IoU") {
  dt <- classify_at_threshold(data, iou_thresh, iou_col)
  n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
  if (n_tp == 0) return(0.0)
  PR <- dt %>%
    arrange(desc(conf)) %>%
    mutate(
      tp_cs = cumsum(result == "TP"),
      fp_cs = cumsum(result == "FP"),
      recall    = tp_cs / (n_tp + n_fn),
      precision = tp_cs / (tp_cs + fp_cs)
    )
  integrate_curve(PR$recall, PR$precision)
}

compute_ap_metrics <- function(data, iou_col = "IoU") {
  aps <- vapply(IOV_THRESHOLDS_50_95, function(t) ap_at_iou(data, t, iou_col), numeric(1))
  list(ap50 = aps[1], ap50_95 = mean(aps))
}

format_legend_label <- function(display_name, ap50, ap5095, width = 24) {
  pad  <- stringr::str_pad(paste0(display_name, ":"), width = width, side = "right")
  str_c(pad,
        "AP50: ", sprintf("%.2f%%", ap50 * 100),
        "   AP50-95: ", sprintf("%.2f%%", ap5095 * 100))
}

# ── Threshold sweep (operating points, for P/R vs T panels) ───────────────────
pr_sweep <- function(data, iou_col = "IoU", iou_thresh = 0.5,
                     conf_thresholds = seq(0.005, 1.0, 0.005)) {
  pred_data <- data %>% filter(match_type %in% c("matched", "unmatched_pred"))
  n_gt      <- data %>% filter(match_type %in% c("matched", "unmatched_gt")) %>% nrow()
  map_dfr(conf_thresholds, function(t) {
    kept <- pred_data %>% filter(conf >= t)
    n_tp <- kept %>% filter(match_type == "matched", .data[[iou_col]] >= iou_thresh) %>% nrow()
    n_fp <- nrow(kept) - n_tp
    prec <- if ((n_tp + n_fp) > 0) n_tp / (n_tp + n_fp) else 0.0
    rec  <- if (n_gt > 0) n_tp / n_gt else 0.0
    f1   <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0.0
    tibble(threshold = t, tp = n_tp, fp = n_fp, fn = n_gt - n_tp,
           precision = prec, recall = rec, f1 = f1)
  })
}

# ── Standard theme (consistent with thesis) ───────────────────────────────────
thesis_theme <- function(base_size = 11) {
  theme_bw(base_size = base_size) %+replace% theme(
    panel.grid.minor = element_blank(),
    legend.background = element_rect(fill = alpha("white", 0.85), color = NA),
    legend.key        = element_rect(fill = "transparent"),
    legend.text       = element_text(family = "Courier New", size = 9),
    legend.margin     = margin(3, 5, 3, 5),
    axis.text         = element_text(size = base_size),
    axis.title        = element_text(size = base_size + 1),
    plot.title        = element_text(size = base_size + 1, face = "bold")
  )
}

# ── Build the three panels for one "Part" ─────────────────────────────────────
# Returns a patchwork of 3 ggplots:
#   [PR curve (ranked list)] | [Precision vs Threshold] | [Recall vs Threshold]
#
# model_list: named list of (label, color, data, sweep_data) tibbles
make_three_panel <- function(model_list, part_title,
                             pr_xlim = c(0, 1), pr_ylim = c(0, 1)) {
  # ── Panel 1: ranked-list PR curve with AP in legend ─────────────────────────
  pr_df <- map_dfr(model_list, ~ tibble(
    recall = .x$pr$recall, precision = .x$pr$precision, label = .x$legend_label
  ))
  # ensure factor order is stable
  pr_df$label <- factor(pr_df$label, levels = map_chr(model_list, ~ .x$legend_label))

  p_pr <- pr_df %>%
    ggplot(aes(recall, precision, color = label)) +
    geom_line(linewidth = 1) +
    scale_x_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       expand = expansion(), limits = pr_xlim, n.breaks = 6) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       expand = expansion(), limits = pr_ylim, n.breaks = 6) +
    scale_color_manual(
      values = set_names(unname(map_chr(model_list, ~ .x$color)),
                         map_chr(model_list, ~ .x$legend_label))
    ) +
    labs(x = "Recall", y = "Precision", color = NULL,
         title = "PR Curve (confidence-ranked)") +
    guides(color = guide_legend(keywidth = unit(1.2, "lines"))) +
    thesis_theme() +
    theme(legend.position        = "inside",
          legend.position.inside = c(0.5, 0.02),
          legend.justification   = c(0.5, 0))

  # ── Panel 2: Precision vs Confidence Threshold ──────────────────────────────
  sweep_df <- map_dfr(model_list, ~ .x$sweep %>% mutate(model = .x$display_label))
  sweep_df$model <- factor(sweep_df$model, levels = map_chr(model_list, ~ .x$display_label))

  color_map <- set_names(map_chr(model_list, ~ .x$color),
                         map_chr(model_list, ~ .x$display_label))
  p_pvt <- sweep_df %>%
    ggplot(aes(threshold, precision, color = model)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = color_map) +
    scale_x_continuous(expand = expansion(mult = 0.02)) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       limits = c(0, 1), expand = expansion()) +
    labs(x = "Confidence Threshold", y = "Precision", color = NULL,
         title = "Precision vs Threshold") +
    thesis_theme() +
    theme(legend.position = "inside",
          legend.position.inside = c(0.5, 0.02),
          legend.justification   = c(0.5, 0))

  # ── Panel 3: Recall vs Confidence Threshold ──────────────────────────────────
  p_rvt <- sweep_df %>%
    ggplot(aes(threshold, recall, color = model)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = color_map) +
    scale_x_continuous(expand = expansion(mult = 0.02)) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       limits = c(0, 1), expand = expansion()) +
    labs(x = "Confidence Threshold", y = "Recall", color = NULL,
         title = "Recall vs Threshold") +
    thesis_theme() +
    theme(legend.position = "inside",
          legend.position.inside = c(0.98, 0.98),
          legend.justification   = c(1, 1))

  (p_pr | p_pvt | p_rvt) +
    plot_annotation(title = part_title,
                    theme = theme(plot.title = element_text(size = 13,
                                                            face = "bold",
                                                            hjust = 0.5)))
}

# ── Load all data files, warn on missing ──────────────────────────────────────
cat("\nChecking and loading data files ...\n")
MODEL_DATA <- list()
for (key in names(FILES)) {
  path <- FILES[[key]]
  if (!file.exists(path)) {
    cat(sprintf("  [MISSING] %s\n", basename(path)))
    next
  }
  MODEL_DATA[[key]] <- load_model_data(path, LABELS[[key]])
}

# ── Helper: build model_list entry ────────────────────────────────────────────
make_entry <- function(key, data, ap_mask, sweep_data) {
  legend <- format_legend_label(LABELS[[key]], ap_mask$ap50, ap_mask$ap50_95)
  list(key           = key,
       display_label = LABELS[[key]],
       color         = COLORS[[key]],
       pr            = make_pr_curve(data, "IoU", LABELS[[key]]),
       sweep         = sweep_data,
       legend_label  = legend,
       ap50          = ap_mask$ap50,
       ap50_95       = ap_mask$ap50_95)
}

save_plot <- function(plt, filename, width = 18, height = 6, dpi = 150) {
  out <- file.path(FIG_DIR, filename)
  ggsave(out, plt, device = "png", width = width, height = height,
         units = "in", dpi = dpi, bg = "white")
  cat(sprintf("  Saved: %s\n", out))
}

# ══════════════════════════════════════════════════════════════════════════════
# PART A — LoRA v3 ckpt-6 vs SAM3-FT ckpt-18
# ══════════════════════════════════════════════════════════════════════════════
cat("\n====== PART A — LoRA v3 ckpt-6 vs SAM3-FT ckpt-18 ======\n")

THRESHOLDS <- seq(0.005, 1.0, 0.005)
part_a_keys <- c("lora_v3_ckpt6", "ft_ckpt18_s005")
part_a_entries <- list()

for (key in part_a_keys) {
  if (is.null(MODEL_DATA[[key]])) { cat(sprintf("  [SKIP] %s\n", key)); next }
  cat(sprintf("\n  Computing AP for: %s\n", LABELS[[key]]))
  ap <- compute_ap_metrics(MODEL_DATA[[key]], "IoU")
  cat(sprintf("    AP@50 = %.2f%%    AP@50:95 = %.2f%%\n",
              ap$ap50 * 100, ap$ap50_95 * 100))
  sweep <- pr_sweep(MODEL_DATA[[key]], iou_col = "IoU",
                    conf_thresholds = THRESHOLDS)
  best  <- sweep[which.max(sweep$f1), ]
  cat(sprintf("    Best F1 = %.4f  @ conf = %.3f  (P = %.4f, R = %.4f)\n",
              best$f1, best$threshold, best$precision, best$recall))
  part_a_entries[[key]] <- make_entry(key, MODEL_DATA[[key]], ap, sweep)
}

if (length(part_a_entries) == 2) {
  plt_a <- make_three_panel(
    part_a_entries,
    "Part A — Fine-tuned Model Comparison: LoRA v3 ckpt-6 vs SAM3-FT ckpt-18"
  )
  save_plot(plt_a, "pr_threshold_sweep_part_A.png")
} else {
  cat("  Not enough models for Part A — skipping plot.\n")
}

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Winner (best F1) vs Flatbug
# ══════════════════════════════════════════════════════════════════════════════
cat("\n====== PART B — Winner vs Flatbug ======\n")

winner_key <- if (length(part_a_entries) == 2) {
  best_f1 <- map_dbl(part_a_entries, ~ max(.x$sweep$f1))
  names(which.max(best_f1))
} else if (length(part_a_entries) == 1) names(part_a_entries)[1] else NULL

if (!is.null(winner_key) && !is.null(MODEL_DATA[["flatbug"]])) {
  cat(sprintf("  Winner: %s\n", LABELS[[winner_key]]))
  cat(sprintf("\n  Computing AP for: %s\n", LABELS[["flatbug"]]))
  ap_fb <- compute_ap_metrics(MODEL_DATA[["flatbug"]], "IoU")
  cat(sprintf("    AP@50 = %.2f%%    AP@50:95 = %.2f%%\n",
              ap_fb$ap50 * 100, ap_fb$ap50_95 * 100))
  sweep_fb <- pr_sweep(MODEL_DATA[["flatbug"]], iou_col = "IoU",
                       conf_thresholds = THRESHOLDS)
  best_fb  <- sweep_fb[which.max(sweep_fb$f1), ]
  cat(sprintf("    Best F1 = %.4f  @ conf = %.3f  (P = %.4f, R = %.4f)\n",
              best_fb$f1, best_fb$threshold, best_fb$precision, best_fb$recall))

  fb_entry <- make_entry("flatbug", MODEL_DATA[["flatbug"]], ap_fb, sweep_fb)

  plt_b <- make_three_panel(
    list(part_a_entries[[winner_key]], fb_entry),
    sprintf("Part B — %s vs Flatbug", LABELS[[winner_key]])
  )
  save_plot(plt_b, "pr_threshold_sweep_part_B_R.png")
} else {
  cat("  Missing winner or Flatbug data — skipping Part B.\n")
}

# ══════════════════════════════════════════════════════════════════════════════
# PART C — Sanity check: ft_ckpt18_s005 filtered ≥0.35 vs ft_ckpt18_s035
#
# FIX for Python bug: both ranked-list PR curves and threshold-sweep panels
# use a fixed 0–100% axis so neither curve is clipped by tight_lim.
# FIX for missing curve: ft_ckpt18_s005 is filtered at the data level
# (keeping all GT/FN rows) so the ranked-list PR curve is computed correctly
# for the "high-confidence only" subset.
# ══════════════════════════════════════════════════════════════════════════════
cat("\n====== PART C — Sanity Check: score_005 filtered >=0.35 vs score_035 ======\n")

THRESHOLDS_C <- seq(0.35, 1.0, 0.005)

if (!is.null(MODEL_DATA[["ft_ckpt18_s005"]]) &&
    !is.null(MODEL_DATA[["ft_ckpt18_s035"]])) {

  # ft_ckpt18_s005: keep all GT rows but remove predictions with conf < 0.35
  data_c1 <- filter_by_conf(MODEL_DATA[["ft_ckpt18_s005"]], 0.35)
  data_c2 <- MODEL_DATA[["ft_ckpt18_s035"]]

  cat(sprintf("  ft_ckpt18_s005 filtered: %d rows  (from %d)\n",
              nrow(data_c1), nrow(MODEL_DATA[["ft_ckpt18_s005"]])))
  cat(sprintf("  ft_ckpt18_s035 full:     %d rows\n", nrow(data_c2)))

  # AP for both (on the conf-filtered subsets)
  for (nm in c("c1", "c2")) {
    dat   <- if (nm == "c1") data_c1 else data_c2
    label <- if (nm == "c1") "ft_ckpt18_s005 (filtered \u22650.35)" else "ft_ckpt18_s035"
    ap    <- compute_ap_metrics(dat, "IoU")
    cat(sprintf("  %s: AP@50 = %.2f%%  AP@50:95 = %.2f%%\n",
                label, ap$ap50 * 100, ap$ap50_95 * 100))
  }

  ap_c1 <- compute_ap_metrics(data_c1, "IoU")
  ap_c2 <- compute_ap_metrics(data_c2, "IoU")

  sweep_c1 <- pr_sweep(data_c1, iou_col = "IoU", conf_thresholds = THRESHOLDS_C)
  sweep_c2 <- pr_sweep(data_c2, iou_col = "IoU", conf_thresholds = THRESHOLDS_C)

  # Check divergence
  merged_c <- inner_join(
    sweep_c1 %>% select(threshold, precision, recall) %>%
      rename(P_s005 = precision, R_s005 = recall),
    sweep_c2 %>% select(threshold, precision, recall) %>%
      rename(P_s035 = precision, R_s035 = recall),
    by = "threshold"
  ) %>%
    mutate(dP = abs(P_s005 - P_s035), dR = abs(R_s005 - R_s035))
  cat(sprintf("  Max |ΔPrecision|: %.6f\n", max(merged_c$dP)))
  cat(sprintf("  Max |ΔRecall|:    %.6f\n", max(merged_c$dR)))
  if (max(merged_c$dP) < 1e-4 && max(merged_c$dR) < 1e-4) {
    cat("  MATCH: both files produce identical results for conf >= 0.35.\n")
  } else {
    cat("  DIVERGENCE detected — greedy matching is order-sensitive.\n")
  }

  # Build legend labels using COLORS for existing keys + custom for filtered
  COLORS_C <- c(COLORS[["ft_ckpt18_s005"]], COLORS[["ft_ckpt18_s035"]])
  LABELS_C <- c(
    sprintf("SAM3-FT ckpt-18 (score\u22650.005, filtered\u22650.35)"),
    sprintf("SAM3-FT ckpt-18 (score\u22650.35)")
  )

  leg_c1 <- format_legend_label(LABELS_C[1], ap_c1$ap50, ap_c1$ap50_95)
  leg_c2 <- format_legend_label(LABELS_C[2], ap_c2$ap50, ap_c2$ap50_95)

  model_c <- list(
    list(display_label = LABELS_C[1], color = COLORS_C[1],
         pr            = make_pr_curve(data_c1, "IoU", LABELS_C[1]),
         sweep         = sweep_c1,
         legend_label  = leg_c1),
    list(display_label = LABELS_C[2], color = COLORS_C[2],
         pr            = make_pr_curve(data_c2, "IoU", LABELS_C[2]),
         sweep         = sweep_c2,
         legend_label  = leg_c2)
  )

  # FIX: use full 0–1 axes so both curves are always visible
  plt_c <- make_three_panel(
    model_c,
    "Part C — Sanity Check: score\u22650.005 (filtered \u22650.35) vs score\u22650.35",
    pr_xlim = c(0, 1), pr_ylim = c(0, 1)
  )
  save_plot(plt_c, "pr_threshold_sweep_part_C_R.png")
} else {
  cat("  Missing ft_ckpt18_s005 or ft_ckpt18_s035 data — skipping Part C.\n")
}

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW — 4-panel: PR + P vs T + R vs T + F1 vs T (all models)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n====== OVERVIEW — all models ======\n")

overview_keys <- intersect(c("lora_v3_ckpt6", "ft_ckpt18_s005", "flatbug"),
                            names(MODEL_DATA))

if (length(overview_keys) >= 2) {
  ov_entries <- list()
  for (key in overview_keys) {
    if (key %in% names(part_a_entries)) {
      ov_entries[[key]] <- part_a_entries[[key]]
    } else if (key == "flatbug" && exists("fb_entry")) {
      ov_entries[[key]] <- fb_entry
    } else {
      cat(sprintf("  Computing AP for overview: %s\n", LABELS[[key]]))
      ap_ov <- compute_ap_metrics(MODEL_DATA[[key]], "IoU")
      sw_ov <- pr_sweep(MODEL_DATA[[key]], iou_col = "IoU", conf_thresholds = THRESHOLDS)
      ov_entries[[key]] <- make_entry(key, MODEL_DATA[[key]], ap_ov, sw_ov)
    }
  }

  pr_all <- map_dfr(ov_entries, ~ tibble(
    recall = .x$pr$recall, precision = .x$pr$precision, label = .x$legend_label
  ))
  pr_all$label <- factor(pr_all$label, levels = map_chr(ov_entries, ~ .x$legend_label))

  sw_all <- map_dfr(ov_entries, ~ .x$sweep %>% mutate(model = .x$display_label))
  sw_all$model <- factor(sw_all$model, levels = map_chr(ov_entries, ~ .x$display_label))
  col_all <- set_names(map_chr(ov_entries, ~ .x$color),
                        map_chr(ov_entries, ~ .x$display_label))
  col_leg <- set_names(map_chr(ov_entries, ~ .x$color),
                        map_chr(ov_entries, ~ .x$legend_label))

  p_ov_pr <- pr_all %>%
    ggplot(aes(recall, precision, color = label)) +
    geom_line(linewidth = 1) +
    scale_x_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       expand = expansion(), limits = c(0, 1)) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       expand = expansion(), limits = c(0, 1)) +
    scale_color_manual(values = col_leg) +
    labs(x = "Recall", y = "Precision", color = NULL,
         title = "PR Curve (confidence-ranked)") +
    thesis_theme() +
    theme(legend.position = "inside",
          legend.position.inside = c(0.5, 0.02),
          legend.justification   = c(0.5, 0))

  p_ov_pvt <- sw_all %>%
    ggplot(aes(threshold, precision, color = model)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = col_all) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       limits = c(0, 1), expand = expansion()) +
    labs(x = "Confidence Threshold", y = "Precision", color = NULL,
         title = "Precision vs Threshold") +
    thesis_theme() +
    theme(legend.position = "inside",
          legend.position.inside = c(0.5, 0.02),
          legend.justification   = c(0.5, 0))

  p_ov_rvt <- sw_all %>%
    ggplot(aes(threshold, recall, color = model)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = col_all) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       limits = c(0, 1), expand = expansion()) +
    labs(x = "Confidence Threshold", y = "Recall", color = NULL,
         title = "Recall vs Threshold") +
    thesis_theme() +
    theme(legend.position = "inside",
          legend.position.inside = c(0.98, 0.98),
          legend.justification   = c(1, 1))

  p_ov_f1 <- sw_all %>%
    ggplot(aes(threshold, f1, color = model)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = col_all) +
    scale_y_continuous(labels = scales::label_percent(1, drop0trailing = TRUE),
                       limits = c(0, 1), expand = expansion()) +
    labs(x = "Confidence Threshold", y = "F1 Score", color = NULL,
         title = "F1 Score vs Threshold") +
    thesis_theme() +
    theme(legend.position = "inside",
          legend.position.inside = c(0.98, 0.98),
          legend.justification   = c(1, 1))

  plt_ov <- (p_ov_pr | p_ov_pvt) / (p_ov_rvt | p_ov_f1) +
    plot_annotation(
      title = "Overview — Precision\u2013Recall Threshold Sweep: All Models",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
    )
  save_plot(plt_ov, "pr_threshold_sweep_overview_R.png", width = 14, height = 10)
} else {
  cat("  Not enough models for Overview — skipping.\n")
}

cat("\nDone. All figures saved to:", FIG_DIR, "\n")
