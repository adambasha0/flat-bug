## PR curves for thesis experiment models (score 0.005, mask 0.5)
## Produces one PNG per model in:
##   figures/graphs_thesis_experiments_using_fb_eval_refactored/
##
## Models:
##   - SAM3 base (score 0.005)
##   - SAM3 fine-tuned with LoRA checkpoint 10 (mask 0.5, score 0.005)
##   - SAM3 fine-tuned round 4 epoch 18 (mask 0.5, score 0.005)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path_arg <- args[grepl(file_arg, args)]
script_dir <- if (length(script_path_arg) > 0) {
  dirname(normalizePath(sub(file_arg, "", script_path_arg[1])))
} else {
  getwd()
}
oldwd <- getwd()
on.exit(setwd(oldwd), add = TRUE)

## Paths are resolved relative to this script, so it runs from any working
## directory and from either checkout. Override with environment variables:
##   FB_HELPERS_DIR  directory containing the `helpers/` R utilities
##   FB_DATA_DIR     directory holding the *_experiment_N with-bbox CSVs
##   FB_FIGURE_DIR   output directory for the PR-curve PDFs
env_or <- function(var, default) {
  v <- Sys.getenv(var, unset = "")
  if (nzchar(v)) normalizePath(v, mustWork = FALSE) else default
}
abs_from_script <- function(...) normalizePath(file.path(script_dir, ...), mustWork = FALSE)
## Two supported layouts. Standalone (default): this folder is self-contained —
## CSVs in `experiments_1_3/data/` (unpack `data/*.zip` first), output under
## `<thesis root>/_output/`. Embedded: this folder sits inside the larger
## evaluation working tree, which keeps its own `statistics/{data,figures}`
## directories two levels up; detected by the upstream `helpers/` beside them.
EMBEDDED <- dir.exists(abs_from_script("..", "..", "helpers"))

HELPERS_DIR <- env_or("FB_HELPERS_DIR", abs_from_script("..", "helpers"))
DATA_DIR <- env_or("FB_DATA_DIR",
                   if (EMBEDDED) abs_from_script("..", "..", "data",
                                                 "thesis_experiments_using_fb_eval_refactored",
                                                 "with_bb")
                   else abs_from_script("data"))
FIGURE_DIR <- env_or("FB_FIGURE_DIR",
                     if (EMBEDDED) abs_from_script("..", "..", "figures",
                                                   "graphs_thesis_experiments_using_fb_eval_refactored")
                     else abs_from_script("..", "_output", "pr_curves"))

## flatbug_init.R sources its siblings via the relative path `helpers/...`,
## so run from the directory that holds `helpers/` (paths above are absolute).
setwd(dirname(HELPERS_DIR))
source(file.path("helpers", "flatbug_init.R"))

# ── Helper functions ──────────────────────────────────────────────────────────

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
  fn_unmatched <- data %>% filter(match_type == "unmatched_gt")  %>% mutate(result = "FN")
  fp_unmatched <- data %>% filter(match_type == "unmatched_pred") %>% mutate(result = "FP")
  tp_matched   <- data %>% filter(match_type == "matched" & .data[[iou_col]] >= iou_thresh) %>%
    mutate(result = "TP")
  bad_matches  <- data %>% filter(match_type == "matched" & .data[[iou_col]] < iou_thresh)
  fp_bad <- bad_matches %>% mutate(result = "FP")
  fn_bad <- bad_matches %>% mutate(result = "FN", conf = 0)
  bind_rows(fn_unmatched, fp_unmatched, tp_matched, fp_bad, fn_bad)
}

format_ap_label <- function(name, ap50, ap5095, width = 13) {
  name     <- sub("bounding box", "bbox", name)
  name_pad <- stringr::str_pad(paste0(name, ":"), width = width, side = "right")
  ap50_str <- sprintf("%.2f%%", ap50  * 100)
  ap95_str <- sprintf("%.2f%%", ap5095 * 100)
  indent   <- strrep(" ", width)
  # Leading blank line makes the block 3 lines tall so the swatch — drawn at
  # the row centre — lands on the first content row (name + AP50) rather than
  # between the two rows, keeping it level with the "IoU …" text.
  str_c("\n", name_pad, "AP50:    ", ap50_str, "\n", indent, "AP50-95: ", ap95_str)
}

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
                 w  = which.min(abs(PR$precision_curve - 0.975)),
                 P  = PR$precision_curve[w],
                 R  = PR$recall_curve[w]) %>% select(!w)
        }) %>% ungroup()
    }, .progress = if (show_progress()) paste("AP Curve", iou_type_label) else FALSE)) %>%
    unnest(result) %>%
    rename(IoU = IoU_thresh) %>%
    mutate(iou_type = iou_type_label)
}

load_data <- function(csv_path, model_name) {
  data.table::fread(csv_path) %>%
    as_tibble() %>%
    mutate(
      model     = model_name,
      area      = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
      size      = sqrt(area),
      match_type = case_when(
        idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",
        idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred",
        idx_1 != -1 & idx_2 != -1 ~ "matched"
      ),
      conf = replace_na(conf2, 0.)
    ) %>%
    filter(size >= 32) %>%
    select(!c(conf1, conf2))
}

generate_pr_plot <- function(csv_path, model_name, out_name, y_min = 0) {
  cat(sprintf("\n──── %s ────\n", model_name))
  cat(sprintf("     CSV: %s\n", csv_path))

  data <- load_data(csv_path, model_name)

  PR_curve_mask <- make_pr_curve(data, "IoU",    "IoU (mask)")
  PR_curve_bb   <- make_pr_curve(data, "IoU_bb", "IoU (bounding box)")

  AP_curve_mask <- compute_ap_curve(data, "IoU",    "IoU (mask)")
  AP_curve_bb   <- compute_ap_curve(data, "IoU_bb", "IoU (bounding box)")

  AP_results <- bind_rows(
    AP_curve_mask %>% summarize(model = model_name, iou_type = "IoU (mask)",
                                AP50     = AP[IoU == 0.5],
                                AP50_95  = mean(AP[between(IoU, 0.5, 0.95)])),
    AP_curve_bb   %>% summarize(model = model_name, iou_type = "IoU (bounding box)",
                                AP50     = AP[IoU == 0.5],
                                AP50_95  = mean(AP[between(IoU, 0.5, 0.95)]))
  )

  cat(sprintf("     AP50 (mask):      %6.2f%%\n", AP_results$AP50[AP_results$iou_type == "IoU (mask)"]            * 100))
  cat(sprintf("     AP50-95 (mask):   %6.2f%%\n", AP_results$AP50_95[AP_results$iou_type == "IoU (mask)"]         * 100))
  cat(sprintf("     AP50 (bbox):      %6.2f%%\n", AP_results$AP50[AP_results$iou_type == "IoU (bounding box)"]    * 100))
  cat(sprintf("     AP50-95 (bbox):   %6.2f%%\n", AP_results$AP50_95[AP_results$iou_type == "IoU (bounding box)"] * 100))

  PR_combined <- bind_rows(
    PR_curve_mask %>% select(recall_curve, precision_curve, iou_type),
    PR_curve_bb   %>% select(recall_curve, precision_curve, iou_type)
  ) %>%
    left_join(
      AP_results %>%
        mutate(label = format_ap_label(iou_type, AP50, AP50_95) %>% factor(., .)) %>%
        select(iou_type, label),
      by = "iou_type"
    )

  # y-axis: zoom to [y_min, 1]. Full range (y_min = 0) uses 10 % ticks labelled
  # every 20 %; a zoomed range (e.g. Flatbug from 0.7) uses 5 % ticks labelled
  # every 10 % so the near-1 curve fills the panel.
  y_step     <- if (y_min >= 0.5) 0.05 else 0.1
  y_mult     <- if (y_min >= 0.5) 20   else 10
  y_breaks   <- seq(y_min, 1, y_step)
  y_labeller <- function(b) ifelse(round(b * y_mult) %% 2 == 0,
                                   scales::label_percent(accuracy = 1)(b), "")

  p <- PR_combined %>%
    ggplot(aes(recall_curve, precision_curve, color = label)) +
    geom_line(linewidth = 1) +
    scale_x_continuous(
      # Tick/gridline every 10 %, but label only every other one
      # (0 %, 20 %, 40 % …) so labels don't collide at this small print size.
      breaks = seq(0, 1, 0.1),
      labels = function(b) ifelse(round(b * 10) %% 2 == 0,
                                  scales::label_percent(accuracy = 1)(b), ""),
      expand = expansion(), limits = 0:1
    ) +
    scale_y_continuous(
      breaks = y_breaks, labels = y_labeller,
      expand = expansion(), limits = c(y_min, 1)
    ) +
    scale_color_manual(values = c("#E69F00", "#56B4E9")) +
    coord_cartesian(clip = "off") +
    labs(x = "Recall", y = "Precision", color = NULL, title = NULL) +
    guides(color = guide_legend(ncol = 1, keywidth = unit(1.2, "lines"), keyheight = unit(2, "lines"))) +
    theme(
      # Bold 4-sided border (replaces the single-sided axis lines from theme_classic)
      panel.border           = element_rect(colour = "black", fill = NA, linewidth = 0.5),
      axis.line              = element_blank(),
      # Full square grid at every major break
      panel.background       = element_rect(fill = "white"),
      panel.grid.major       = element_line(colour = "gray82", linewidth = 0.2),
      panel.grid.minor       = element_blank(),
      # Inside ticks: negative length puts marks inside the panel border
      axis.ticks.length      = unit(-3, "pt"),
      axis.text              = element_text(size = 8),
      axis.text.x            = element_text(size = 8, margin = margin(t = 5)),
      axis.text.y            = element_text(size = 8, margin = margin(r = 5)),
      axis.title.x           = element_text(size = 9, margin = margin(t = 6)),
      axis.title.y           = element_text(size = 9, margin = margin(r = 6)),
      # Legend – centred below the panel, solid rectangular border
      legend.text            = element_text(family = "Courier New", size = 7),
      legend.position        = "bottom",
      legend.direction       = "vertical",
      legend.justification   = "center",
      legend.background      = element_rect(fill = "transparent", colour = NA),
      legend.box.background  = element_rect(fill = "white", colour = "black", linewidth = 0.5),
      legend.box.margin      = margin(0, 0, 0, 0),
      legend.key             = element_rect(fill = "transparent"),
      legend.spacing.x       = unit(0.2, "lines"),
      legend.key.spacing.y   = unit(2, "pt"),
      legend.margin          = margin(4, 4, 4, 4),
      legend.box.spacing     = unit(4, "pt"),
      # No title, tighter margin
      plot.title             = element_blank(),
      plot.margin            = margin(0.4, 1.1, 0.2, 0.2, "lines")
    )

  out_dir <- FIGURE_DIR
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir, sub("\\.png$", ".pdf", out_name))
  ggsave(out_path, p, device = cairo_pdf, width = 3.0, height = 3.6)
  cat(sprintf("     Saved: %s\n", out_path))
}

# ── Dataset definitions ───────────────────────────────────────────────────────
## Unified naming: exp{N}_{model}_pr_curve_score_005.pdf (model: sam3=base,
## flatbug, ft=full fine-tune, lora). CSVs carry the user's _experiment_N suffix.
## y_min zooms the y-axis: Flatbug stays ~78–100 %, so start at 0.70; the other
## models' precision collapses toward 0, so they keep the full 0–1 range.
datasets <- tribble(
  ~csv_path,                                                                                                          ~model_name,                             ~out_name,                             ~y_min,
  file.path(DATA_DIR, "sam3_results_score_005_combined_results_with_bb_experiment_1.csv"),                            "SAM3 base (Exp 1, score 0.005)",        "exp1_sam3_pr_curve_score_005.png",    0,
  file.path(DATA_DIR, "fb_score_005_using_fb_eval_greedy_corrected_combined_results_with_bb_experiment_1.csv"),       "Flatbug (Exp 1, score 0.005)",          "exp1_flatbug_pr_curve_score_005.png", 0.7,
  file.path(DATA_DIR, "sam3_ft_round4_ep18_predictions_mask_05_score_005_combined_results_with_bb_experiment_2.csv"), "SAM3 fine-tuned (Exp 2, score 0.005)",  "exp2_ft_pr_curve_score_005.png",      0,
  file.path(DATA_DIR, "fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_combined_results_with_bb_experiment_3.csv"), "SAM3 LoRA (Exp 3, score 0.005)",  "exp3_lora_pr_curve_score_005.png",    0
)

# ── Generate all plots ────────────────────────────────────────────────────────
cat("\n========== PR CURVES — THESIS EXPERIMENTS (score 0.005, mask 0.5) ==========\n")
pwalk(datasets, generate_pr_plot)
cat("\n========== DONE ==========\n")
