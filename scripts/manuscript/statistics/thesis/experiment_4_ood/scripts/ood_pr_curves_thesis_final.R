## Thesis-format PR curves for the OOD generalization study (SAM3-ft vs FlatBug-L).
##
## Same enhanced visual format as ap_curves_thesis_experiments_refactored_final.R
## (vector cairo_pdf, 4-sided border, inside ticks, Courier legend with AP block),
## but each panel overlays the TWO MODELS (SAM3 ft. vs FlatBug L) for one dataset
## instead of the two IoU types for one model.
##
## One PDF per (dataset x iou_type):
##   - urban_insects  (bbox)   — GT is boxes only
##   - massid45       (bbox)   — real GT masks, but bbox view for cross-dataset parity
##   - massid45       (mask)   — real GT polygon masks -> mask PR is meaningful here
##   - pest24         (bbox)   — GT is boxes only
##
## NOTE vs the experiments script: NO size>=32 filter here. The OOD study is about
## small-object transfer, so every prediction/GT is kept (size_filter = 0).

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

## Paths resolve relative to this script, which lives in `<deliverable>/scripts/`.
## The per-detection CSVs ship in the deliverable as
## `<deliverable>/<dataset>/csv/<dataset>_<sam3|flatbug>.csv`.
## Override with environment variables if they live elsewhere:
##   FB_HELPERS_DIR  directory containing the `helpers/` R utilities
##   OOD_DELIVERABLE deliverable root holding the per-dataset folders
##   FB_FIGURE_DIR   output directory for the PR-curve PDFs
env_or <- function(var, default) {
  v <- Sys.getenv(var, unset = "")
  if (nzchar(v)) normalizePath(v, mustWork = FALSE) else default
}
abs_from_script <- function(...) normalizePath(file.path(script_dir, ...), mustWork = FALSE)

HELPERS_DIR <- env_or("FB_HELPERS_DIR", abs_from_script("..", "..", "helpers"))
DELIVERABLE <- env_or("OOD_DELIVERABLE", abs_from_script(".."))
FIGURE_DIR  <- env_or("FB_FIGURE_DIR", file.path(DELIVERABLE, "_pr_curves_thesis_pdf"))

## flatbug_init.R sources its siblings via the relative path `helpers/...`,
## so run from the directory that holds `helpers/` (paths above are absolute).
setwd(dirname(HELPERS_DIR))
source(file.path("helpers", "flatbug_init.R"))

# ── Helper functions (identical maths to the experiments script) ───────────────

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

# model name + AP block, padded so the two models' AP columns line up in the legend.
format_ap_label <- function(name, ap50, ap5095, width = 13) {
  name_pad <- stringr::str_pad(paste0(name, ":"), width = width, side = "right")
  ap50_str <- sprintf("%.2f%%", ap50   * 100)
  ap95_str <- sprintf("%.2f%%", ap5095 * 100)
  indent   <- strrep(" ", width)
  str_c("\n", name_pad, "AP50:    ", ap50_str, "\n", indent, "AP50-95: ", ap95_str)
}

make_pr_curve <- function(data, iou_col) {
  data %>%
    classify_at_threshold(0.5, iou_col) %>%
    arrange(desc(conf)) %>%
    mutate(
      n_tp = sum(result == "TP"),
      n_fn = sum(result == "FN"),
      tp_cumsum = cumsum(result == "TP"),
      fp_cumsum = cumsum(result == "FP"),
      recall_curve    = tp_cumsum / (n_tp + n_fn),
      precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
    )
}

compute_ap <- function(data, iou_col) {
  ts <- seq(0.5, 0.95, 0.01)
  aps <- sapply(ts, function(t) {
    dt <- classify_at_threshold(data, t, iou_col)
    n_tp <- sum(dt$result == "TP"); n_fn <- sum(dt$result == "FN")
    if (n_tp == 0) return(0)
    PR <- dt %>% arrange(desc(conf)) %>%
      mutate(tp_cumsum = cumsum(result == "TP"),
             fp_cumsum = cumsum(result == "FP"),
             recall_curve    = tp_cumsum / (n_tp + n_fn),
             precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum))
    integrate_curve(PR$recall_curve, PR$precision_curve)
  })
  list(AP50 = aps[1], AP50_95 = mean(aps))
}

load_data <- function(csv_path, size_filter = 0) {
  data.table::fread(csv_path) %>%
    as_tibble() %>%
    mutate(
      area  = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
      size  = sqrt(area),
      match_type = case_when(
        idx_1 != -1 & idx_2 == -1 ~ "unmatched_gt",
        idx_1 == -1 & idx_2 != -1 ~ "unmatched_pred",
        idx_1 != -1 & idx_2 != -1 ~ "matched"
      ),
      conf = replace_na(conf2, 0.)
    ) %>%
    filter(size >= size_filter)
}

# ── Colours: keep the OOD study's model convention (SAM3 blue, FlatBug orange) ──
MODELS <- c("SAM3 ft.", "FlatBug L")
COL    <- c("SAM3 ft." = "#0072B2", "FlatBug L" = "#D55E00")
## Deliverable layout: <root>/<dataset>/csv/<dataset>_<model>.csv
ood_csv <- function(key, suffix) file.path(DELIVERABLE, key, "csv", sprintf("%s_%s.csv", key, suffix))

generate_ood_plot <- function(key, iou_col, out_name, y_min = 0) {
  cat(sprintf("\n──── %s  [%s] ────\n", key, iou_col))
  PR_combined <- NULL
  labels <- character(0)
  for (model in MODELS) {
    suffix  <- if (model == "SAM3 ft.") "sam3" else "flatbug"
    csvp    <- ood_csv(key, suffix)
    d       <- load_data(csvp, 0)
    pr      <- make_pr_curve(d, iou_col)
    ap      <- compute_ap(d, iou_col)
    cat(sprintf("     %-9s AP50=%6.2f%%  AP50-95=%6.2f%%  (%s)\n",
                model, ap$AP50 * 100, ap$AP50_95 * 100, basename(csvp)))
    lab <- format_ap_label(model, ap$AP50, ap$AP50_95)
    labels[model] <- lab
    PR_combined <- bind_rows(
      PR_combined,
      pr %>% transmute(recall_curve, precision_curve, model = model, label = lab)
    )
  }
  # keep model order (SAM3 then FlatBug) so colours map correctly
  PR_combined$label <- factor(PR_combined$label, levels = labels[MODELS])
  pal <- setNames(unname(COL[MODELS]), labels[MODELS])

  y_step     <- if (y_min >= 0.5) 0.05 else 0.1
  y_mult     <- if (y_min >= 0.5) 20   else 10
  y_breaks   <- seq(y_min, 1, y_step)
  y_labeller <- function(b) ifelse(round(b * y_mult) %% 2 == 0,
                                   scales::label_percent(accuracy = 1)(b), "")

  p <- PR_combined %>%
    ggplot(aes(recall_curve, precision_curve, color = label)) +
    geom_line(linewidth = 1) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.1),
      labels = function(b) ifelse(round(b * 10) %% 2 == 0,
                                  scales::label_percent(accuracy = 1)(b), ""),
      expand = expansion(), limits = 0:1
    ) +
    scale_y_continuous(
      breaks = y_breaks, labels = y_labeller,
      expand = expansion(), limits = c(y_min, 1)
    ) +
    scale_color_manual(values = pal) +
    coord_cartesian(clip = "off") +
    labs(x = "Recall", y = "Precision", color = NULL, title = NULL) +
    guides(color = guide_legend(ncol = 1, keywidth = unit(1.2, "lines"), keyheight = unit(2, "lines"))) +
    theme(
      panel.border           = element_rect(colour = "black", fill = NA, linewidth = 0.5),
      axis.line              = element_blank(),
      panel.background       = element_rect(fill = "white"),
      panel.grid.major       = element_line(colour = "gray82", linewidth = 0.2),
      panel.grid.minor       = element_blank(),
      axis.ticks.length      = unit(-3, "pt"),
      axis.text              = element_text(size = 8),
      axis.text.x            = element_text(size = 8, margin = margin(t = 5)),
      axis.text.y            = element_text(size = 8, margin = margin(r = 5)),
      axis.title.x           = element_text(size = 9, margin = margin(t = 6)),
      axis.title.y           = element_text(size = 9, margin = margin(r = 6)),
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
      plot.title             = element_blank(),
      plot.margin            = margin(0.4, 1.1, 0.2, 0.2, "lines")
    )

  out_dir <- FIGURE_DIR
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir, out_name)
  ggsave(out_path, p, device = cairo_pdf, width = 3.0, height = 3.6)
  # PNG sibling (same layout) — for markdown previews; the PDF is the thesis asset.
  png_path <- sub("\\.pdf$", ".png", out_path)
  ggsave(png_path, p, device = "png", width = 3.0, height = 3.6, dpi = 300)
  cat(sprintf("     Saved: %s (+ .png)\n", out_path))
}

# ── Dataset definitions ───────────────────────────────────────────────────────
# key | iou_col | out_name | y_min. bbox for every set (GT = boxes); mask ONLY for
# MassID45, whose GT carries real instance polygons.
plots <- tribble(
  ~key,            ~iou_col,  ~out_name,                             ~y_min,
  "urban_insects", "IoU_bb",  "ood_urban_insects_pr_curve.pdf",      0,
  "massid45",      "IoU_bb",  "ood_massid45_bbox_pr_curve.pdf",      0,
  "massid45",      "IoU",     "ood_massid45_mask_pr_curve.pdf",      0,
  "pest24",        "IoU_bb",  "ood_pest24_pr_curve.pdf",             0
)

cat("\n========== OOD PR CURVES — THESIS FORMAT (SAM3 ft. vs FlatBug L) ==========\n")
pwalk(plots, generate_ood_plot)
cat("\n========== DONE ==========\n")
