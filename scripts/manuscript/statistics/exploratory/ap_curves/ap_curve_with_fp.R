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

bb_data <- tibble(
  files = list.files("data", pattern = "csv$", full.names = T)
) %>% 
  filter(str_detect(files, "compare_backbone_sizes_._full")) %>% 
  mutate(
    data = map(files, data.table::fread),
    model = str_extract(files, "(?<=^data/compare_backbone_sizes_)[A-Z](?=_full.csv$)") %>% 
      factor(c("L","M","S","N"))
  ) %>% 
  select(!files) %>% 
  unnest(data) %>%
  mutate(
    area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    size = sqrt(area),
    dataset = str_extract(image, "^[^_]+"),
    short = short_name(dataset),
    result = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "FN",
      idx_1 == -1 & idx_2 != -1 ~ "FP",
      idx_1 != -1 & idx_2 != -1 ~ "TP"
    )
  ) %>% 
  filter(size >= 32) %>% 
  mutate(
    conf = replace_na(conf2, 0.)
  ) %>% 
  select(!c(conf1, conf2)) 

PR_curve <- bb_data %>% 
  group_by(model) %>% 
  arrange(desc(conf)) %>% 
  mutate(
    recall_curve = cumsum(result == "TP") / sum(result != "FP"),
    precision_curve = cumsum(result == "TP") / cumsum(result == "TP" | result == "FP")
  ) %>% 
  ungroup 

AP_curve <- tibble(
  IoU = seq(0.5, 0.95, 0.01)
) %>% 
  mutate(
    result = map(IoU, function(t) {
      bb_data %>% 
        filter(IoU >= t | result == "FN" | result == "FP") %>% 
        group_by(model) %>% 
        arrange(desc(conf)) %>% 
        mutate(
          recall_curve = cumsum(result == "TP") / sum(result != "FP"),
          precision_curve = cumsum(result == "TP") / cumsum(result == "TP" | result == "FP")
        ) %>% 
        summarize(
          AP = integrate_curve(recall_curve, precision_curve),
          w = which.min(abs(precision_curve - 0.975)),
          P = precision_curve[w],
          R = recall_curve[w]
        ) %>% 
        select(!w)
    }, .progress = if (show_progress()) "Computing AP Curve" else F)
  ) %>% 
  unnest(result)

AP_results <- AP_curve %>% 
  group_by(model) %>% 
  summarize(
    AP50 = AP[IoU == 0.5],
    AP50_95 = mean(AP[between(IoU, 0.5, 0.95)])
  )

PR_plt <- PR_curve %>% 
  left_join(
    AP_results %>% 
      arrange(model) %>% 
      mutate(
        label = str_c(model, " (", "AP50%: ", paste0(format(round(AP50 * 100, 2), nsmall = 2), "%"), " | AP50-95%: ", paste0(format(round(AP50_95 * 100, 2), nsmall = 2), "%"), ")") %>% 
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
    limits = c(0.9, 1),
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
    legend.position.inside = c(0.5, 0.15),
    plot.margin = margin(0.5, 1, 0, 0.25, "lines")
  )

ggsave(
  "figures/pr_curve_ap_flatbug_small_correction_1.pdf", 
  PR_plt,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 2.75,
  antialias = "subpixel"
)

AP_ltx <- AP_results %>% 
  pivot_longer(c(AP50, AP50_95), names_to = "metric") %>% 
  mutate(
    ltx = str_c("\\defexperiment{ap}{", model, str_remove(metric, "^AP"), "}{", round(value, 3), "}")
  ) %>% 
  pull(ltx) %>% 
  str_c(collapse = "\n") 

tryCatch({
  add_group("Addendum (experiment 1) - Average Precision")
}, error = function(e) {
  message("Latex data group already exists; skipping add_group().")
})
write_data("Addendum (experiment 1) - Average Precision", AP_ltx)
