# =============================================================================
# Conversation Dynamics — Ready-to-Use Functions (REVISED)
# -----------------------------------------------------------------------------
# Authors: 
#   Zhenchao, Shashanka, & Nilam
#
# Key revision:
# - Correlations are now "safe": if insufficient complete pairs, return NA
#   rather than erroring (fixes the 14 failures).
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
})

# ---- Internal helpers --------------------------------------------------------

# Ensure required columns exist
cd_check_cols <- function(audio, cols) {
  missing <- setdiff(cols, names(audio))
  if (length(missing)) {
    stop("Missing required column(s): ", paste(missing, collapse = ", "))
  }
}

# Coerce 'overlap' to logical: supports logical, numeric 0/1, or "True"/"False"
cd_as_logical_overlap <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x)) return(x != 0)
  x <- tolower(as.character(x))
  ifelse(x %in% c("true","t","1"), TRUE,
         ifelse(x %in% c("false","f","0"), FALSE, NA))
}

# NEW: Safe correlation (returns NA if <2 complete pairs instead of erroring)
cd_safe_cor <- function(x, y, method = "spearman") {
  ok <- complete.cases(x, y)
  if (sum(ok) < 2) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = method))
}

# (Optional) Add `overlap` if missing: TRUE when current turn starts before
# the previous turn ends AND speakers differ.
cd_add_overlap <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop"))
  if ("overlap" %in% names(audio)) {
    # CHANGED: normalize overlap to logical even if it already exists
    return(audio %>% mutate(overlap = cd_as_logical_overlap(overlap)))
  }
  audio %>%
    arrange(start) %>%
    mutate(
      prev_speaker = lag(speaker),
      prev_stop    = lag(stop),
      overlap = if_else(!is.na(prev_stop) & start < prev_stop & speaker != prev_speaker,
                        TRUE, FALSE)
    )
}

# (Optional) Derive n_words from a text column
cd_add_n_words <- function(audio, text_col = "utterance") {
  cd_check_cols(audio, text_col)
  if ("n_words" %in% names(audio)) return(audio)
  if (!requireNamespace("stringr", quietly = TRUE))
    stop("Please install 'stringr' to derive n_words from text.")
  audio %>%
    mutate(n_words = stringr::str_count(.data[[text_col]], "\\S+"))
}

# ---- Metrics: Speaking Time --------------------------------------------------

cd_speaking_time <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop"))
  audio %>%
    mutate(duration = stop - start) %>%
    group_by(speaker) %>%
    summarise(total_duration = sum(duration, na.rm = TRUE), .groups = "drop") %>%
    mutate(share = total_duration / sum(total_duration))
}

# ---- Metrics: Turn Length ----------------------------------------------------

cd_turn_length_metrics <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop"))
  aug <- audio %>%
    arrange(start) %>%
    mutate(duration = stop - start,
           prev_speaker  = lag(speaker),
           prev_duration = lag(duration))
  
  med <- aug %>%
    group_by(speaker) %>%
    summarise(tl_median = median(duration, na.rm = TRUE), .groups = "drop")
  
  cv  <- aug %>%
    group_by(speaker) %>%
    summarise(tl_cv = sd(duration, na.rm = TRUE) / mean(duration, na.rm = TRUE),
              .groups = "drop")
  
  adapt <- aug %>%
    filter(prev_speaker != speaker) %>%
    group_by(speaker) %>%
    summarise(
      # CHANGED: safe cor
      tl_adapt = cd_safe_cor(duration, prev_duration, method = "spearman"),
      .groups = "drop"
    )
  
  pred <- aug %>%
    group_by(speaker) %>%
    arrange(start, .by_group = TRUE) %>%
    mutate(prev_own = lag(duration)) %>%
    summarise(
      # CHANGED: safe cor
      tl_predict = cd_safe_cor(duration, prev_own, method = "spearman"),
      .groups = "drop"
    )
  
  med %>% left_join(cv, by = "speaker") %>%
    left_join(adapt, by = "speaker") %>%
    left_join(pred,  by = "speaker")
}

# ---- Metrics: Speech Rate (WPM) ---------------------------------------------

cd_speech_rate_metrics <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop","n_words"))
  dat <- audio %>%
    arrange(start) %>%
    mutate(
      duration_min = (stop - start) / 60,
      duration_min = if_else(is.finite(duration_min) & duration_min > 0, duration_min, NA_real_),
      wpm = n_words / duration_min
    )
  
  med <- dat %>%
    group_by(speaker) %>%
    summarise(sr_median = median(wpm, na.rm = TRUE), .groups = "drop")
  
  cv  <- dat %>%
    group_by(speaker) %>%
    summarise(sr_cv = sd(wpm, na.rm = TRUE) / mean(wpm, na.rm = TRUE),
              .groups = "drop")
  
  adapt <- dat %>%
    mutate(prev_speaker = lag(speaker), prev_wpm = lag(wpm)) %>%
    filter(prev_speaker != speaker) %>%
    group_by(speaker) %>%
    summarise(
      # CHANGED: safe cor
      sr_adapt = cd_safe_cor(wpm, prev_wpm, method = "spearman"),
      .groups = "drop"
    )
  
  pred <- dat %>%
    group_by(speaker) %>%
    arrange(start, .by_group = TRUE) %>%
    mutate(prev_own_wpm = lag(wpm)) %>%
    summarise(
      # CHANGED: safe cor
      sr_predict = cd_safe_cor(wpm, prev_own_wpm, method = "spearman"),
      .groups = "drop"
    )
  
  med %>% left_join(cv, by = "speaker") %>%
    left_join(adapt, by = "speaker") %>%
    left_join(pred,  by = "speaker")
}

# ---- Metrics: Backchannels ---------------------------------------------------

cd_backchannel_rate <- function(audio, duration_threshold = 1) {
  cd_check_cols(audio, c("speaker","start","stop","overlap"))
  audio %>%
    mutate(
      duration = stop - start,
      overlap = cd_as_logical_overlap(overlap),
      backchannel = if_else(!is.na(overlap) & overlap & duration < duration_threshold, 1L, 0L)
    ) %>%
    group_by(speaker) %>%
    summarise(
      turns_total = n(),
      backchannel_n = sum(backchannel, na.rm = TRUE),
      backchannel_prop = backchannel_n / turns_total,
      .groups = "drop"
    )
}

# ---- Metrics: Response Time --------------------------------------------------

cd_response_time_metrics <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop","overlap"))
  dat <- audio %>%
    arrange(start) %>%
    mutate(
      overlap = cd_as_logical_overlap(overlap),
      prev_speaker = lag(speaker),
      prev_stop    = lag(stop),
      response_time = if_else(!overlap & prev_speaker != speaker, start - prev_stop, NA_real_)
    )
  
  med <- dat %>%
    group_by(speaker) %>%
    summarise(rt_median = median(response_time, na.rm = TRUE), .groups = "drop")
  
  cv  <- dat %>%
    group_by(speaker) %>%
    summarise(rt_cv = sd(response_time, na.rm = TRUE) / mean(response_time, na.rm = TRUE),
              .groups = "drop")
  
  adapt <- dat %>%
    mutate(prev_rt = lag(response_time), prev_speaker2 = lag(speaker)) %>%
    filter(prev_speaker2 != speaker) %>%
    group_by(speaker) %>%
    summarise(
      # CHANGED: safe cor
      rt_adapt = cd_safe_cor(response_time, prev_rt, method = "spearman"),
      .groups = "drop"
    )
  
  pred <- dat %>%
    group_by(speaker) %>%
    arrange(start, .by_group = TRUE) %>%
    mutate(prev_own_rt = lag(response_time)) %>%
    summarise(
      # CHANGED: safe cor
      rt_predict = cd_safe_cor(response_time, prev_own_rt, method = "spearman"),
      .groups = "drop"
    )
  
  med %>% left_join(cv, by = "speaker") %>%
    left_join(adapt, by = "speaker") %>%
    left_join(pred,  by = "speaker")
}

# ---- Wrapper: All Metrics ----------------------------------------------------

cd_all_metrics <- function(audio) {
  list(
    speaking_time = cd_speaking_time(audio),
    turn_length   = cd_turn_length_metrics(audio),
    speech_rate   = if ("n_words" %in% names(audio)) cd_speech_rate_metrics(audio) else NULL,
    backchannel   = if ("overlap" %in% names(audio)) cd_backchannel_rate(audio) else NULL,
    response_time = if ("overlap" %in% names(audio)) cd_response_time_metrics(audio) else NULL
  )
}


# =============================================================================
# Batch processing script (unchanged logic; now should yield 0 failures)
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(stringr)
  library(tibble)
})

transcript_dir <- "~/Desktop/downloaded_transcripts_backbiter"

files <- list.files(
  path = transcript_dir,
  pattern = "^transcript_backbiter_.*\\.csv$",
  full.names = TRUE
)

length(files)  # expect 1596

extract_conversation_id <- function(path) {
  fn <- basename(path)
  id <- str_match(fn, "^transcript_backbiter_(.+)\\.csv$")[, 2]
  if (is.na(id)) stop("Could not extract conversation_id from: ", fn)
  id
}

process_one_transcript <- function(path) {
  conversation_id <- extract_conversation_id(path)
  
  audio <- readr::read_csv(path, show_col_types = FALSE)
  
  # Ensure overlap exists + standardized
  audio <- cd_add_overlap(audio)
  
  # Ensure n_words exists
  if (!("n_words" %in% names(audio))) {
    audio <- cd_add_n_words(audio, text_col = "utterance")
  }
  
  res <- cd_all_metrics(audio)
  
  metrics_tbl <- res$speaking_time %>%
    full_join(res$turn_length,   by = "speaker") %>%
    { if (!is.null(res$speech_rate))   full_join(., res$speech_rate,   by = "speaker") else . } %>%
    { if (!is.null(res$backchannel))   full_join(., res$backchannel,   by = "speaker") else . } %>%
    { if (!is.null(res$response_time)) full_join(., res$response_time, by = "speaker") else . } %>%
    mutate(conversation_id = conversation_id, .before = 1)
  
  metrics_tbl
}

safe_process <- purrr::safely(process_one_transcript)
results <- purrr::map(files, safe_process)

ok  <- purrr::map(results, "result")
err <- purrr::map(results, "error")

combined <- bind_rows(ok)

error_log <- tibble(
  file = files,
  error = purrr::map_chr(err, ~ if (is.null(.x)) NA_character_ else conditionMessage(.x))
) %>%
  filter(!is.na(error))

cat("Processed:", length(files) - nrow(error_log), "files successfully\n")
if (nrow(error_log) > 0) {
  cat("Failures:", nrow(error_log), "\n")
  print(error_log)
}

write_csv(combined, "candor_conversation_dynamics_metrics.csv")
combined
