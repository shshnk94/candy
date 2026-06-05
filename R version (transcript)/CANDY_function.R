# =============================================================================
# Conversation Dynamics — Ready-to-Use Functions
# -----------------------------------------------------------------------------
# Authors: 
#   Zhenchao, Shashanka, & Nilam
# 
# Assumptions:
# - Input data.frame `audio` has at least: speaker, start, stop (row = turn).
# - Optional columns:
#     * n_words   (for speech-rate metrics; see helper to compute from text)
#     * overlap   (logical / 0-1 / "True"/"False"); see helper to add if missing
#
# Output:
# - Each metric function returns a tibble with one row per `speaker`.
# - A convenience wrapper `cd_all_metrics()` returns a named list of results.
#
# Usage:
#   source("ConvoDynamics.R")
#   res <- cd_all_metrics(audio)
#   res$speaking_time; res$turn_length; 
#   res$speech_rate; res$backchannel; res$response_time
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

# (Optional) Add `overlap` if missing: TRUE when current turn starts before
# the previous turn ends AND speakers differ.
cd_add_overlap <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop"))
  if ("overlap" %in% names(audio)) return(audio)
  audio %>%
    arrange(start) %>%
    mutate(
      prev_speaker = lag(speaker),
      prev_stop    = lag(stop),
      overlap = if_else(!is.na(prev_stop) & start < prev_stop & speaker != prev_speaker,
                        TRUE, FALSE)
    )
}

# (Optional) Derive n_words from a text column (commented pattern shown here)
cd_add_n_words <- function(audio, text_col = "utterance") {
  cd_check_cols(audio, text_col)
  if ("n_words" %in% names(audio)) return(audio)
  if (!requireNamespace("stringr", quietly = TRUE))
    stop("Please install 'stringr' to derive n_words from text.")
  audio %>%
    mutate(n_words = stringr::str_count(.data[[text_col]], "\\S+"))
  }

# ---- Metrics: Speaking Time --------------------------------------------------

# Speaking time share per speaker (proportion of total talk time)
cd_speaking_time <- function(audio) {
  cd_check_cols(audio, c("speaker","start","stop"))
  audio %>%
    mutate(duration = stop - start) %>%
    group_by(speaker) %>%
    summarise(total_duration = sum(duration, na.rm = TRUE), .groups = "drop") %>%
    mutate(share = total_duration / sum(total_duration))
}

# ---- Metrics: Turn Length ----------------------------------------------------

# Turn-length metrics:
#   tl_median  = median(duration)
#   tl_cv      = sd(duration) / mean(duration)
#   tl_adapt   = Spearman cor of duration_t with counterpart duration_{t-1}
#   tl_predict = Spearman cor of duration_t with own duration_{t-1}
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
    summarise(tl_adapt = cor(duration, prev_duration,
                             method = "spearman", use = "complete.obs"),
              .groups = "drop")
  
  pred <- aug %>%
    group_by(speaker) %>%
    arrange(start, .by_group = TRUE) %>%
    mutate(prev_own = lag(duration)) %>%
    summarise(tl_predict = cor(duration, prev_own,
                               method = "spearman", use = "complete.obs"),
              .groups = "drop")
  
  med %>% left_join(cv, by = "speaker") %>%
    left_join(adapt, by = "speaker") %>%
    left_join(pred,  by = "speaker")
}

# ---- Metrics: Speech Rate (WPM) ---------------------------------------------

# Speech-rate metrics (requires n_words):
#   sr_median  = median(WPM)
#   sr_cv      = sd(WPM)/mean(WPM)
#   sr_adapt   = Spearman cor of WPM_t with counterpart WPM_{t-1}
#   sr_predict = Spearman cor of WPM_t with own WPM_{t-1}
# Guards against zero/invalid durations.
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
    summarise(sr_adapt = cor(wpm, prev_wpm, method = "spearman", use = "complete.obs"),
              .groups = "drop")
  
  pred <- dat %>%
    group_by(speaker) %>%
    arrange(start, .by_group = TRUE) %>%
    mutate(prev_own_wpm = lag(wpm)) %>%
    summarise(sr_predict = cor(wpm, prev_own_wpm, method = "spearman", use = "complete.obs"),
              .groups = "drop")
  
  med %>% left_join(cv, by = "speaker") %>%
    left_join(adapt, by = "speaker") %>%
    left_join(pred,  by = "speaker")
}

# ---- Metrics: Backchannels ---------------------------------------------------

# Backchannel proportion per speaker (default: overlap == TRUE & duration < 1 sec):
# Returns turns_total, backchannel_n, backchannel_prop
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

# Response-time metrics (requires overlap):
# Definition: silence between previous turn's stop and current turn's start,
# using only non-overlapping, cross-speaker turn taking.
# Returns: rt_median, rt_cv, rt_adapt, rt_predict
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
    summarise(rt_adapt = cor(response_time, prev_rt, method = "spearman", 
                             use = "complete.obs"),
              .groups = "drop")
  
  pred <- dat %>%
    group_by(speaker) %>%
    arrange(start, .by_group = TRUE) %>%
    mutate(prev_own_rt = lag(response_time)) %>%
    summarise(rt_predict = cor(response_time, prev_own_rt, method = "spearman", 
                               use = "complete.obs"),
              .groups = "drop")
  
  med %>% left_join(cv, by = "speaker") %>%
    left_join(adapt, by = "speaker") %>%
    left_join(pred,  by = "speaker")
}

# ---- Wrapper: All Metrics ----------------------------------------------------

# Returns a named list of available metrics. Functions requiring missing columns return NULL.
cd_all_metrics <- function(audio) {
  list(
    speaking_time = cd_speaking_time(audio),
    turn_length   = cd_turn_length_metrics(audio),
    speech_rate   = if ("n_words" %in% names(audio)) cd_speech_rate_metrics(audio) else NULL,
    backchannel   = if ("overlap" %in% names(audio)) cd_backchannel_rate(audio) else NULL,
    response_time = if ("overlap" %in% names(audio)) cd_response_time_metrics(audio) else NULL
  )
}

