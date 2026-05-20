import os
from abc import ABC, abstractmethod
from typing import Any, List, Dict
import pandas as pd

# FIXED: new helper. Used by Backchannels and ResponseTime to consume the
# dataset's precomputed `overlap` column instead of re-deriving it.
def _coerce_overlap(series: pd.Series) -> pd.Series:

    """
    Coerce an `overlap` column to a boolean Series. Mirrors R's
    `cd_as_logical_overlap`: accepts logical, numeric 0/1, or
    "True"/"False"-style strings.
    """

    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"true", "t", "1"})

# FIXED: was `pd.Series.autocorr(lag=1)` (Pearson); R uses Spearman.
def predictability(values: pd.Series) -> float:

    """
    Lag-1 Spearman autocorrelation of a single speaker's chronologically
    ordered series. NaNs are handled pairwise by pandas (matches R's
    `use = "complete.obs"`).
    """

    if len(values) < 2:
        return float("nan")
    return values.corr(values.shift(1), method="spearman")

# FIXED: was a single conversation-level Spearman pairing speaker-A's i-th
# turn with speaker-B's i-th turn after `reset_index` (positional pairing —
# ignored actual turn-taking). Now: per-speaker, lagged across the full
# sorted conversation, restricted to cross-speaker transitions ending at
# `speaker` — matches R's cor(value, prev_value) inside group_by(speaker)
# after filter(prev_speaker != speaker).
def adaptability(
    conversation: pd.DataFrame,
    value_col: str,
    speaker: str
) -> float:

    """
    Per-speaker adaptability: Spearman correlation of `value_col` at turn t
    (current speaker) with `value_col` at turn t-1 (counterpart's previous
    turn), restricted to genuine cross-speaker transitions ending at
    `speaker`. Caller passes `conversation` already sorted by `start`.
    """

    prev_value = conversation[value_col].shift(1)
    prev_speaker = conversation["speaker"].shift(1)
    mask = (conversation["speaker"] == speaker) & (prev_speaker != speaker) & prev_speaker.notna()
    if mask.sum() < 2:
        return float("nan")
    return conversation.loc[mask, value_col].corr(prev_value.loc[mask], method="spearman")

class Feature(ABC):

    def __init__(
        self,
        name: str
    ):
        self._name = name

    @abstractmethod
    def extract(
        self,
        conversation: Any,
        **optional_kwargs
    ):
        """
        Extract the feature from the conversation data.
        This method should be implemented by all subclasses.
        """
        pass

    @property
    def get_name(self) -> str:
        return self._name

    def __call__(
        self,
        conversation: Any,
        **kwargs
    ):
        return self.extract(
            conversation=conversation,
            **kwargs
        )

    def _format_results(
        self,
        speaker_metrics: Dict[str, pd.Series] = None,
        conversation_metrics: Dict[str, float] = None,
        speakers: List[str] = None
    ):

        """
        Helper method to format results consistently across all feature extractors.

        Args:
            speaker_metrics: Dict with metric names as keys, Series with speaker data as values
            conversation_metrics: Dict with global metric names and their values
        """

        results = {}

        # Handle speaker-specific metrics
        if speaker_metrics:
            for metric, scores in speaker_metrics.items():
                #FIXME: check if all speakers are present in scores.index
                mapping = {s: f"{s.lower().strip()}.{metric}" for s in speakers}
                scores.index = scores.index.map(mapping)
                results.update(scores.to_dict())

        # Handle conversation-level metrics
        if conversation_metrics:
            results.update(conversation_metrics)

        return results

class SpeakingTime(Feature):

    def __init__(self):
        super().__init__(name="speaking_time")

    def extract(
        self,
        conversation: pd.DataFrame,
        total_duration: float = None
    ):

        """
        Extract speaking time per participant.
        Returns `total_duration` (seconds) and `share` (proportion of all
        speakers' summed durations) per speaker, matching R's
        `cd_speaking_time`.
        """

        turn_duration_field = "delta" if "text" in conversation.columns else "duration"

        durations = conversation.groupby("speaker")[turn_duration_field].sum()
        # FIXED: was `× 100 / total_duration` (percentage). R's cd_speaking_time
        # returns share as a proportion of the sum of speaker durations.
        share = durations / durations.sum()

        return self._format_results(
            speaker_metrics={
                "total_duration": durations,
                "share": share,
            },
            speakers=conversation['speaker'].unique()
        )

class TurnLength(Feature):

    def __init__(self):
        super().__init__(name="turn_length")

    def extract(
        self,
        conversation: pd.DataFrame,
        **kwargs):

        """
        Per-speaker turn-length statistics: median, mean, CV, lag-1 Spearman
        autocorrelation (predictability), and per-speaker Spearman
        adaptability (current turn duration vs. counterpart's previous turn
        duration over cross-speaker transitions).
        """

        turn_duration_field = "delta" if "text" in conversation.columns else "duration"
        # FIXED: kind="stable" preserves turn_id order at start ties, matching
        # R's arrange(start) on the source CSV. (See conversation_dynamics.py
        # for the upstream turn_id-based pre-sort.)
        conversation = conversation.sort_values(by="start", kind="stable").reset_index(drop=True)

        median = conversation.groupby('speaker')[turn_duration_field].median()
        mean = conversation.groupby('speaker')[turn_duration_field].mean()
        cv = conversation.groupby('speaker')[turn_duration_field].std() / mean

        speakers = conversation['speaker'].unique()

        # FIXED: predict and adapt are now per-speaker (one value each via the
        # helpers at the top of this file). Previous code emitted a single
        # conversation-level `adaptability` and Pearson autocorr `predictability`.
        predict = pd.Series({
            s: predictability(conversation.loc[conversation['speaker'] == s, turn_duration_field])
            for s in speakers
        })
        adapt = pd.Series({
            s: adaptability(conversation, turn_duration_field, s)
            for s in speakers
        })

        return self._format_results(
            speaker_metrics={
                "median": median,
                "mean": mean,
                "cv": cv,
                "predictability": predict,
                "adaptability": adapt,
            },
            speakers=speakers
        )

class Pauses(Feature):

    def __init__(self):
        super().__init__(name="pauses")

    def extract(
        self,
        conversation: pd.DataFrame,
        total_duration: float = None
        ):

        """
        Extract the average pause percentage for each participant in the conversation.
        """

        if "text" in conversation.columns:
            print("Warning: Pauses feature is designed to work with raw segments without transcripts.")
            return {}

        # check the next speaker
        conversation["next_speaker"] = conversation["speaker"].shift(-1)
        conversation["next_start"] = conversation["start"].shift(-1)

        # gap can be negative if there is overlap
        conversation["gap"] = conversation["next_start"] - conversation["end"]

        # find within-turn gaps greater than 180ms
        mask = (conversation["speaker"] == conversation["next_speaker"])
        mask = mask & (conversation["gap"] > 0.18)

        speaking_time = conversation.groupby("speaker")["duration"].sum()
        pauses = conversation.loc[mask, ["speaker", "gap"]] \
            .groupby("speaker")["gap"] \
            .size() / speaking_time

        speakers = conversation['speaker'].unique()
        return self._format_results(
            speaker_metrics={"pause_pct": pauses},
            speakers=speakers
        )

class SpeakerRate(Feature):

    def __init__(self):
        super().__init__(name="speaker_rate")

    def extract(
        self,
        conversation: pd.DataFrame,
        total_duration: float = None
    ):

        """
        Per-speaker speaking rate (words per minute): median, mean, CV,
        Spearman predictability, and per-speaker Spearman adaptability.
        Requires the transcript text column.
        """

        if "text" not in conversation.columns:
            print("Warning: SpeakerRate feature requires transcript data.")
            return {}

        # FIXED: kind="stable" preserves turn_id order at start ties (see TurnLength).
        conversation = conversation.sort_values(by="start", kind="stable").reset_index(drop=True).copy()
        # FIXED: prefer CANDOR's precomputed `n_words`. Previous code did
        # `len(text.split(' '))`, which overcounts when text has multiple
        # consecutive spaces. The fallback uses R's stringr::str_count("\\S+")
        # equivalent.
        if "n_words" in conversation.columns:
            n_words_series = conversation["n_words"].astype(float)
        else:
            n_words_series = conversation["text"].str.count(r"\S+").astype(float)
        # FIXED: WPM is now `n_words / ((stop - start) / 60)` to match R's
        # float-op order exactly. Using the precomputed `delta` column can
        # differ from `stop - start` by ~1 ULP — enough to flip rank order at
        # near-ties for sr_adapt and sr_predict (Spearman is rank-based).
        conversation["speechrate"] = n_words_series / ((conversation["stop"] - conversation["start"]) / 60)

        median = conversation.groupby('speaker')['speechrate'].median()
        mean = conversation.groupby('speaker')['speechrate'].mean()
        cv = conversation.groupby('speaker')['speechrate'].std() / mean

        speakers = conversation['speaker'].unique()

        # FIXED: per-speaker predict and adapt (previously a single
        # conversation-level adaptability and Pearson autocorr predictability).
        predict = pd.Series({
            s: predictability(conversation.loc[conversation['speaker'] == s, "speechrate"])
            for s in speakers
        })
        adapt = pd.Series({
            s: adaptability(conversation, "speechrate", s)
            for s in speakers
        })

        return self._format_results(
            speaker_metrics={
                "median": median,
                "mean": mean,
                "cv": cv,
                "predictability": predict,
                "adaptability": adapt,
            },
            speakers=speakers
        )

class Backchannels(Feature):

    def __init__(self):
        super().__init__(name="backchannels")

    def extract(
        self,
        conversation: pd.DataFrame,
        **kwargs
    ):

        """
        Backchannels follow R's definition: a turn is a backchannel when it
        overlaps with the immediately preceding (different-speaker) turn and
        is shorter than 1 second. Returns turns_total, backchannel_n, and
        backchannel_prop (proportion of own turns) per speaker.
        """

        duration_field = "delta" if "text" in conversation.columns else "duration"

        # FIXED: kind="stable" preserves turn_id order at start ties.
        conversation = conversation.sort_values(by="start", kind="stable").reset_index(drop=True).copy()

        # FIXED: previous implementation defined a backchannel as a turn fully
        # contained inside another segment. R's definition (and what we now
        # use) is: turn whose `overlap` flag is true AND duration < 1s. Prefer
        # CANDOR's precomputed `overlap`; fall back to deriving it the same
        # way R's cd_add_overlap does.
        if "overlap" in conversation.columns:
            overlap = _coerce_overlap(conversation["overlap"])
        else:
            prev_speaker = conversation["speaker"].shift(1)
            prev_stop = conversation["stop"].shift(1)
            overlap = (conversation["start"] < prev_stop) & (conversation["speaker"] != prev_speaker)

        conversation["backchannel"] = (overlap & (conversation[duration_field] < 1.0)).astype(int)

        # FIXED: backchannel_prop is now `backchannel_n / own_turns_total`
        # (proportion of own turns), matching R. Previous code reported
        # `count_speaker_X * 100 / turns_of_OTHER_speaker` (percentage of the
        # other speaker's turns).
        turns_total = conversation.groupby("speaker").size().astype(float)
        backchannel_n = conversation.groupby("speaker")["backchannel"].sum().astype(float)
        backchannel_prop = backchannel_n / turns_total

        speakers = conversation["speaker"].unique()
        return self._format_results(
            speaker_metrics={
                "turns_total": turns_total,
                "backchannel_n": backchannel_n,
                "backchannel_prop": backchannel_prop,
            },
            speakers=speakers
        )

class ResponseTime(Feature):

    def __init__(self):
        super().__init__(name="response_time")

    def extract(
        self,
        conversation: pd.DataFrame,
        **kwargs
    ):

        """
        Per-speaker response-time statistics. Response time is defined only
        on genuine cross-speaker, non-overlapping transitions:
        `response_time = start_t - stop_{t-1}` when `prev_speaker != speaker`
        and the current turn does not overlap the previous one. Other rows
        get NA so they are excluded from medians/means/correlations.
        """

        # FIXED: kind="stable" preserves turn_id order at start ties.
        conversation = conversation.sort_values(by="start", kind="stable").reset_index(drop=True).copy()

        prev_speaker = conversation["speaker"].shift(1)
        prev_stop = conversation["stop"].shift(1)
        # FIXED: was `start.shift(-1) - stop` for every row regardless of speaker
        # or overlap (so within-speaker pauses and overlapping turns leaked into
        # the response-time series). Now: response_time is defined only on
        # genuine cross-speaker, non-overlapping transitions, matching R.
        # Prefer CANDOR's precomputed `overlap`; fall back to a derivation.
        if "overlap" in conversation.columns:
            overlap = _coerce_overlap(conversation["overlap"])
        else:
            overlap = (conversation["start"] < prev_stop) & (conversation["speaker"] != prev_speaker)
        valid = (prev_speaker != conversation["speaker"]) & prev_speaker.notna() & ~overlap

        conversation["response_time"] = (conversation["start"] - prev_stop).where(valid)

        median = conversation.groupby('speaker')['response_time'].median()
        mean = conversation.groupby('speaker')['response_time'].mean()
        cv = conversation.groupby('speaker')['response_time'].std() / mean

        speakers = conversation['speaker'].unique()

        # FIXED: per-speaker predict and adapt (previously a single
        # conversation-level adaptability and Pearson autocorr predictability).
        predict = pd.Series({
            s: predictability(conversation.loc[conversation['speaker'] == s, "response_time"])
            for s in speakers
        })
        adapt = pd.Series({
            s: adaptability(conversation, "response_time", s)
            for s in speakers
        })

        return self._format_results(
            speaker_metrics={
                "median": median,
                "mean": mean,
                "cv": cv,
                "predictability": predict,
                "adaptability": adapt,
            },
            speakers=speakers
        )
