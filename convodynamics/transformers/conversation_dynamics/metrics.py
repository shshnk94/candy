import os
from abc import ABC, abstractmethod
from typing import Any, List, Dict
import pandas as pd

def adaptability(
    speaker_a: pd.Series, 
    speaker_b: pd.Series
):

    """
    Calculate the adaptibility between two speakers using Spearman correlation.
    """

    return speaker_a.reset_index(drop=True).corr(speaker_b.reset_index(drop=True), method='spearman').item()

def predictability(
    speaker: pd.Series
):

    """
    Calculate the predictability of a speaker's turn lengths using lag-1 autocorrelation.
    """

    return speaker.autocorr(lag=1)

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
                mapping = {s: f"{s.lower().strip()}_{metric}" for s in speakers}
                scores.index = scores.index.map(mapping)
                results.update(scores.to_dict())

        # Handle conversation-level metrics
        if conversation_metrics: results.update(conversation_metrics)
        
        return results

class SpeakingTime(Feature):

    def __init__(self):
        super().__init__(name="speaking_time")

    def extract(
        self, 
        conversation: pd.DataFrame,
        total_duration: float):

        """
        Extract the speaking time for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their speaking times as values.
        """

        speaking_time = conversation.groupby("speaker")["duration"].sum() * 100 / total_duration

        return self._format_results(
            speaker_metrics={"speaking_time": speaking_time},
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
        Extract the average turn length for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their average turn lengths as values.
        """

        median = conversation.groupby('speaker')['duration'].median()

        mean = conversation.groupby('speaker')['duration'].mean()
        cv = conversation.groupby('speaker')['duration'].std() / mean

        speakers = conversation['speaker'].unique()
        speaker_a, speaker_b = (
            conversation.loc[conversation['speaker'] == speakers[0], "duration"], 
            conversation.loc[conversation['speaker'] == speakers[1], "duration"]
        )        
        
        return self._format_results(
            speaker_metrics={
                "turn_length_median": median,
                "turn_length_mean": mean,
                "turn_length_cv": cv,
                "turn_length_predictability": pd.Series({
                    speakers[0]: predictability(speaker_a),
                    speakers[1]: predictability(speaker_b)
                })
            },
            # FIXME: adaptability should be bidirectional?
            conversation_metrics={"turn_length_adaptability": adaptability(speaker_a, speaker_b)},
            speakers=speakers
        )

class Pauses(Feature):

    def __init__(self):
        super().__init__(name="pauses")

    def extract(
        self, 
        conversation: pd.DataFrame, 
        total_duration: float):

        """
        Extract the average pause percentage for each participant in the conversation.
        """

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
            speaker_metrics={"avg_pause_pct": pauses},
            speakers=speakers
        )

class SpeakerRate(Feature):

    def __init__(self):
        super().__init__(name="speaker_rate")

    def extract(
        self, 
        conversation: pd.DataFrame,
        text_field: str
    ):
    
        """
        Extract the speaking rate (words per minute) for each participant in the conversation.
        Since deciding on word boundaries is non-trivial, we use the provided transcript to count words.
        """

        def n_words(x):
            return len(x.split(' '))

        conversation["speechrate"] = conversation[text_field].apply(n_words) * 60 / conversation["meta.delta"]

        median = conversation.groupby('speaker')['speechrate'].median()
        mean = conversation.groupby('speaker')['speechrate'].mean()
        cv = conversation.groupby('speaker')['speechrate'].std() / mean

        speakers = conversation['speaker'].unique()
        speaker_a, speaker_b = (
            conversation.loc[conversation['speaker'] == speakers[0], "speechrate"], 
            conversation.loc[conversation['speaker'] == speakers[1], "speechrate"]
        )

        return self._format_results(
            speaker_metrics={
                "speaker_rate_median": median,
                "speaker_rate_cv": cv,
                "speaker_rate_predictability": pd.Series({
                    speakers[0]: predictability(speaker_a),
                    speakers[1]: predictability(speaker_b)
                })
            },
            conversation_metrics={"speaker_rate_adaptability": adaptability(speaker_a, speaker_b)},
            speakers=speakers
        )

class Backchannels(Feature):

    def __init__(self):
        super().__init__(name="backchannels")

    def is_backchannel(self, current_segment, segments):

        check = any(
            # check if current segment is within another segment
            (segments["start"] <= current_segment["start"]) & \
            (segments["end"] >= current_segment["end"]) & \
            # check if current segment is <= 1 second
            (current_segment["end"] - current_segment["start"] <= 1.0)
        )

        return check

    def extract(
        self, 
        conversation: pd.DataFrame,
        **kwargs
    ):

        conversation['backchannel'] = conversation.apply(self.is_backchannel, args=(conversation,), axis=1)
        turns = conversation.groupby("speaker").size().astype(float)
        backchannels = conversation.groupby("speaker")["backchannel"].sum().astype(float)

        speakers = conversation["speaker"].unique()
        
        # defined as the proportion of the number of turns taken by the other speaker that are backchannels
        backchannels.loc[speakers[0]] = backchannels.loc[speakers[0]] * 100 / turns.loc[speakers[1]]
        backchannels.loc[speakers[1]] = backchannels.loc[speakers[1]] * 100 / turns.loc[speakers[0]]

        return self._format_results(
            speaker_metrics={"backchannels": backchannels},
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

        conversation = conversation.sort_values(by="start").reset_index(drop=True)
        conversation['response_time'] = conversation['start'].shift(-1) - conversation['end']
    
        median = conversation.groupby('speaker')['response_time'].median()
        mean = conversation.groupby('speaker')['response_time'].mean()
        cv = conversation.groupby('speaker')['response_time'].std() / mean

        speakers = conversation['speaker'].unique()
        speaker_a, speaker_b = (
            conversation.loc[conversation['speaker'] == speakers[0], "response_time"], 
            conversation.loc[conversation['speaker'] == speakers[1], "response_time"]
        )

        return self._format_results(
            speaker_metrics={
                "response_time_median": median,
                "response_time_cv": cv,
                "response_time_predictability": pd.Series({
                    speakers[0]: predictability(speaker_a),
                    speakers[1]: predictability(speaker_b)
                })
            },
            conversation_metrics={"response_time_adaptability": adaptability(speaker_a, speaker_b)},
            speakers=speakers
        )