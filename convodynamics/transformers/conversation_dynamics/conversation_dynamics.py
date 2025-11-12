from typing import Any, List, Dict
import pandas as pd
from convokit import Corpus
from convokit.transformer import Transformer

from .metrics import (
    Feature, 
    TurnLength,
    SpeakingTime,
    Pauses,
    SpeakerRate,
    Backchannels,
    ResponseTime
)

# simple registry to allow registering by name (case/format agnostic)
_METRICS_REGISTRY = {
    "speaking_time": SpeakingTime,
    "turn_length": TurnLength,
    "pauses": Pauses,
    "speaker_rate": SpeakerRate,
    "backchannels": Backchannels,
    "response_time": ResponseTime,
}

class ConversationDynamicsTransformer(Transformer):

    def __init__(self):

        pass

    def register_metrics(
        self,
        metrics: List[str]
    ) -> None:
        
        """
        Register a list of feature extraction metrics.
        """

        self.metrics = []
        for metric_name in metrics:

            # some robustness in matching metric names
            metric_name = metric_name.lower().strip()
            if metric_name in _METRICS_REGISTRY:
                metric = _METRICS_REGISTRY[metric_name]()
                self.metrics.append(metric)

            else:
                raise ValueError(f"Metric '{metric_name}' not recognized. Available metrics: {list(_METRICS_REGISTRY.keys())}")

    def remove_shortest_speaker(
        self, 
        segments: pd.DataFrame
    ):
    
        """
        Remove the speaker with the shortest total speaking time from the segments DataFrame.
        """

        shortest_speaker = segments.groupby("speaker")["duration"].sum().idxmin()
        segments = segments[segments["speaker"] != shortest_speaker]
    
        return segments

    def transform(
        self,
        corpus: Corpus
    ):

        for conversation in corpus.iter_conversations():

            segments = conversation.retrieve_meta("diarization_segments")

            # if diarization segments are available, we will use them
            if segments is not None:

                total_duration = conversation.retrieve_meta("total_duration")

                # remove shortest speaker -- usually noise
                if segments['speaker'].nunique() > 2:
                    segments = self.remove_shortest_speaker(segments)

                metrics = {}
                for metric in self.metrics:

                    metric_name = metric.get_name
                    print("Extracting feature:", metric_name)
    
                    if metric_name == "speaker_rate":

                        transcripts = conversation.get_utterances_dataframe(exclude_meta=False)
                        metrics[metric_name] = metric(
                            conversation=transcripts, 
                            text_field="text"
                        )

                        continue

                    metrics[metric_name] = metric(
                        conversation=segments,
                        total_duration=total_duration
                    ) 

            # if no diarization segments, we will use the utterance transcripts
            else:

                transcripts = conversation.get_utterances_dataframe(exclude_meta=True)

                metrics = {}
                for metric in self.metrics:

                    metric_name = metric.get_name
                    print("Extracting feature:", metric_name)

                    metrics[metric_name] = metric(
                        conversation=transcripts, 
                        text_field="text"
                    )

            conversation.add_meta("conversation_dynamics_features", metrics)
        
        return corpus