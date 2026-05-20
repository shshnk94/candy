from pathlib import Path
from typing import List
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

    def transform(
        self,
        corpus: Corpus
    ):

        for conversation in corpus.iter_conversations():

            transcripts = conversation.get_utterances_dataframe(exclude_meta=False)
            transcripts = transcripts.rename(columns=lambda c: c.split(".", 1)[1] if c.startswith("meta.") else c)
            # FIXED: ConvoKit may reload utterances in a non-CSV order, which
            # would change tie-break behavior for metrics that sort by `start`.
            # The converter encodes turn_id in the utterance id as
            # `{conv_id}_{turn_id}` — recover it and re-sort to canonical
            # CSV order so stable-sort-by-start matches R's CSV-input order.
            transcripts["turn_id"] = transcripts.index.to_series().str.rsplit("_", n=1).str[-1].astype(int)
            transcripts = transcripts.sort_values("turn_id").reset_index(drop=True)

            # extract all registered metrics
            metrics = {}
            for metric in self.metrics:

                metric_name = metric.get_name
                print("Extracting feature:", metric_name)

                metrics[metric_name] = metric(conversation=transcripts)

            conversation.add_meta("conversation_dynamics_features", metrics)

        return corpus
    
    def export(
        self,
        corpus: Corpus,
        output_path: Path
    ):
    
        rows = []
        for conversation in corpus.iter_conversations():

            features = conversation.retrieve_meta("conversation_dynamics_features")
            if features:
            # Map speaker IDs → generic sorted labels to align columns across conversations
                speaker_ids = sorted(conversation.get_speaker_ids())
                speaker_map = {sid.lower().strip(): f"speaker.{i}" for i, sid in enumerate(speaker_ids)}

            def normalize_key(key):
                for sid, label in speaker_map.items():
                    if key.startswith(sid):
                        return label + key[len(sid):]
                return key

            # conversation dynamics features (nested: metric → {key: value})
            for metric_name, metric_features in features.items():
                for feature_key, feature_value in metric_features.items():
                    row = {
                        "conversation_id": conversation.id,
                        "metric": metric_name,
                        "feature": normalize_key(feature_key),
                        "value": feature_value
                    }
                    rows.append(row)
        
        rows = pd.DataFrame(rows).set_index("conversation_id")
        rows.to_csv(output_path / "conversation_dynamics_features.csv")