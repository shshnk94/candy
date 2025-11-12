import os
from pathlib import Path
from dotenv import load_dotenv

from convokit import Corpus
from dataclasses import dataclass

from convodynamics.converters import CandorConverter
from convodynamics.transformers import (
    SpeechDiarizationTransformer, 
    ConversationDynamicsTransformer
)

@dataclass
class Args:
    datapath: str
    transcript_type: str = None
if __name__ == "__main__":


    # all constants -- TBD later
    args = Args(
        datapath="data/conversations",
        transcript_type="audiophile"
    )

    # convert the Candor dataset to ConvoKit format
    converter = CandorConverter(
        datapath=args.datapath,
        transcript_type=args.transcript_type
    )

    folder_name = converter.to_convokit()
    print("Converted dataset to ConvoKit format...")

    # load the converted corpus
    corpus = Corpus(filename=Path(args.datapath) / folder_name)
    print(folder_name)

    # diarize audio if available
    load_dotenv()
    diarizer = SpeechDiarizationTransformer(
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
        diarization_model="pyannote/speaker-diarization-3.1"
    )
    corpus = diarizer.transform(corpus)

    # extract conversation dynamics features from Di Stasi et al (2024)
    dynamics_extractor = ConversationDynamicsTransformer()
    dynamics_extractor.register_metrics([
        "speaking_time",
        "turn_length",
        "pauses",
        "speaker_rate",
        "backchannels",
        "response_time"
    ])
    corpus = dynamics_extractor.transform(corpus)

    # random conversation to check features
    print(corpus.random_conversation().retrieve_meta("conversation_dynamics_features"))
