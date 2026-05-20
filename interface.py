import argparse
from pathlib import Path

from convokit import Corpus

from candy.converters import CandorConverter
from candy.transformers import ConversationDynamicsTransformer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Candor dataset for conversation dynamics analysis.")
    parser.add_argument("--datapath", type=str, required=True, help="Path to the Candor dataset.")
    parser.add_argument("--transcript_type", type=str, default="audiophile", help="Type of transcript to use (default: audiophile).")
    parser.add_argument("--output_path", type=str, default="results", help="Path to save the transformed corpus.")
    args = parser.parse_args()

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

    # dump the transformed corpus with conversation dynamics features
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    dynamics_extractor.export(corpus, output_path)