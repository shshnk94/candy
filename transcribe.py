import argparse
import gc
import pandas as pd
import logging
import sys
from pathlib import Path

import torch
import whisperx

from candy.aggregation import aggregate_audiophile_turns, aggregate_backbiter_turns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def get_conversations(datapath: Path) -> list[str]:
    """
    Discover all conversation IDs in the data path.
    Conversation IDs are the folder names directly under datapath.
    Returns a list of conversation ID strings.
    """
    datapath = Path(datapath)
    conversation_ids = [item.name for item in datapath.iterdir() if item.is_dir()]
    return conversation_ids

def load_speaker_audio(convo_dir: Path) -> list[tuple[str, Path]]:
    """
    Build (user_id, audio_path) tuples from survey.csv rows.

    Each speaker audio is expected at raw/[speaker_id].mp4.
    Returns a list where each tuple is (user_id, path_to_audio_file).
    """

    survey_path = convo_dir / "survey.csv"
    if not survey_path.exists():
        logger.error(f"{survey_path} not found")
        return []
    
    survey = pd.read_csv(survey_path)

    # get unique speaker IDs from survey, dropping any NaN values
    speakers = survey["user_id"].dropna().unique()

    audios = []
    for speaker in speakers:
        audio_path = convo_dir / "processed" / f"{speaker}.mp4"
        if audio_path.exists():
            audios.append((speaker, audio_path))
        else:
            logger.warning(f"Audio file not found for speaker {speaker}: {audio_path}")

    return audios

def transcribe_and_align(
    audio_path: Path,
    user_id: str,
    model,
    align_model,
    align_metadata,
    device: str,
    batch_size: int,
    language: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transcribe an audio file for a single speaker, run forced alignment.

        Args:
        audio_path: Path to speaker's audio file (MP3, MKV, WAV, etc.)
        user_id: CANDOR user_id for this speaker
        model: WhisperX model
        align_model: Alignment model
        align_metadata: Alignment metadata
        device: 'cuda' or 'cpu'
        batch_size: Batch size for transcription
        language: Language code (e.g., 'en')

        Returns a tuple of:
            - word_rows: word-level DataFrame with 'speaker' = user_id
            - segments: segment-level DataFrame with 'speaker' = user_id
    """

    # load audio and transcribe with WhisperX
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # forced alignment
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    allwords = []
    segments = aligned["segments"]
    for idx, segment in enumerate(segments):

        segment["speaker"] = user_id
        start, end, text = segment["start"], segment["end"], segment["text"].strip()

        words = segment["words"]
        if words:
            for word_info in words:
                allwords.append(
                    {
                        "segment_id": idx,
                        "segment_start": start,
                        "segment_end": end,
                        "segment_text": text,
                        "word": word_info["word"],
                        "word_start": word_info["start"],
                        "word_end": word_info["end"],
                        "word_score": word_info["score"],
                        "speaker": user_id,
                        "language": language,
                    }
                )
        else:
            allwords.append(
                {
                    "segment_id": idx,
                    "segment_start": start,
                    "segment_end": end,
                    "segment_text": text,
                    "word": "",
                    "word_start": "",
                    "word_end": "",
                    "word_score": "",
                    "speaker": user_id,
                    "language": language,
                }
            )

    allwords = pd.DataFrame(allwords)
    return allwords

def write_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write a pandas DataFrame to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

def main():

    parser = argparse.ArgumentParser(description="Transcribe conversation MP3s with WhisperX + forced alignment.")
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the conversations directory (e.g. data/conversations).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        help="WhisperX model size (default: large-v2). Options: tiny, base, small, medium, large-v2, large-v3.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for transcription (default: 'en' for English).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16). Lower if running out of GPU memory.",
    )
    args = parser.parse_args()

    # get conversation IDs from data path
    datapath = Path(args.datapath)
    conversations = get_conversations(datapath)

    # load WhisperX model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading WhisperX model '{args.model}' on {device}")
    model = whisperx.load_model(
        args.model, 
        language=args.language, 
        device=device
    )

    # load alignment model
    logger.info(f"Loading alignment model")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language,
        device=device
    )

    # iterate over conversations and transcribe each speaker separately
    for idx, convo_id in enumerate(conversations):

        convo_dir = datapath / convo_id
        logger.info(f"[{idx}/{len(conversations)}] Transcribing {convo_id}")

        try:
            # Load speaker audio files from survey speaker IDs.
            speaker_audio = load_speaker_audio(convo_dir)
            if not speaker_audio:
                logger.error(f"No speaker audio files found in {convo_dir}")
                continue

            # Transcribe each speaker's audio separately
            allwords = []
            for user_id, audio_path in speaker_audio:
                logger.info(f"  Transcribing speaker {user_id}: {audio_path.name}")
                words = transcribe_and_align(
                    audio_path=audio_path,
                    user_id=user_id,
                    model=model,
                    align_model=align_model,
                    align_metadata=align_metadata,
                    device=device,
                    batch_size=args.batch_size,
                    language=args.language,
                )

                allwords.append(words)

            allwords = pd.concat(allwords, ignore_index=True)

            # word-level CSV (one row per word with WhisperX alignment timings)
            write_to_csv(allwords, convo_dir / "transcription" / "transcript_whisper_words.csv")
            logger.info(f"Wrote {len(allwords)} words to {convo_dir / 'transcription' / 'transcript_whisper_words.csv'}")

            # turn-level CSVs (CANDOR audiophile + backbiter formats)
            audiophile_turns = aggregate_audiophile_turns(allwords)
            audiophile_path = convo_dir / "transcription" / "transcript_whisper_audiophile.csv"
            write_to_csv(audiophile_turns, audiophile_path)
            logger.info(f"Wrote {len(audiophile_turns)} audiophile turns to {audiophile_path}")

            backbiter_turns = aggregate_backbiter_turns(allwords)
            backbiter_path = convo_dir / "transcription" / "transcript_whisper_backbiter.csv"
            write_to_csv(backbiter_turns, backbiter_path)
            logger.info(f"Wrote {len(backbiter_turns)} backbiter turns to {backbiter_path}")
            
        except Exception:
            logger.exception(f"Failed to transcribe {convo_id}")

        # free intermediate GPU memory between conversations
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()