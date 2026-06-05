# CANDY

A tool designed to extract interpretable macro-level features from conversation, based on the methodology described by [Di Stasi et al. (2023)](https://psycnet.apa.org/record/2024-16512-001).

## Features

The package currently supports deriving the following conversation-level, speaker-specific metrics:

- **Speaking Time**: Proportion of an interactant’s speaking time relative to the entire conversation.
- **Turn Length**: Duration of an interactant’s speech turns.
- **Speech Rate**: Speed at which an interactant talks in words per minute (WPM), excluding within-turn pauses.
- **Backchannels**: Instances of sub-1-s utterances during the other interactant’s turn.
- **Response Time**: Duration of silence between the end of one interactant’s turn and the first voiced utterance from the other interactant.
- **Pause**: Instances of silence within an interactant’s speech turns that last at least 180 ms, weighted by total speaking time.

For speaking time, pause, and backchannels, we calculate the central tendency for each interactant. For turn length, speech rate, and response time, we calculate four measures for each conversation dynamics metric: median, coefficient of variation, adaptability, and predictability.

## R

### Components

- CANDY_function.R are the packaged functions, ready to use with instructions and assumptions below.  
- CANDY_tutorials.Rmd & CANDY_tutorials.html are notebook files detailing the calculation of each metric for references and customization.
- CANDY_batch_process.R is a demonstration of using CANDY_function.R to calculate all conversation dynamics metrics of over 1,600 conversations from Conversation: A Naturalistic Dataset of Online Recordings (CANDOR).
- CANDY_CANDOR_demo.Rmd is the demontration file for analyzing how various conversation dynamics metrics are related to individual differences in the Big Five personality traits and pre-to-post-conversation changes in affect valence and arousal using CANDOR.

### Assumptions

Input data.frame has at least: speaker, start, stop (row = turn).  

Optional columns:  
- n_words   (for speech-rate metrics; see helper in .R to compute from text)  
- overlap   (logical / 0-1 / "True"/"False"); see helper in .R to add if missing  

### Output

  Each metric function returns a tibble with one row per `speaker`.  
  A convenience wrapper `cd_all_metrics()` returns a named list of results.

### Usage
```
  source("ConvoDynamics.R")  
  res <- cd_all_metrics(audio)

  res$speaking_time; res$turn_length;   
  res$speech_rate; res$backchannel; res$response_time
```

## Python

### Installation

This project is managed with [uv](https://docs.astral.sh/uv/). To install `candy` and its dependencies into a virtual environment, run:

```bash
uv sync
```

Alternatively, install into an existing environment with pip:

```bash
pip install -e .
```

#### Hugging Face Hub

The transcription pipeline (`scripts/transcribe.py`) downloads WhisperX models from the Hugging Face Hub. Create a free account at [huggingface.co](https://huggingface.co), generate an access token, and place it in a `.env` file at the repository root:

```
HUGGINGFACE_TOKEN="hf_..."
```

### Usage

Extract conversation-dynamics features from a dataset of conversations:

```bash
python scripts/interface.py --datapath /path/to/conversations
```

To (re)generate transcripts from raw audio with WhisperX and forced alignment first:

```bash
python scripts/transcribe.py --datapath /path/to/conversations
```

Replace `/path/to/conversations` with your conversation data directory.

### Command-line Arguments

`scripts/interface.py`
- `--datapath`: Path to the dataset of conversations (required)
- `--transcript_type`: Type of transcript to use (default: `audiophile`)
- `--output_path`: Directory to save the transformed corpus (default: `results`)

`scripts/transcribe.py`
- `--datapath`: Path to the conversations directory (required)
- `--model`: WhisperX model size (default: `large-v2`)
- `--language`: Language code for transcription (default: `en`)
- `--batch_size`: Batch size for transcription (default: `16`)

### Repository structure

```
.
├── candy/                  # Main package
│   ├── __init__.py
│   ├── aggregation.py      # Aggregate word-level transcripts into turns (audiophile/backbiter)
│   ├── converters/         # Convert raw datasets (e.g. CANDOR) to ConvoKit corpora
│   ├── transformers/       # ConvoKit transformers, incl. conversation-dynamics metrics
│   └── associators/        # Relate metrics to conversation outcomes
├── scripts/                # Standalone CLI scripts (interface.py, transcribe.py, helpers)
├── notebooks/              # Exploratory analysis (analysis.ipynb)
├── data/                   # Input datasets (gitignored)
├── results/                # Extracted features and outputs
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Pinned dependency lockfile
├── README.md               # Project documentation
└── LICENSE                 # License file
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

Di Stasi, M., Templeton, E., & Quoidbach, J. (2024). Zooming out on bargaining tables: Exploring which conversation dynamics predict negotiation outcomes. *Journal of Applied Psychology*, *109*(7), 1077–1093. https://doi.org/10.1037/apl0001136
