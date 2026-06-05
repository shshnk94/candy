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

- .R file for packaged functions, ready to use with instructions and assumptions below.  
- .Rmd notebook file (.html version as well) detailing the calculation of each metric for references and customization.
- .Rmd file demonstrating the use of the tool by analyzing Conversation: A Naturalistic Dataset of Online Recordings (CANDOR).
 

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

To install the required dependencies and `candy`, run:

```bash
pip install -r requirements.txt
pip install .
```

Write about Huggingface Hub registerations as well

### Usage

After installation, you can extract macro features from a conversation file:

```bash
python -m candy.macro_metrics --datapath /path/to/conversations
```

Replace `/path/to/conversations` with your conversation data directory.

### Command-line Arguments

- `--datapath`: Path to the folder containing conversation data (required)

### Repository structure

```
.
├── candy/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── feature.py          # Abstract Feature base class
│   ├── macro_metrics.py    # Macro-level feature implementations
│   └── utils.py            # Utility functions (adaptability, predictability)
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation configuration
├── README.md              # Project documentation
└── LICENSE                # License file
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

Di Stasi, M., Templeton, E., & Quoidbach, J. (2024). Zooming out on bargaining tables: Exploring which conversation dynamics predict negotiation outcomes. *Journal of Applied Psychology*, *109*(7), 1077–1093. https://doi.org/10.1037/apl0001136
