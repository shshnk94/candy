# convodynamics

A tool designed to extract interpretable macro-level features from conversation, based on the methodology described by [Di Stasi et al. (2023)](https://psycnet.apa.org/record/2024-16512-001).

## Installation

To install the required dependencies and `convodynamics`, run:

```bash
pip install -r requirements.txt
pip install .
```

Write about Huggingface Hub registerations as well

## Usage

After installation, you can extract macro features from a conversation file:

```bash
python -m convodynamics.macro_metrics --datapath /path/to/conversations
```

Replace `/path/to/conversations` with your conversation data directory.

### Command-line Arguments

- `--datapath`: Path to the folder containing conversation data (required)

## Repository structure

```
.
├── convodynamics/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── feature.py          # Abstract Feature base class
│   ├── macro_metrics.py    # Macro-level feature implementations
│   ├── preprocess.py       # Audio preprocessing and speaker diarization
│   └── utils.py            # Utility functions (adaptability, predictability)
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation configuration
├── README.md              # Project documentation
└── LICENSE                # License file
```

## Features

The package currently supports extraction of the following macro-level conversation features:

- **Speaking Time**: Percentage of total conversation time each speaker talks
- **Turn Length**: Statistical measures of speaking turn durations (median, mean, CV, predictability)
- **Pauses**: Average pause percentages between speaking turns
- **Adaptability**: Cross-speaker behavioral adaptation measures

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References