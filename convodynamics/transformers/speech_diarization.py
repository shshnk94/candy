import torch
import torchaudio
import pandas as pd
from pyannote.audio import Pipeline

from convokit import Corpus
from convokit.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechDiarizationTransformer(Transformer):

    def __init__(
        self, 
        huggingface_token: str,
        diarization_model: str
    ):
        
        self.huggingface_token = huggingface_token
        self.diarization_model = diarization_model

        self.pipeline = Pipeline.from_pretrained(self.diarization_model).to(device)

    def _diarize_conversation(
        self, 
        audio_file: str
    ):

        info = torchaudio.info(audio_file)
        total_duration = info.num_frames / info.sample_rate

        # diarize the waveform
        diarization = self.pipeline(audio_file)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })

        segments = pd.DataFrame(segments)
        return segments, total_duration

    def transform(
        self, 
        corpus: Corpus
    ):
        
        for conversation in corpus.iter_conversations():

            audio_file = conversation.retrieve_meta("audio_file")
            segments, total_duration = self._diarize_conversation(audio_file)
            
            conversation.add_meta("diarization_segments", segments)
            conversation.add_meta("total_duration", total_duration)

        return corpus