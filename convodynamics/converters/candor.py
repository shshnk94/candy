from pathlib import Path
import pandas as pd

from convokit import Utterance, Speaker, Corpus

from .base import BaseConverter

class CandorConverter(BaseConverter):

    def __init__(
        self, 
        datapath: str,
        transcript_type: str = None):

        super().__init__(datapath)
        self.transcript_type = transcript_type

    def to_convokit(self):

        conversation_folder = Path(self.datapath).glob('*')
        speakers, surveys, utterances = [], [], []

        # iterate through conversations
        for convo_path in conversation_folder:

            convo_id = convo_path.name

            # load survey data
            survey_path = convo_path / "survey.csv"
            survey = pd.read_csv(survey_path)
            surveys.append(survey)

            # load utterance transcript
            transcript_path = convo_path / "transcription" / f"transcript_{self.transcript_type}.csv"
            transcript = pd.read_csv(transcript_path)

            # load metadata
            # with open(convo_path / "metadata.json", 'r') as f:
            #     metadata = json.load(f)

            # get speakers for this conversation -- map to ConvoKit Speaker objects
            speakers = {speaker: Speaker(id=speaker, meta={}) for speaker in transcript["speaker"].unique()}

            meta_fields = set(transcript.columns) - {"turn_id", "speaker", "start", "utterance"}
            for index, row in transcript.iterrows():

                utterance = Utterance(
                    id=row["turn_id"],
                    speaker=speakers[row["speaker"]],
                    conversation_id=convo_id,
                    reply_to=row["turn_id"] - 1 if row["turn_id"] > 0 else None,
                    timestamp=row["start"],
                    text=row["utterance"],
                    meta={k: row[k] for k in meta_fields}
                )

                utterances.append(utterance)

        corpus = Corpus(utterances=utterances)
        surveys = pd.concat(surveys, ignore_index=True)

        # attach survey metadata to speakers
        speaker_outcomes = ['sex', 'politics', 'race', 'edu', 'employ', 'employ_7_TEXT', 'age']
        for speaker in corpus.iter_speakers():
            survey = surveys[surveys['user_id'] == speaker.id].sort_values(by='date')
            survey = survey.groupby('user_id').agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None)

            # cleaning the "employ_7_TEXT" and "employ" fields
            employment = ['employed', 'unemployed', 'temp_leave', 'disabled', 'retired', 'homemaker', 'other']
            survey["employ"] = survey["employ"].apply(lambda x: employment[int(x) - 1] if not pd.isna(x) else x)

            speaker.meta = survey.to_dict(orient='records')[0]

        # attach survey metadata to conversations
        conversation_outcomes = list(set(surveys.columns) - set(speaker_outcomes) - {'convo_id', 'user_id', "partner_id"})
        for conversation in corpus.iter_conversations():

            convo_path = Path(self.datapath) / conversation.id
            speakers = conversation.get_speaker_ids()

            # load survey data for this conversation
            survey = surveys[surveys['convo_id'] == conversation.id]
            survey = survey.set_index('user_id')
            survey = survey[conversation_outcomes]

            conversation.meta = survey.to_dict()

            # add audio file path
            audio_file = convo_path / "processed" / f"{convo_id}.mp3"
            conversation.add_meta("audio_file", str(audio_file))

        folder_name = f"candor_{self.transcript_type}"
        corpus.dump(
            name = folder_name, 
            base_path = self.datapath)

        return folder_name