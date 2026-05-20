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
        # self.has_survey = True

    def to_convokit(self):

        conversation_folder = list(Path(self.datapath).glob('*'))

        # iterate through conversations
        utterances = []
        for convo_path in conversation_folder:

            convo_id = convo_path.name

            # load utterance transcript
            transcript_path = convo_path / "transcription" / f"transcript_{self.transcript_type}.csv"
            # speaker IDs like "5e7751362226d42b..." look like scientific notation
            # and crash pandas' float inference; force string dtype.
            transcript = pd.read_csv(transcript_path, dtype={"speaker": str})

            # load metadata
            # with open(convo_path / "metadata.json", 'r') as f:
            #     metadata = json.load(f)

            # get speakers for this conversation -- map to ConvoKit Speaker objects
            speakers = {speaker: Speaker(id=speaker, meta={}) for speaker in transcript["speaker"].unique()}

            meta_fields = set(transcript.columns) - {"turn_id", "speaker", "utterance"}
            for _, row in transcript.iterrows():

                utterance = Utterance(
                    id=f"{convo_id}_{row['turn_id']}",
                    speaker=speakers[row["speaker"]],
                    conversation_id=convo_id,
                    reply_to=f"{convo_id}_{row['turn_id'] - 1}" if row["turn_id"] > 0 else None,
                    timestamp=row["start"],
                    text=row["utterance"],
                    meta={k: row[k] for k in meta_fields}
                )

                utterances.append(utterance)

        corpus = Corpus(utterances=utterances)
        # surveys = pd.concat(surveys, ignore_index=True)

        # loading survey data if available, and attaching to speakers and conversations
        speaker_outcomes = ['sex', 'politics', 'race', 'edu', 'employ', 'employ_7_TEXT', 'age']
        employment_outcomes = ['employed', 'unemployed', 'temp_leave', 'disabled', 'retired', 'homemaker', 'other']
        for convo_path in conversation_folder:

            convo_id = convo_path.name

            # load survey data
            survey_path = convo_path / "survey.csv"
            if not survey_path.exists(): continue
            survey = pd.read_csv(survey_path).set_index('user_id')

            conversation = corpus.get_conversation(convo_id)

            # load survey data for this conversation
            conversation_outcomes = list(set(survey.columns) - set(speaker_outcomes) - {'convo_id', 'user_id', "partner_id"})
            conversation_survey_outcomes = survey[conversation_outcomes]
            conversation.meta = conversation_survey_outcomes.to_dict()

            # add audio file path
            audio_file = convo_path / "processed" / f"{convo_id}.mp3"
            conversation.add_meta("audio_file", str(audio_file))
            
            for speaker in conversation.iter_speakers():

                speaker_survey_outcomes = survey.loc[speaker.id]

                # cleaning the "employ" field
                employ = speaker_survey_outcomes["employ"]
                if not pd.isna(employ):
                    speaker_survey_outcomes["employ"] = employment_outcomes[int(employ) - 1]

                speaker.meta = speaker_survey_outcomes.to_dict()

        folder_name = f"candor_{self.transcript_type}"
        corpus.dump(name = folder_name, base_path = self.datapath)

        return folder_name