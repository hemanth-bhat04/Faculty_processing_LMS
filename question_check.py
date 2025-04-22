import re
from deepgram import DeepgramClient, PrerecordedOptions

class AudioIntelligence:
    def __init__(self, input_data):
        if isinstance(input_data, str) and input_data.endswith(('.mp3', '.wav', '.m4a')):
            self.file_path = input_data
            self.speech_text, self.speaker_labels = self.speech_to_text_with_diarization()
        else:
            self.speech_text = input_data
            self.speaker_labels = self.process_text_transcript(input_data)
        
        # Ensure question-answer analysis runs on both audio and text inputs
        self.trainer_questions, self.student_questions, self.trainer_answers, self.student_answers, self.detailed_answers, self.unique_students = self.analyze_questions_and_answers()
    
    def is_question(self, sentence: str) -> bool:
        """Determines if a given sentence is a question."""
        question_words = ["who", "what", "where", "when", "why", "how", "do", "does", "did", "is", "are", "was", "were", "can", "could", "should", "would"]
        return sentence.strip().endswith("?") or any(sentence.lower().startswith(q + " ") for q in question_words)
    
    def is_answer(self, question: str, response: str) -> bool:
        """Determines if a response is a valid answer to a question."""
        if not response:
            return False  # Ignore empty responses
        
        # Allow short responses like 'Yes', 'No' if they logically follow a question
        short_responses = {"yes", "no", "maybe", "okay", "sure", "right", "correct"}
        if response.lower() in short_responses:
            return True
        
        question_keywords = set(question.lower().split())
        response_keywords = set(response.lower().split())  
        common_words = question_keywords.intersection(response_keywords)
        
        informative_starts = ["some common", "one way is", "examples include", "you can use", "it depends on", "the answer is", "typically", "usually", "it involves", "it consists of"]
        if any(response.lower().startswith(start) for start in informative_starts):
            return True
        
        return len(common_words) > 1 or len(response.split()) > 3  # More than 1 shared word or a longer response

    def speech_to_text_with_diarization(self) -> tuple:
        """
        Transcribes audio and applies speaker diarization.
        """
        DEEPGRAM_API_KEY = '7239a79ba362d43f04adeb717be1ec9ea8f2cdcd'
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        try:
            with open(self.file_path, 'rb') as buffer_data:
                payload = {'buffer': buffer_data}
                options = PrerecordedOptions(
                    model="nova-3",
                    language="en-US",
                    diarize=True     
                )

                response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
                results = response['results']
                speaker_sentences = {}

                for channel in results['channels']:
                    current_speaker = None
                    current_sentence = []

                    for word in channel['alternatives'][0]['words']:
                        speaker = word.get('speaker', 'Unknown') if isinstance(word, dict) else getattr(word, 'speaker', 'Unknown')
                        text = word['word']

                        if speaker != current_speaker:
                            if current_speaker is not None:
                                speaker_sentences[current_speaker] = speaker_sentences.get(current_speaker, []) + [
                                    " ".join(current_sentence)]
                            current_speaker = speaker
                            current_sentence = [text]
                        else:
                            current_sentence.append(text)

                    if current_sentence:
                        speaker_sentences[current_speaker] = speaker_sentences.get(current_speaker, []) + [
                            " ".join(current_sentence)]

                formatted_transcript = "\n".join(
                    [f"Speaker {speaker}: {' '.join(sentences)}" for speaker, sentences in speaker_sentences.items()]
                )

                return formatted_transcript, speaker_sentences

        except Exception as e:
            print("Error:", e)
            return "", {}
    
    def process_text_transcript(self, transcript: str):
        """Processes a manually provided text transcript and extracts speaker labels."""
        speaker_sentences = {}
        lines = transcript.split("\n")
        for line in lines:
            if line.startswith("Speaker"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    speaker, sentence = parts[0].strip(), parts[1].strip()
                    speaker_sentences[speaker] = speaker_sentences.get(speaker, []) + [sentence]
        return speaker_sentences
    
    def analyze_questions_and_answers(self):
        """Analyzes questions and answers in the transcript."""
        trainer_questions = 0
        student_questions = 0
        trainer_answers = 0
        student_answers = 0
        detailed_answers = {}
        unique_students = set()
        last_question = None
        last_speaker = None

        for speaker, sentences in self.speaker_labels.items():
            for sentence in sentences:
                if self.is_question(sentence):
                    if speaker == "Speaker 0":  # Trainer
                        trainer_questions += 1
                    else:
                        student_questions += 1
                        unique_students.add(speaker)  # Track unique student speakers
                    last_question = sentence
                    last_speaker = speaker
                elif last_question and speaker != last_speaker and self.is_answer(last_question, sentence):
                    if speaker == "Speaker 0":
                        trainer_answers += 1
                    else:
                        student_answers += 1
                    detailed_answers[(last_question, last_speaker)] = (sentence, speaker)  # Store Q&A pairs with speaker info
                    last_question = None  # Reset

        return trainer_questions, student_questions, trainer_answers, student_answers, detailed_answers, len(unique_students)


def analyze_classroom_audio(input_data):
    intelligence = AudioIntelligence(input_data)

    return {
        "transcript": intelligence.speech_text,
        "speaker_labels": intelligence.speaker_labels,
        "trainer_questions": intelligence.trainer_questions,
        "student_questions": intelligence.student_questions,
        "trainer_answers": intelligence.trainer_answers,
        "student_answers": intelligence.student_answers,
        "detailed_answers": intelligence.detailed_answers,
        "unique_students": intelligence.unique_students  # Add unique students count
    }

if __name__ == '__main__':
    result = analyze_classroom_audio(
        input_data='classroom_audio.mp3'  # Can also pass raw transcript text
    )
    
    print(result['transcript'])
    print("Trainer Questions:", result['trainer_questions'])
    print("Student Questions:", result['student_questions'])
    print("Trainer Answers:", result['trainer_answers'])
    print("Student Answers:", result['student_answers'])
    print("Detailed Answers:", result['detailed_answers'])
    print("Unique Students Participated:", result['unique_students'])