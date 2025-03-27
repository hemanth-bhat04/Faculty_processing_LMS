from deepgram import DeepgramClient, PrerecordedOptions

def transcribe_audio(file_path):
    """
    Transcribes audio without applying speaker diarization.
    """
    DEEPGRAM_API_KEY = '7239a79ba362d43f04adeb717be1ec9ea8f2cdcd'
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    try:
        with open(file_path, 'rb') as buffer_data:
            payload = {'buffer': buffer_data}
            options = PrerecordedOptions(
                model="nova-3",
                language="en-US"
            )

            response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
            results = response['results']
            transcript = " ".join(
                [word['word'] for channel in results['channels'] for word in channel['alternatives'][0]['words']]
            )
            
            return transcript

    except Exception as e:
        print("Error:", e)
        return ""