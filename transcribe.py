from deepgram import DeepgramClient, PrerecordedOptions

def transcribe_audio(file_path):
    """
    Transcribes audio with optimized Deepgram API options.
    """
    DEEPGRAM_API_KEY = '7239a79ba362d43f04adeb717be1ec9ea8f2cdcd'
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    try:
        with open(file_path, 'rb') as buffer_data:
            payload = {'buffer': buffer_data}
            options = PrerecordedOptions(
                model="nova",  # Use a simpler model
                language="en-US",
                diarize=True,
                punctuate=True # Disable diarization if not required
            )

            response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
            results = response['results']

            # Extract transcript
            transcript = " ".join(
                [word['word'] for channel in results['channels'] for word in channel['alternatives'][0]['words']]
            )

            return transcript.strip()

    except Exception as e:
        print("Error:", e)
        return ""
    
transcript = transcribe_audio('D:/Faculty_proc_new/Faculty_processing_LMS/audio_file2.mp3')
#print("Transcription:", transcript)

