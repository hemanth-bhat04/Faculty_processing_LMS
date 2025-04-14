import json
import soundfile as sf
from nlp_keywords import get_weighted_queries
from smatch import semantic_smart_answer
from transcribe import transcribe_audio  # Use the actual transcribe function
import requests
import boto3
from transformers import pipeline

from botocore.client import Config
from queue import Queue
from fetch_keywords import fetch_keywords
s3 = boto3.client('s3',
                  aws_access_key_id="AKIA43Y7V2OHU52XAOHQ",
                  aws_secret_access_key="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",
                  config=Config(signature_version='s3v4', s3={'use_accelerate_endpoint': True}),
                  region_name="ap-south-1")


# Google Drive file download function 
def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response_object):
        for key, value in response_object.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response_object):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response_object.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response)


# S3 upload function 
def upload_to_aws(local_file, s3_file, bucket="copyrvswaroop"):
    try:
        s3.upload_file(local_file, bucket, s3_file)
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False

def generate_s3_link(bucket, s3_file, region):
    s3_link = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_file}"
    return s3_link


# Fetch the audio file from Google Drive 
google_drive_file_id = "1PaoqrhwPMB61DSdAc8jRMm_92HaORS9C"  
local_audio_file_path = "audio_file1.mp3"
download_file_from_google_drive(google_drive_file_id, local_audio_file_path)

def split_audio_into_chunks(file_path, chunk_duration=300):
    """
    Splits the audio into chunks of specified duration (default: 5 minutes = 300 seconds).
    Using soundfile to split the audio with proper error handling.
    """
    try:
        # Create output directory if it doesn't exist
        import os
        output_dir = "audio_chunks"
        os.makedirs(output_dir, exist_ok=True)

        print("\n=== Processing Audio File ===")
        print(f"Input file: {file_path}")

        # Read the audio file
        audio, sample_rate = sf.read(file_path)
        chunk_size_samples = int(chunk_duration * sample_rate)
        chunks = []

        # Calculate total chunks needed
        total_chunks = (len(audio) + chunk_size_samples - 1) // chunk_size_samples
        print(f"Total chunks to process: {total_chunks}")

        for i in range(total_chunks):
            try:
                # Calculate chunk boundaries
                start_idx = i * chunk_size_samples
                end_idx = min(start_idx + chunk_size_samples, len(audio))
                
                # Extract chunk using proper slicing
                chunk = audio[start_idx:end_idx]
                
                # Create chunk path and save
                chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
                sf.write(chunk_path, chunk, sample_rate)
                chunks.append(chunk_path)

                # Upload to S3 and get link
                s3_file_key = f"audio_chunks/{os.path.basename(chunk_path)}"
                upload_success = upload_to_aws(chunk_path, s3_file_key)
                if upload_success:
                    s3_link = generate_s3_link("copyrvswaroop", s3_file_key, "ap-south-1")
                    print(f"Processed chunk {i+1}/{total_chunks}:")
                    print(f"  - File: {chunk_path}")
                    print(f"  - S3 Link: {s3_link}\n")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}/{total_chunks}: {str(e)}")
                continue
        
        print("=== Audio Processing Complete ===\n")
        return chunks

    except Exception as e:
        print(f"Error in split_audio_into_chunks: {str(e)}")
        return []

def normalize_keyword(keyword: str) -> str:
    """Normalize the keyword by lowercasing and removing punctuation."""
    import re
    return re.sub(r'[^\w\s]', '', keyword.lower()).strip()

def count_questions_in_transcript(transcript: str) -> int:
    """
    Counts the number of questions in the transcript using simple regex patterns.
    A question is identified by:
    1. Presence of a question mark
    2. Starting with common question words
    """
    import re
    
    # Split into sentences
    sentences = re.split('[.!?]', transcript)
    question_count = 0
    
    # Common question patterns
    question_patterns = [
        r'\?$',  # Ends with question mark
        r'^(what|why|how|when|where|who|which|whose|do|does|did|is|are|can|could|would|will|should)\b.*',  # Starts with question words
    ]
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if not sentence:
            continue
            
        # Check if sentence matches any question pattern
        for pattern in question_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                question_count += 1
                break
    
    return question_count

# Load a pre-trained pipeline for text classification
question_classifier = pipeline("text-classification", model="distilbert-base-uncased")

def count_questions_with_transformers(transcript: str) -> int:
    """
    Counts the number of questions in the transcript using Hugging Face Transformers.
    A question is identified based on the model's classification.
    """
    sentences = transcript.split(".")  # Split transcript into sentences
    question_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ensure the sentence length is within the model's token limit
        if len(sentence) > 512:
            print(f"Skipping long sentence: {sentence[:50]}... (length: {len(sentence)})")
            continue

        # Use the classifier to predict if the sentence is a question
        result = question_classifier(sentence)
        if "?" in sentence or result[0]["label"] == "QUESTION":  # Adjust label based on the model
            question_count += 1

    return question_count

def process_audio_chunks(file_path):
    chunks = split_audio_into_chunks(file_path)
    if not chunks:
        print("Error: No audio chunks were created")
        return set(), 0  # Return empty set for keywords and 0 for questions
    
    transcript_queue = Queue()
    
    for chunk_path in chunks:
        transcript = transcribe_audio(chunk_path)
        if transcript:
            transcript_queue.put(transcript)  # Direct transcript without correction
        else:
            transcript_queue.put("")

    processed_transcripts = []
    temp_transcript = ""
    
    while not transcript_queue.empty():
        chunk_transcript = transcript_queue.get()
        temp_transcript += " " + chunk_transcript.strip()
        
        if len(temp_transcript.split()) > 10:
            processed_transcripts.append(temp_transcript.strip())
            temp_transcript = ""
    
    if temp_transcript.strip():
        processed_transcripts.append(temp_transcript.strip())
    
    extracted_keywords = set()
    total_questions = 0  # Initialize question count
    
    for transcript in processed_transcripts:
        subject = "computer science"
        level = "computer science"
        _, phrasescorelist, _, _ = get_weighted_queries(transcript, len(transcript), subject, level)
        extracted_keywords.update(normalize_keyword(kw[0]) for kw in phrasescorelist)
        
        # Use the simpler question counting method
        total_questions += count_questions_in_transcript(transcript)
    
    print(f"Total Questions Asked in Class: {total_questions}")
    return extracted_keywords, total_questions


# Process the audio file and extract keywords
corrected_transcript_keywords, total_questions = process_audio_chunks("audio_file.mp3")

if corrected_transcript_keywords:  # Only proceed if we have keywords
    # Fetching keywords from fetch_keywords.py (silently)
    hardcoded_keywords = fetch_keywords('Oy4duAOGdWQ')
    flat_keywords = [str(keyword) for sublist in hardcoded_keywords for keyword in sublist]

    # Semantic Matching
    semantic_result = semantic_smart_answer(
        student_answer=" ".join(corrected_transcript_keywords),
        question=(
            "Analyze the transcript and compare it with the expected lecture topics. "
            "Focus on key concepts rather than exact word matches. "
            "Recognize synonyms, paraphrasing, and related terminology. "
            "Ensure that essential topics are covered, and penalize missing core concepts while allowing variations in phrasing. "
            "If similar words or phrases convey the same meaning, consider them a match. "
            "Provide a semantic similarity score based on concept coverage, relevance, and accuracy."
        ),
        answer=" ".join(flat_keywords),
        details=1
    )

    # Debugging: Print the inputs to the API
    print("\nDebug: Inputs to Semantic Matching API:")
    print(f"Student Answer: {corrected_transcript_keywords}")
    print(f"Answer: {flat_keywords}")

    # Debugging: Print the raw semantic_result
    print("\nDebug: Raw Semantic Result:")
    print(semantic_result)

    # Parse and display the results
    try:
        # Check if semantic_result is a string and needs parsing
        if isinstance(semantic_result, str):
            response = json.loads(semantic_result)  # Parse the semantic_result into a dictionary
        else:
            response = semantic_result  # Use it directly if it's already a dictionary

        # Parse the nested JSON string inside the "content" field
        if isinstance(response['content'], str):
            try:
                # Extract the JSON part from the content field
                content_start = response['content'].find("{")
                if content_start != -1:
                    response['content'] = json.loads(response['content'][content_start:])
                else:
                    raise ValueError("No JSON object found in 'content' field")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error decoding nested JSON in 'content': {e}")
                print(f"Raw 'content' field: {response['content']}")
                response['content'] = {
                    "answer_match": "0%",
                    "missing_concepts": [],
                    "additional_concepts": [],
                    "reasons": "Error parsing the 'content' field."
                }

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw semantic_result: {semantic_result}")
        response = {"content": {"answer_match": "0%", "missing_concepts": [], "additional_concepts": [], "reasons": ""}}

    # Debugging: Print the parsed response
    print("\nDebug: Parsed Response from API:")
    print(json.dumps(response, indent=4))  # Pretty print the response

    # Extract details from the response
    try:
        answer_match = response['content'].get('answer_match', "0%")
        missing_concepts = response['content'].get('missing_concepts', [])
        additional_concepts = response['content'].get('additional_concepts', [])
        reasons = response['content'].get('reasons', "")
    except (KeyError, IndexError, AttributeError) as e:
        print(f"Error extracting details from response: {e}")
        answer_match = "0%"
        missing_concepts = []
        additional_concepts = []
        reasons = ""

    print("\n=== Lecture Analysis ===")
    print(f"Answer Match: {answer_match}")
    print("Missing Keywords:")
    for idx, keyword in enumerate(missing_concepts, 1):
        print(f"{idx}. {keyword}")

    print("\nAdditional Concepts:")
    for idx, concept in enumerate(additional_concepts, 1):
        print(f"{idx}. {concept}")

    print("\nReasons:")
    print(reasons)

    print(f"\nTotal Questions Asked: {total_questions}")
    print("=====================")

else:
    print("Error: Could not process audio file")



