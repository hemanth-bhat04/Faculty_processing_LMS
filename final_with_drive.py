import json
import soundfile as sf
from nlp_keywords import get_weighted_queries
from smatch import semantic_smart_answer
from transcribe import transcribe_audio  # Use the actual transcribe function
import requests
import boto3
#from transformers import pipeline

from botocore.client import Config
from queue import Queue
from fetch_keywords import fetch_keywords, fetch_all_keywords
from question_check import analyze_classroom_audio
s3 = boto3.client('s3',
                  aws_access_key_id="AKIA43Y7V2OHU52XAOHQ",
                  aws_secret_access_key="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",
                  config=Config(signature_version='s3v4', s3={'use_accelerate_endpoint': True}),
                  region_name="ap-south-1")


# Google Drive file download function 
import os
import requests

def download_file_from_google_drive(file_id, destination):
    # Delete the existing file if it exists
    if os.path.exists(destination):
        os.remove(destination)
        print(f"Deleted existing file: {destination}")

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
    print(f"Downloaded file to {destination}")


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
google_drive_file_id = "1e11nDLwLFr5hVWlMyPmjYWFg4s0mgYrt"  
local_audio_local_audio_file_path = "audio_file2.mp3"
print(f"Downloading new file with ID: {google_drive_file_id}")
download_file_from_google_drive(google_drive_file_id, local_audio_local_audio_file_path)

def split_audio_into_chunks(local_audio_file_path, chunk_duration=240):
    """
    Splits the audio into chunks of specified duration (default: 4 minutes = 240 seconds).
    Using soundfile to split the audio with proper error handling.
    """
    try:
        # Create output directory if it doesn't exist
        import os
        output_dir = "audio_chunks"
        os.makedirs(output_dir, exist_ok=True)

        print("\n=== Processing Audio File ===")
        print(f"Input file: {local_audio_file_path}")

        # Read the audio file
        audio, sample_rate = sf.read(local_audio_file_path)
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
        #r'\?$',  # Ends with question mark --double check if needed or not
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

# Commenting out the pipeline initialization
# question_classifier = pipeline("text-classification", model="distilbert-base-uncased")

# Commenting out the transformer-based question counting function
# def count_questions_with_transformers(transcript: str) -> int:
#     sentences = transcript.split(".")
#     question_count = 0
#     batch_size = 16  # Adjust based on available memory
#     batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

#     for batch in batches:
#         try:
#             results = question_classifier(batch)
#             for sentence, result in zip(batch, results):
#                 if "?" in sentence or result["label"] == "QUESTION":
#                     question_count += 1
#         except Exception as e:
#             print(f"Error processing batch: {e}")
#     return question_count

def process_audio_chunks(local_audio_file_path):
    # Split the audio file into smaller chunks
    chunks = split_audio_into_chunks(local_audio_file_path)
    print(f"Chunks created: {chunks}")
    if not chunks:
        print("Error: No audio chunks were created")
        return [], 0, set(), set(), []

    transcript_queue = Queue()

    # Process each chunk individually
    for chunk_path in chunks:
        transcript = transcribe_audio(chunk_path)
        #print(f"Transcript for {chunk_path}: {transcript}")
        if transcript:
            transcript_queue.put(transcript)
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

    # Combine all transcripts into a single transcript
    complete_transcript = " ".join(processed_transcripts)
    #print(f"\nDEBUG - Complete Transcript: {complete_transcript}")
    #print(f"DEBUG - Complete Transcript Length: {len(complete_transcript)}")
    if len(complete_transcript) < 100:
        print("WARNING: Transcript might be empty or too short")

    # Use regex-based question counting
    question_count = count_questions_in_transcript(complete_transcript)
    print(f"DEBUG - Total Questions Detected: {question_count}")

    # Placeholder for missed keywords
    primary_missed_keywords = set()
    secondary_missed_keywords = set()
    top_10_secondary_missed_keywords = []

    return processed_transcripts, question_count, primary_missed_keywords, secondary_missed_keywords, top_10_secondary_missed_keywords

def compare_keywords(critical_all_keywords, dynamic_critical_keywords):
    """
    Compare two lists of keywords and return common, missing, and unique keywords.
    """
    # Convert lists to sets for efficient comparison
    critical_all_set = set(critical_all_keywords)
    dynamic_critical_set = set(dynamic_critical_keywords)

    # Find common keywords
    common_keywords = critical_all_set & dynamic_critical_set

    # Find keywords missing in dynamic_critical_keywords
    missing_in_dynamic = critical_all_set - dynamic_critical_set

    # Find keywords unique to dynamic_critical_keywords
    unique_to_dynamic = dynamic_critical_set - critical_all_set

    # Return the results as a dictionary
    return {
        "common_keywords": sorted(common_keywords),
        "missing_in_dynamic": sorted(missing_in_dynamic),
        "unique_to_dynamic": sorted(unique_to_dynamic),
    }

def fetch_and_unionize_keywords(video_ids):
    """
    Fetch dynamic critical keywords for multiple video IDs and unionize them.
    """
    all_keywords = set()
    fetched_keywords = {}  # Cache to store fetched keywords for each video ID

    for video_id in video_ids:
        if video_id in fetched_keywords:
            # Use cached keywords if already fetched
            dynamic_keywords = fetched_keywords[video_id]
        else:
            try:
                # Fetch keywords for the video ID
                dynamic_keywords = fetch_all_keywords(video_id)
                fetched_keywords[video_id] = dynamic_keywords  # Cache the result
                print(f"Fetched keywords for video ID {video_id}: {dynamic_keywords}")
            except Exception as e:
                print(f"Error fetching keywords for video ID {video_id}: {e}")
                dynamic_keywords = []

        # Add to the union set
        all_keywords.update(dynamic_keywords)

    return sorted(all_keywords)  # Return sorted unionized keywords

'''def is_question(sentence: str) -> bool:
    """
    Determines if a given sentence is a question.
    """
    question_words = ["who", "what", "where", "when", "why", "how", "do", "does", "did", "is", "are", "was", "were", "can", "could", "should", "would"]
    return sentence.strip().endswith("?") or any(sentence.lower().startsWith(q + " ") for q in question_words)'''

'''def process_transcript_for_questions(transcript):
    """
    Process the transcript to count the number of questions.
    """
    sentences = transcript.split(".")
    question_count = sum(1 for sentence in sentences if is_question(sentence))
    return question_count'''

# Main processing logic
video_ids = ['Oy4duAOGdWQ', 'P2PMgnQSHYQ', 'efR1C6CvhmE']  # Example video IDs
print(f"Processing video IDs: {video_ids}")

# Fetch and unionize keywords for all video IDs
unionized_keywords = fetch_and_unionize_keywords(video_ids)
print(f"Unionized Keywords from All Videos: {unionized_keywords}")

# Process the audio file and extract keywords
processed_transcripts, total_questions, primary_missed_keywords, secondary_missed_keywords, top_10_secondary_missed_keywords = process_audio_chunks(local_audio_local_audio_file_path)

# Combine all processed transcripts into a single transcript
complete_transcript = " ".join(processed_transcripts)

if processed_transcripts:  # Only proceed if we have transcripts
    # Clear previous keywords
    critical_keywords = unionized_keywords  # Use the unionized keywords from all videos
    flat_keywords = []

    # Fetch hardcoded keywords dynamically based on the current audio file or transcript
    try:
        hardcoded_keywords = fetch_keywords('Oy4duAOGdWQ')  # Replace with dynamic input if needed
        flat_keywords = [str(keyword) for sublist in hardcoded_keywords for keyword in sublist]
        print(f"Flat Keywords from 5-Minute Segments: {flat_keywords}")
    except Exception as e:
        print(f"Error fetching hardcoded keywords: {e}")
        flat_keywords = []

    # Combine unionized keywords and flat keywords for semantic matching
    combined_keywords = sorted(set(critical_keywords + flat_keywords))
    print(f"Combined Keywords for Semantic Matching: {combined_keywords}")

    # Compare Critical All Keywords and Dynamic Critical Keywords
    comparison_results = compare_keywords(flat_keywords, critical_keywords)

    # Semantic Matching
    semantic_result = semantic_smart_answer(
        student_answer=" ".join(processed_transcripts),
        question=(
            "Evaluate the semantic similarity between the student's answer and the correct answer, focusing on core conceptual alignment. "
            "Allow for rephrased, synonymous, or paraphrased ideas to count as matching, and disregard differences in structure, grammar, or formatting. "
            "Only include **critical concepts** that are central to understanding the topic in the list of missing concepts. "
            "Do not list superficial or highly specific terms as missing if the overall concept is conveyed. "
            "Also, include additional concepts only if they introduce **new technical ideas** not present in the correct answer. "
            "Avoid penalizing extra elaboration if it supports the main idea. "
            "Ensure the output is deterministic — consistent results should be produced for the same input. "
            "Provide a semantic similarity score that reflects conceptual understanding — not word-level overlap. "
            "Output in the following JSON format: "
            "{"
            "\"answer_match\": \"<percentage value>\", "
            "\"missing_concepts\": [\"<key concepts from the correct answer that are truly missing in the student's answer>\"], "
            "\"additional_concepts\": [\"<key concepts introduced by the student that are not present in the correct answer. Include only technically relevant concepts.>\"], "
            "\"reasons\": \"<clear explanation of the matching score and reasoning behind each listed missing or additional concept. Avoid vague justifications.>\""
            "}."
        ),
        answer=" ".join(combined_keywords),
        details=1
    )

    # Parse and display the results
    try:
        # Check if semantic_result is a string and needs parsing
        if isinstance(semantic_result, str):
            response = json.loads(semantic_result)  # Parse the semantic_result into a dictionary
        else:
            response = semantic_result  # Use it directly if it's already a dictionary

        # Check if 'content' exists in the response
        if 'content' in response:
            # Parse the nested JSON string inside the "content" field
            if isinstance(response['content'], str):
                try:
                    # Extract the JSON part from the content field
                    response['content'] = json.loads(response['content'])
                except json.JSONDecodeError as e:
                    print(f"Error decoding 'content' field: {e}")
                    response['content'] = {
                        "answer_match": "0%",
                        "missing_concepts": [],
                        "additional_concepts": [],
                        "reasons": "Error parsing the 'content' field."
                    }
        else:
            # Handle missing 'content' key
            print("Error: 'content' key is missing in the response.")
            response['content'] = {
                "answer_match": "0%",
                "missing_concepts": [],
                "additional_concepts": [],
                "reasons": "The 'content' key is missing in the API response."
            }

        # Extract details from the response
        answer_match = response['content'].get('answer_match', "0%")
        missing_concepts = response['content'].get('missing_concepts', [])
        additional_concepts = response['content'].get('additional_concepts', [])
        reasons = response['content'].get('reasons', "")

        print("\n=== Lecture Analysis ===")
        print(f"Answer Match: {answer_match}")

        # Print Primary Missed Keywords
        print("\nPrimary Missed Keywords (Most Important):")
        for idx, keyword in enumerate(missing_concepts[:20], 1):  # Limit to top 20
            print(f"{idx}. {keyword}")

        # Print Keyword Comparison Results
        print("\n=== Keyword Comparison ===")
        print(f"Common Keywords: {comparison_results['common_keywords']}")
        print(f"Top 20 Missing in Dynamic Critical Keywords: {comparison_results['missing_in_dynamic'][:20]}")
        print(f"Top 20 Unique to Dynamic Critical Keywords: {comparison_results['unique_to_dynamic'][:20]}")

        # Print Reasons from Semantic Matching
        print("\nReasons from Semantic Matching:")
        print(reasons)

        #print(f"\nTotal Questions Asked: {total_questions}")
        print("=====================")

    except Exception as e:
        print(f"Error processing semantic result: {e}")

else:
    print("Error: Could not process audio file")

# Analyze classroom audio
result = analyze_classroom_audio('D:/Faculty_proc_new/Faculty_processing_LMS/audio_file2.mp3')

# Print the results
#print("\n=== Transcription ===")
#print(result['transcript'])

print("\n=== Question Analysis ===")
print("Trainer Questions:", result['trainer_questions'])
print("Student Questions:", result['student_questions'])
print("Unique Students Participated:", result['unique_students'])
#print("Trainer Answers:", result['trainer_answers'])
#print("Student Answers:", result['student_answers'])
#print("Detailed Answers:", result['detailed_answers'])



