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
from fetch_keywords import fetch_keywords, fetch_all_keywords
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
google_drive_file_id = "13qfbbFB38m_5iqtJCpReEx29HWUV2cqp"  
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
        print(f"Transcript for {chunk_path}: {transcript}")
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
    print(f"\nDEBUG - Complete Transcript: {complete_transcript}")
    print(f"DEBUG - Complete Transcript Length: {len(complete_transcript)}")
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

from local_dynamic_critical_all_keywords import JunkWordProcessor
from difflib import get_close_matches

# Initialize JunkWordProcessor
junk_processor = JunkWordProcessor()

# Function to rank and filter missed keywords
def get_primary_missed_keywords(corrected_keywords, critical_keywords, top_n=10):
    """
    Identify the most important missed keywords using ranking and filtering.
    """
    # Normalize keywords
    corrected_keywords = {junk_processor.normalize_keyword(kw) for kw in corrected_keywords}
    critical_keywords = {junk_processor.normalize_keyword(kw) for kw in critical_keywords}

    # Filter out junk keywords from critical keywords
    critical_keywords = {kw for kw in critical_keywords if kw not in junk_processor.junk_keywords}

    # Find missed keywords
    missed_keywords = critical_keywords - corrected_keywords

    # Rank missed keywords by closeness to corrected keywords
    ranked_missed_keywords = []
    for missed in missed_keywords:
        # Find the closest match in corrected keywords
        closest_match = get_close_matches(missed, corrected_keywords, n=1, cutoff=0.6)
        similarity_score = len(closest_match[0]) if closest_match else 0  # Use length as a simple score
        ranked_missed_keywords.append((missed, similarity_score))

    # Sort by similarity score (descending) and return the top N
    ranked_missed_keywords.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in ranked_missed_keywords[:top_n]]

# Process the audio file and extract keywords
corrected_transcript_keywords, total_questions, primary_missed_keywords, secondary_missed_keywords, top_10_secondary_missed_keywords = process_audio_chunks(local_audio_local_audio_file_path)

# Ensure corrected_transcript_keywords is not empty
if not corrected_transcript_keywords:
    print("Error: No keywords extracted from the transcript.")
    corrected_transcript_keywords = []

# Fetch critical_all_keywords dynamically
try:
    critical_all_keywords = fetch_all_keywords('Oy4duAOGdWQ')  # Replace with dynamic input if needed
except Exception as e:
    print(f"Error fetching critical_all_keywords: {e}")
    critical_all_keywords = []

# Fetch hardcoded_keywords dynamically
try:
    hardcoded_keywords = fetch_keywords('Oy4duAOGdWQ')  # Replace with dynamic input if needed
    flat_keywords = [str(keyword) for sublist in hardcoded_keywords for keyword in sublist]
except Exception as e:
    print(f"Error fetching hardcoded_keywords: {e}")
    flat_keywords = []

# Get the top 10 primary missed keywords
primary_missed_keywords = get_primary_missed_keywords(corrected_transcript_keywords, critical_all_keywords, top_n=10)

# Populate top_10_secondary_missed_keywords based on secondary_missed_keywords
top_10_secondary_missed_keywords = list(secondary_missed_keywords)[:10]  # Adjust logic as needed

# Debug: Print Primary Missed Keywords
print("\nPrimary Missed Keywords (Most Important):")
if primary_missed_keywords:
    for idx, keyword in enumerate(primary_missed_keywords, 1):
        print(f"{idx}. {keyword}")
else:
    print("No primary missed keywords detected.")

# Debug: Print Secondary Missed Keywords
print("\nSecondary Missed Keywords:")
if secondary_missed_keywords:
    for idx, keyword in enumerate(secondary_missed_keywords, 1):
        print(f"{idx}. {keyword}")
else:
    print("No secondary missed keywords detected.")

# Semantic Matching
semantic_result = semantic_smart_answer(
    student_answer=" ".join(corrected_transcript_keywords),
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
        "\"additional_concepts\": [\"<key concepts introduced by the student that are not present in the correct answer>\"], "
        "\"reasons\": \"<clear explanation of the matching score and reasoning behind each listed missing or additional concept. Avoid vague justifications.>\""
        "}."
    ),
    answer=" ".join(flat_keywords),
    details=1
)

# Parse and display the results
try:
    if isinstance(semantic_result, str):
        response = json.loads(semantic_result)
    else:
        response = semantic_result

    if isinstance(response['content'], str):
        try:
            content_start = response['content'].find("{")
            if content_start != -1:
                response['content'] = json.loads(response['content'][content_start:])
            else:
                raise ValueError("No JSON object found in 'content' field")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error decoding nested JSON in 'content': {e}")
            response['content'] = {
                "answer_match": "0%",
                "missing_concepts": [],
                "additional_concepts": [],
                "reasons": "Error parsing the 'content' field."
            }

    answer_match = response['content'].get('answer_match', "0%")
    missing_concepts = response['content'].get('missing_concepts', [])
    additional_concepts = response['content'].get('additional_concepts', [])
    reasons = response['content'].get('reasons', "")

    print("\n=== Lecture Analysis ===")
    print(f"Answer Match: {answer_match}")

    # Refine Primary Missed Keywords Based on Match Score
    match_score = float(answer_match.strip('%'))
    if match_score >= 95:
        print("\nHigh match score detected (>= 95%). Filtering primary missed keywords to top 5...")
        primary_missed_keywords = primary_missed_keywords[:2]  # Limit to top 2 for high match scores
    elif match_score >= 90:
        print("\nModerate match score detected (>= 90%). Filtering primary missed keywords to top 3...")
        primary_missed_keywords = primary_missed_keywords[:3]  # Limit to top 3 for moderate match scores
    else:
        print("\nMatch score below 90%. Displaying all primary missed keywords...")

    # Consolidated printing of missed keywords
    print("\nPrimary Missed Keywords (Filtered):")
    if primary_missed_keywords:
        for idx, keyword in enumerate(primary_missed_keywords, 1):
            print(f"{idx}. {keyword}")
    else:
        print("No primary missed keywords detected.")

    print("\nSecondary Missed Keywords:")
    if secondary_missed_keywords:
        for idx, keyword in enumerate(secondary_missed_keywords, 1):
            print(f"{idx}. {keyword}")
    else:
        print("No secondary missed keywords detected.")

    print("\nReasons:")
    print(reasons)

except Exception as e:
    print(f"Error parsing semantic result: {e}")



