import json
import soundfile as sf
from nlp_keywords import get_weighted_queries
from smatch import semantic_smart_answer
from transcribe import transcribe_audio  # Use the actual transcribe function
import requests
import boto3
import os
from queue import Queue
from fetch_keywords import fetch_keywords, fetch_all_keywords
from question_check import analyze_classroom_audio
from botocore.config import Config
from urllib.parse import urlparse

# S3 client configuration
s3 = boto3.client('s3',
                  aws_access_key_id="AKIA43Y7V2OHU52XAOHQ",
                  aws_secret_access_key="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",
                  config=Config(signature_version='s3v4', s3={'use_accelerate_endpoint': True}),
                  region_name="ap-south-1")

# Google Drive file download function 
def download_file_from_google_drive(file_id, destination):
    if os.path.exists(destination):
        os.remove(destination)
        print(f"Deleted existing file: {destination}")

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
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

# S3 download function

def download_from_s3(s3_url, local_path):
    try:
        parsed = urlparse(s3_url)
        bucket = parsed.netloc.split('.')[0]
        key = parsed.path.lstrip('/')

        if not bucket or not key:
            raise ValueError("Invalid S3 URL: Missing bucket or key")

        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"Downloading from S3 bucket: {bucket}, key: {key} -> {local_path}")
        s3.download_file(bucket, key, local_path)
        print(f"Successfully downloaded: {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

# S3 upload function 
def upload_to_aws(local_file, s3_file, bucket="copyrvswaroop"):
    try:
        s3.upload_file(local_file, bucket, s3_file)
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False

def generate_s3_link(bucket, s3_file, region):
    return f"https://{bucket}.s3.{region}.amazonaws.com/{s3_file}"

def split_audio_into_chunks(local_audio_file_path, chunk_duration=240):
    try:
        output_dir = "audio_chunks"
        os.makedirs(output_dir, exist_ok=True)

        print("\n=== Processing Audio File ===")
        print(f"Input file: {local_audio_file_path}")

        audio, sample_rate = sf.read(local_audio_file_path)
        chunk_size_samples = int(chunk_duration * sample_rate)
        chunks = []

        total_chunks = (len(audio) + chunk_size_samples - 1) // chunk_size_samples
        print(f"Total chunks to process: {total_chunks}")

        for i in range(total_chunks):
            try:
                start_idx = i * chunk_size_samples
                end_idx = min(start_idx + chunk_size_samples, len(audio))
                chunk = audio[start_idx:end_idx]

                chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
                sf.write(chunk_path, chunk, sample_rate)
                chunks.append(chunk_path)

                s3_file_key = f"audio_chunks/{os.path.basename(chunk_path)}"
                if upload_to_aws(chunk_path, s3_file_key):
                    s3_link = generate_s3_link("copyrvswaroop", s3_file_key, "ap-south-1")
                    print(f"Processed chunk {i+1}/{total_chunks}:\n  - File: {chunk_path}\n  - S3 Link: {s3_link}\n")
            except Exception as e:
                print(f"Error processing chunk {i+1}/{total_chunks}: {str(e)}")
                continue

        print("=== Audio Processing Complete ===\n")
        return chunks

    except Exception as e:
        print(f"Error in split_audio_into_chunks: {str(e)}")
        return []

def process_audio_file(input_type, input_source):
    """
    Process either Google Drive or S3 audio file based on input type.
    For Google Drive, download and split the audio file into chunks.
    For S3, process the list of pre-split audio chunks directly.
    """
    if input_type == "google_drive":
        print(f"Downloading from Google Drive with ID: {input_source}")
        local_audio_file_path = "audio_file2.mp3"
        download_file_from_google_drive(input_source, local_audio_file_path)
        return split_audio_into_chunks(local_audio_file_path)  # Split and process chunks

    elif input_type == "s3":
        print("Processing pre-split audio chunks from S3...")
        if not isinstance(input_source, list):
            raise ValueError("For S3 input, input_source must be a list of S3 URLs.")

        chunk_paths = []
        for idx, url in enumerate(input_source):
            local_path = f"chunk_{idx}.wav"
            success = download_from_s3(url, local_path)
            if success and os.path.exists(local_path):
                chunk_paths.append(local_path)
            else:
                print(f"Skipping invalid or failed download: {url}")
        
        return chunk_paths  # Return the list of downloaded chunk paths

    else:
        raise ValueError("Invalid input type. Must be 'google_drive' or 's3'")

def normalize_keyword(keyword: str) -> str:
    """Normalize the keyword by lowercasing and removing punctuation."""
    import re
    return re.sub(r'[^\w\s]', '', keyword.lower()).strip()

def count_questions_in_transcript(transcript: str) -> int:
    """
    Counts the number of questions in the transcript using simple regex patterns.
    """
    import re
    
    # Split into sentences
    sentences = re.split('[.!?]', transcript)
    question_count = 0
    
    # Common question patterns
    question_patterns = [
        r'^(what|why|how|when|where|who|which|whose|do|does|did|is|are|can|could|would|will|should)\b.*',
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

def process_audio_chunks(chunks):
    if not chunks:
        print("Error: No audio chunks were created")
        return [], 0, set(), set(), []

    transcript_queue = Queue()

    # Process each chunk individually
    for chunk_path in chunks:
        transcript = transcribe_audio(chunk_path)
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
    critical_all_set = set(critical_all_keywords)
    dynamic_critical_set = set(dynamic_critical_keywords)

    common_keywords = critical_all_set & dynamic_critical_set
    missing_in_dynamic = critical_all_set - dynamic_critical_set
    unique_to_dynamic = dynamic_critical_set - critical_all_set

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
            dynamic_keywords = fetched_keywords[video_id]
        else:
            try:
                dynamic_keywords = fetch_all_keywords(video_id)
                fetched_keywords[video_id] = dynamic_keywords
                print(f"Fetched keywords for video ID {video_id}: {dynamic_keywords}")
            except Exception as e:
                print(f"Error fetching keywords for video ID {video_id}: {e}")
                dynamic_keywords = []

        all_keywords.update(dynamic_keywords)

    return sorted(all_keywords)

# Main processing logic
def main(input_type="google_drive", input_source=None):
    """
    Main function to process audio files from Google Drive or S3.
    """
    if not input_source:
        if input_type == "google_drive":
            input_source = "1e11nDLwLFr5hVWlMyPmjYWFg4s0mgYrt"  # Default Google Drive ID
        else:
            raise ValueError("S3 URLs must be provided when input_type is 's3'")

    # Process the audio file based on input type
    chunks = process_audio_file(input_type, input_source)

    if not chunks:
        print("Error: No audio chunks were created or downloaded.")
        return

    # Process the audio chunks and extract keywords
    processed_transcripts, total_questions, primary_missed_keywords, secondary_missed_keywords, top_10_secondary_missed_keywords = process_audio_chunks(chunks)

    # Combine all processed transcripts into a single transcript
    complete_transcript = " ".join(processed_transcripts)

    if processed_transcripts:
        video_ids = ['Oy4duAOGdWQ', 'P2PMgnQSHYQ', 'efR1C6CvhmE']  # Example video IDs
        print(f"Processing video IDs: {video_ids}")

        # Fetch and unionize keywords for all video IDs
        unionized_keywords = fetch_and_unionize_keywords(video_ids)
        print(f"Unionized Keywords from All Videos: {unionized_keywords}")

        critical_keywords = unionized_keywords
        flat_keywords = []

        try:
            hardcoded_keywords = fetch_keywords('Oy4duAOGdWQ')
            flat_keywords = [str(keyword) for sublist in hardcoded_keywords for keyword in sublist]
            print(f"Flat Keywords from 5-Minute Segments: {flat_keywords}")
        except Exception as e:
            print(f"Error fetching hardcoded keywords: {e}")
            flat_keywords = []

        combined_keywords = sorted(set(critical_keywords + flat_keywords))
        print(f"Combined Keywords for Semantic Matching: {combined_keywords}")

        comparison_results = compare_keywords(flat_keywords, critical_keywords)

        semantic_result = semantic_smart_answer(
            student_answer=" ".join(processed_transcripts),
            question=(  # Semantic matching question logic remains unchanged
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

        try:
            if isinstance(semantic_result, str):
                response = json.loads(semantic_result)
            else:
                response = semantic_result

            if 'content' in response:
                if isinstance(response['content'], str):
                    try:
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
                print("Error: 'content' key is missing in the response.")
                response['content'] = {
                    "answer_match": "0%",
                    "missing_concepts": [],
                    "additional_concepts": [],
                    "reasons": "The 'content' key is missing in the API response."
                }

            answer_match = response['content'].get('answer_match', "0%")
            missing_concepts = response['content'].get('missing_concepts', [])
            additional_concepts = response['content'].get('additional_concepts', [])
            reasons = response['content'].get('reasons', "")

            print("\n=== Lecture Analysis ===")
            print(f"Answer Match: {answer_match}")

            print("\nPrimary Missed Keywords (Most Important):")
            for idx, keyword in enumerate(missing_concepts[:20], 1):
                print(f"{idx}. {keyword}")

            print("\n=== Keyword Comparison ===")
            print(f"Common Keywords: {comparison_results['common_keywords']}")
            print(f"Top 20 Missing in Dynamic Critical Keywords: {comparison_results['missing_in_dynamic'][:20]}")
            print(f"Top 20 Unique to Dynamic Critical Keywords: {comparison_results['unique_to_dynamic'][:20]}")

            print("\nReasons from Semantic Matching:")
            print(reasons)

        except Exception as e:
            print(f"Error processing semantic result: {e}")

    else:
        print("Error: Could not process audio file")

    # Analyze classroom audio
    result = analyze_classroom_audio(chunks[0])  # Use the first chunk for analysis

    print("\n=== Question Analysis ===")
    print("Trainer Questions:", result['trainer_questions'])
    print("Student Questions:", result['student_questions'])
    print("Unique Students Participated:", result['unique_students'])

if __name__ == '__main__':
    # Example usage:
    # For Google Drive:
    #main(input_type="google_drive", input_source="1e11nDLwLFr5hVWlMyPmjYWFg4s0mgYrt")
    
      #For S3:
    main(input_type="s3", input_source=[
        "https://copyrvswaroop.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_0.wav",
        "https://copyrvswaroop.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_1.wav"
    ])