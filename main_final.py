import json
import soundfile as sf
from nlp_keywords import get_weighted_queries
from smatch import semantic_smart_answer
from transcribe import transcribe_audio
from fetch_videos import get_course_vids_secs
import requests
import boto3
import os
import re
import time
import ast
from queue import Queue
from fetch_keywords import fetch_keywords, fetch_all_keywords
from question_check import analyze_classroom_audio
from botocore.config import Config
from urllib.parse import urlparse
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# S3 client configuration
s3 = boto3.client('s3',
                  aws_access_key_id="AKIA43Y7V2OHU52XAOHQ",
                  aws_secret_access_key="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",
                  config=Config(signature_version='s3v4', s3={'use_accelerate_endpoint': True}),
                  region_name="ap-south-1")


def generate_filename(extension="mp3"):
    current_time_in_minutes = int(time.time() // 60)
    return f"{current_time_in_minutes}_class_audio.{extension}"


def extract_gdrive_file_id(gdrive_url):
    match = re.search(r"\/d\/([a-zA-Z0-9_-]+)\/", gdrive_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid Google Drive URL")


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
def upload_to_aws(local_file, s3_file, bucket="chatlms"):
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
        chunk_links = []

        total_chunks = (len(audio) + chunk_size_samples - 1) // chunk_size_samples
        # print(f"Total chunks to process: {total_chunks}")

        for i in range(total_chunks):
            try:
                start_idx = i * chunk_size_samples
                end_idx = min(start_idx + chunk_size_samples, len(audio))
                chunk = audio[start_idx:end_idx]

                chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
                sf.write(chunk_path, chunk, sample_rate)
                chunks.append(chunk_path)

                s3_file_key = os.path.basename(chunk_path)
                if upload_to_aws(chunk_path, s3_file_key):
                    s3_link = generate_s3_link("chatlms", s3_file_key, "ap-south-1")
                    chunk_links.append(s3_link)
                    # print(f"Processed chunk {i+1}/{total_chunks}:\n  - File: {chunk_path}\n  - S3 Link: {s3_link}\n")
            except Exception as e:
                print(f"Error processing chunk {i+1}/{total_chunks}: {str(e)}")
                continue

        print("=== Audio Processing Complete ===\n")
        return chunks, chunk_links

    except Exception as e:
        print(f"Error in split_audio_into_chunks: {str(e)}")
        return [], []

def process_audio_file(input_type, input_source):
    """
    Process either Google Drive or S3 audio file based on input type.
    For Google Drive, download and split the audio file into chunks.
    For S3, process the list of pre-split audio chunks directly.
    """
    if input_type == "google_drive":
        local_audio_file_path = generate_filename(extension="mp3")
        print(f"Downloading from Google Drive into: {local_audio_file_path}")
        download_file_from_google_drive(input_source, local_audio_file_path)
        chunk_paths, _ = split_audio_into_chunks(local_audio_file_path)
        return chunk_paths
    
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
    
    return re.sub(r'[^\w\s]', '', keyword.lower()).strip()

def count_questions_in_transcript(transcript: str) -> int:
    """
    Counts the number of questions in the transcript using simple regex patterns.
    """
    
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
        return [], 0, ""

    transcripts = [transcribe_audio(chunk) or "" for chunk in chunks]
    full_transcript = " ".join(transcripts).strip()

    if len(full_transcript) < 100:
        print("WARNING: Transcript may be too short")
    question_count = len(re.findall(
        r'\b(what|why|how|when|where|who|which|do|does|did|is|are|can|could|would|will|should)\b',
        full_transcript.lower()))
    return transcripts, question_count, full_transcript


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
    for vid in video_ids[0]:
        try:
            all_keywords.update(fetch_all_keywords(vid))
        except Exception as e:
            print(f"Keyword fetch error for {vid}: {e}")

    return sorted(all_keywords)


def get_audio_duration(audio_file_path):
    """Returns the duration of the audio file in seconds."""
    audio, sample_rate = sf.read(audio_file_path)
    return len(audio) / sample_rate

# Main processing logic
def process_audio(course_id, input_type="google_drive", input_source=None, server_type="dev"):
    """
    Main function to process audio files based on course_id and server_type.
    For Google Drive, process audio files using the Google Drive ID.
    For S3, process audio files using the provided list of S3 URLs.
    """
    if input_type == "google_drive":
        # Extract file ID from Google Drive URL
        input_source = extract_gdrive_file_id(input_source)

    video_ids = get_course_vids_secs(course_id, server_type, video_type=2)
    
    # Process the audio file based on input type
    chunks = process_audio_file(input_type, input_source)
    if not chunks:
        print("No audio chunks found.")
        return

    transcripts, total_questions, full_transcript = process_audio_chunks(chunks)
    if not transcripts:
        print("Failed to process transcripts.")
        return

    weighted_keywords = get_weighted_queries(full_transcript, len(full_transcript), "computer science", "computer science")[0]

    if isinstance(weighted_keywords, str):
        try:
            weighted_keywords = json.loads(weighted_keywords)
        except json.JSONDecodeError:
            print("Warning: falling back to custom regex parser due to JSONDecodeError")
        try:
            weighted_keywords_parsed = {}
            pattern = re.findall(r'"([^"\n]+)"\^(\d+(?:\.\d+)?)', weighted_keywords)
            for kw, score in pattern:
                weighted_keywords_parsed[kw] = float(score)
            weighted_keywords = weighted_keywords_parsed
        except Exception as e:
            print("RAW weighted_keywords string:", weighted_keywords)
            print(f"Error parsing weighted keywords: {e}")
            weighted_keywords = {}

    # --- CHANGED SECTION: Select 70% of keywords based on audio duration ---
    # Get duration of the first chunk (assuming all chunks are from the same file)
    audio_duration = get_audio_duration(chunks[0]) if chunks else 0

    # Sort keywords by weight (descending)
    sorted_keywords = sorted(
        {k: v for k, v in weighted_keywords.items() if v > 1}.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Take 70% of the keywords (rounded up)
    num_keywords = int(len(sorted_keywords) * 0.7 + 0.5)
    selected_keywords = dict(sorted_keywords[:num_keywords])
    reference_keywords = list(selected_keywords.keys())
    # --- END CHANGED SECTION ---

    transcript_lower = full_transcript.lower()
    keyword_presence = {
        k: 1 if re.search(r'\b' + re.escape(k.lower()) + r'\b', transcript_lower) else 0
        for k in reference_keywords
    }

    semantic_result = semantic_smart_answer(
        student_answer=full_transcript,
        question=(
            "Evaluate the semantic similarity between the student's answer and the correct answer, focusing on core conceptual alignment. "
            "Only include critical concepts that are truly missing. Avoid superficial or redundant terms. "
            "Output JSON with answer_match, missing_concepts, additional_concepts, and reasons."
        ),
        answer=" ".join(reference_keywords),
        details=1
    )

    if isinstance(semantic_result, str):
        try:
            response = json.loads(semantic_result)
        except Exception as e:
            response = {"content": {
                "answer_match": "0%",
                #"missing_concepts": [],
                #"additional_concepts": [],
                "reasons": f"Parsing error: {e}"
            }}
    else:
        response = semantic_result

    if 'content' in response and isinstance(response['content'], str):
        try:
            response['content'] = json.loads(response['content'])
        except Exception:
            response['content'] = {
                "answer_match": "0%",
                "missing_concepts": [],
                "additional_concepts": [],
                "reasons": "Could not parse semantic content."
            }

    analysis_result = analyze_classroom_audio(chunks[0])

    # --- IDF Calculation using sklearn ---
    # Use only the transcript as a single document for now
    vectorizer = TfidfVectorizer(vocabulary=reference_keywords, lowercase=True, use_idf=True, smooth_idf=True)
    tfidf_matrix = vectorizer.fit_transform([full_transcript])
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    # Adjust top_20_weighted by IDF (multiply original weight by IDF)
    idf_weighted = {}
    for k in reference_keywords:
        idf = idf_scores.get(k.lower(), 1.0)
        idf_weighted[k] = selected_keywords[k] * idf

    # Sort again by the new IDF-weighted score, but keep only the same keys as before to not affect logic
    idf_sorted = dict(sorted(idf_weighted.items(), key=lambda x: x[1], reverse=True))

    # Pipe character output for keyword presence
    # print("\nKeyword | Presence (pipe format):")
    # for k in reference_keywords:
    #     print(f"{k} | {keyword_presence[k]}")

    return json.dumps([{
        "answer_match": response['content'].get('answer_match', "0%"),
        # "missing_concepts": response['content'].get('missing_concepts', []),
        # "reasons": response['content'].get('reasons', ""),
        # "reference_keywords": reference_keywords,
        "keyword_presence": keyword_presence,
        # "top_20_weighted_keywords": top_20_weighted,
        "trainer_questions": analysis_result['trainer_questions'],
        "student_questions": analysis_result['student_questions'],
        "unique_students_participated": analysis_result['unique_students']
    }], indent=4)


if __name__ == '__main__':
    # Example usage:
    # For Google Drive:
    # process_audio(input_type="google_drive", input_source="1e11nDLwLFr5hVWlMyPmjYWFg4s0mgYrt")
    # result = process_audio(input_type="google_drive", input_source="https://drive.google.com/file/d/1Ll9yM0YPFTl1UZsUHZEsPCl7S0PIroyC/view?usp=drive_link", course_id=212)
    # print(result)

    # For S3:
    # result = process_audio(course_id=212, input_type="s3", input_source=["https://copyrvswaroop.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_0.wav", "https://copyrvswaroop.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_1.wav"])
    # result = process_audio(course_id=212, input_type="s3", input_source=["https://chatlms.s3.ap-south-1.amazonaws.com/chunk_0.wav", "https://chatlms.s3.ap-south-1.amazonaws.com/chunk_1.wav", "https://chatlms.s3.ap-south-1.amazonaws.com/chunk_2.wav", "https://chatlms.s3.ap-south-1.amazonaws.com/chunk_3.wav"])
    result = process_audio(course_id=212, input_type="s3", input_source=["https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_0.wav", "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_1.wav", "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_2.wav", "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_3.wav"])

    print(result)
