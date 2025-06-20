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
from queue import Queue
from fetch_keywords import fetch_keywords, fetch_all_keywords
from question_check import analyze_classroom_audio
from botocore.config import Config
from urllib.parse import urlparse
from collections import Counter, defaultdict

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id="AKIA43Y7V2OHU52XAOHQ",
    aws_secret_access_key="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",
    config=Config(signature_version='s3v4', s3={'use_accelerate_endpoint': True}),
    region_name="ap-south-1"
)

def generate_filename(extension="mp3"):
    return f"{int(time.time() // 60)}_class_audio.{extension}"

def extract_gdrive_file_id(gdrive_url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)/", gdrive_url)
    if match:
        return match.group(1)
    raise ValueError("Invalid Google Drive URL")

def download_file_from_google_drive(file_id, destination):
    if os.path.exists(destination):
        os.remove(destination)

    def get_confirm_token(response):
        return next((v for k, v in response.cookies.items() if k.startswith('download_warning')), None)

    def save_response_content(response):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    save_response_content(response)

def download_from_s3(s3_url, local_path):
    try:
        parsed = urlparse(s3_url)
        bucket = parsed.netloc.split('.')[0]
        key = parsed.path.lstrip('/')
        if os.path.exists(local_path):
            os.remove(local_path)
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"S3 Download Error: {e}")
        return False

def upload_to_aws(local_file, s3_file, bucket="chatlms"):
    try:
        s3.upload_file(local_file, bucket, s3_file)
        return True
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return False

def generate_s3_link(bucket, s3_file, region):
    return f"https://{bucket}.s3.{region}.amazonaws.com/{s3_file}"

def split_audio_into_chunks(local_audio_file_path, chunk_duration=240):
    try:
        output_dir = "audio_chunks"
        os.makedirs(output_dir, exist_ok=True)
        audio, sample_rate = sf.read(local_audio_file_path)
        chunk_size_samples = int(chunk_duration * sample_rate)
        chunks = []
        for i in range((len(audio) + chunk_size_samples - 1) // chunk_size_samples):
            start_idx = i * chunk_size_samples
            end_idx = min(start_idx + chunk_size_samples, len(audio))
            chunk = audio[start_idx:end_idx]
            chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk, sample_rate)
            chunks.append(chunk_path)
            upload_to_aws(chunk_path, os.path.basename(chunk_path))
        return chunks
    except Exception as e:
        print(f"Chunking Error: {e}")
        return []

def process_audio_file(input_type, input_source):
    if input_type == "google_drive":
        local_audio_file_path = generate_filename("mp3")
        download_file_from_google_drive(input_source, local_audio_file_path)
        return split_audio_into_chunks(local_audio_file_path)
    elif input_type == "s3":
        chunk_paths = []
        for idx, url in enumerate(input_source):
            local_path = f"chunk_{idx}.wav"
            if download_from_s3(url, local_path):
                chunk_paths.append(local_path)
        return chunk_paths
    raise ValueError("Invalid input type")

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

def fetch_and_unionize_keywords(video_ids):
    all_keywords = set()
    for vid in video_ids[0]:
        try:
            all_keywords.update(fetch_all_keywords(vid))
        except Exception as e:
            print(f"Keyword fetch error for {vid}: {e}")
    return sorted(all_keywords)

def process_audio(course_id, input_type="google_drive", input_source=None, server_type="dev"):
    if input_type == "google_drive":
        input_source = extract_gdrive_file_id(input_source)

    video_ids = get_course_vids_secs(course_id, server_type, video_type=2)
    chunks = process_audio_file(input_type, input_source)
    if not chunks:
        print("No audio chunks found.")
        return

    transcripts, total_questions, full_transcript = process_audio_chunks(chunks)
    if not transcripts:
        print("Failed to process transcripts.")
        return

    weighted_keywords = get_weighted_queries(
        full_transcript,
        len(full_transcript),
        "computer science",
        "computer science"
    )[0]

    import ast  # Safe parsing fallback inside the function

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


    top_20_weighted = dict(sorted(
        {k: v for k, v in weighted_keywords.items() if v > 1}.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20])
    reference_keywords = list(top_20_weighted.keys())

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
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Use only the transcript as a single document for now
    vectorizer = TfidfVectorizer(vocabulary=reference_keywords, lowercase=True, use_idf=True, smooth_idf=True)
    tfidf_matrix = vectorizer.fit_transform([full_transcript])
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    # Adjust top_20_weighted by IDF (multiply original weight by IDF)
    idf_weighted = {}
    for k in reference_keywords:
        idf = idf_scores.get(k.lower(), 1.0)
        idf_weighted[k] = top_20_weighted[k] * idf

    # Sort again by the new IDF-weighted score, but keep only the same keys as before to not affect logic
    idf_sorted = dict(sorted(idf_weighted.items(), key=lambda x: x[1], reverse=True))

    # Pipe character output for keyword presence
    print("\nKeyword | Presence (pipe format):")
    for k in reference_keywords:
        print(f"{k} | {keyword_presence[k]}")

    return json.dumps([{
        "answer_match": response['content'].get('answer_match', "0%"),
        "missing_concepts": response['content'].get('missing_concepts', []),
        "reasons": response['content'].get('reasons', ""),
        "reference_keywords": reference_keywords,
        "keyword_presence": keyword_presence,
        "top_20_weighted_keywords": top_20_weighted,
        "trainer_questions": analysis_result['trainer_questions'],
        "student_questions": analysis_result['student_questions'],
        "unique_students_participated": analysis_result['unique_students']
    }], indent=4)

if __name__ == '__main__':
    result = process_audio(
        course_id=212,
        input_type="s3",
        input_source=[
            "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_0.wav",
            "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_1.wav",
            "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_2.wav",
            "https://chatlms.s3.ap-south-1.amazonaws.com/audio_chunks/chunk_3.wav"
        ]
    )
    #print(result)
