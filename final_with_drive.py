import json
import language_tool_python
import soundfile as sf
from nlp_keywords import get_weighted_queries
from smatch import semantic_smart_answer
from transcribe import transcribe_audio  # Use the actual transcribe function
import requests
import boto3
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


# Initialize grammar checker
tool = language_tool_python.LanguageTool('en-US')

# Fetch the audio file from Google Drive (replace with actual file ID)
google_drive_file_id = "13qfbbFB38m_5iqtJCpReEx29HWUV2cqp"  
local_audio_file_path = "audio_file.mp3"
download_file_from_google_drive(google_drive_file_id, local_audio_file_path)

def split_audio_into_chunks(file_path, chunk_duration=300):
    """
    Splits the audio into chunks of specified duration (default: 5 minutes = 300 seconds).
    Using soundfile to split the audio.
    """
    audio, sample_rate = sf.read(file_path)
    chunk_size_samples = chunk_duration * sample_rate  # Convert duration in seconds to samples

    chunks = []
    for i in range(0, len(audio), chunk_size_samples):
        chunk = audio[i:i + chunk_size_samples]
        chunk_path = f"chunk_{i // chunk_size_samples}.wav"
        sf.write(chunk_path, chunk, sample_rate)
        chunks.append(chunk_path)

        # Upload to S3 after splitting
        s3_file_key = f"audio_chunks/{chunk_path}"
        upload_success = upload_to_aws(chunk_path, s3_file_key, bucket="copyrvswaroop")
        if upload_success:
            print(f"Successfully uploaded chunk {chunk_path} to S3.")
            s3_link = generate_s3_link("copyrvswaroop", s3_file_key, "ap-south-1")
            print(f"S3 Link: {s3_link}")
        else:
            print(f"Failed to upload chunk {chunk_path} to S3.")
    
    return chunks


def split_text(text, chunk_size=1000):
    """Splits text into chunks of specified size (default ~1000 characters)."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def correct_grammar(transcript: str) -> str:
    """Corrects grammar using LanguageTool in chunks to avoid overload."""
    chunks = split_text(transcript, chunk_size=1000)  # Adjust size if needed
    corrected_chunks = [tool.correct(chunk) for chunk in chunks]
    return " ".join(corrected_chunks)


def normalize_keyword(keyword: str) -> str:
    """Normalize the keyword by lowercasing and removing punctuation."""
    import re
    return re.sub(r'[^\w\s]', '', keyword.lower()).strip()


def analyze_questions(transcript):
    import re
    
    # Add debug prints
    print("\nDEBUG - Question Analysis:")
    print(f"Transcript length: {len(transcript)} characters")
    
    # Split transcript into sentences (basic splitting on .!?)
    sentences = re.split('[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"Number of sentences found: {len(sentences)}")
    
    # Common question patterns
    patterns = [
        r'\?',  # Questions with question mark
        r'^(what|who|where|when|why|how|could|would|will)\s+.*',  # WH questions 
        r'^(do|does|did|is|are|was|were|have|has|had)\s+\w+.*',  # Auxiliary verb questions
        r'^(can|could|would|will|should|may|might)\s+\w+.*'  # Modal questions
    ]
    
    questions = []
    for sentence in sentences:
        # Check if sentence matches any pattern
        is_question = False
        for pattern in patterns:
            if re.search(pattern, sentence.lower()):
                is_question = True
                break
                
        if is_question and len(sentence) > 10:
            questions.append(sentence)
    
    print(f"DEBUG - Questions detected: {len(questions)}\n")
    return len(questions), questions

def process_audio_chunks(file_path):
    chunks = split_audio_into_chunks(file_path)
    transcript_queue = Queue()
    
    for chunk_path in chunks:
        transcript = transcribe_audio(chunk_path)
        if transcript:
            transcript_queue.put(correct_grammar(transcript))
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
    
    # In process_audio_chunks, before analyzing questions
    complete_transcript = " ".join(processed_transcripts)
    print(f"\nDEBUG - Complete Transcript Length: {len(complete_transcript)}")
    if len(complete_transcript) < 100:  # If transcript is suspiciously short
        print("WARNING: Transcript might be empty or too short")
    
    # Analyze questions in the complete transcript
    question_count, questions_list = analyze_questions(complete_transcript)
    print("\nQuestion Analysis Results:")
    print(f"Total questions asked: {question_count}")
    print("\nQuestions found:")
    for i, question in enumerate(questions_list, 1):
        print(f"{i}. {question}")
        
    
    extracted_keywords = set()
    
    for transcript in processed_transcripts:
        subject = "computer science"
        level = "computer science"
        _, phrasescorelist, _, _ = get_weighted_queries(transcript, len(transcript), subject, level)
        extracted_keywords.update(normalize_keyword(kw[0]) for kw in phrasescorelist)
    
    return extracted_keywords, question_count  # Return both values


corrected_transcript_keywords, total_questions = process_audio_chunks("audio_file.mp3")  # Capture both return values
print("Extracted Keywords:", corrected_transcript_keywords)


# Fetching keywords from fetch_keywords.py
# Using 'Oy4duAOGdWQ' as video_id in this case
hardcoded_keywords = fetch_keywords('Oy4duAOGdWQ')
flat_keywords = [str(keyword) for sublist in hardcoded_keywords for keyword in sublist]  # Flatten the list of tuples

# Add this before semantic matching
def normalize_all_keywords(keywords):
    return [normalize_keyword(kw) for kw in keywords if kw.strip()]

corrected_transcript_keywords = normalize_all_keywords(corrected_transcript_keywords)
flat_keywords = normalize_all_keywords(flat_keywords)

# Step 4: Semantic Matching
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
    details=1  # Request all details (includes missing, additional keywords, and reasons)
)

# Parse the response and display relevant information
response = json.loads(semantic_result)

# Extract all the relevant details from the response
answer_match = response['content'][0]['answer_match']
missing_concepts = response['content'][0]['missing_concepts']
additional_concepts = response['content'][0]['additional_concepts']
reasons = response['content'][0]['reasons']

# Compute missed and extra keywords manually
extra_keywords = [kw for kw in flat_keywords if kw not in additional_concepts]

# Replace the existing missed_keywords computation section with:

def get_top_missed_keywords(corrected_keywords, missing_concepts, transcript, n=10):
    """Get top N missed keywords based on weights from transcript"""
    # Get weights for the complete transcript
    _, phrasescorelist, _, _ = get_weighted_queries(transcript, len(transcript), "computer science", "computer science")
    
    # Create a dictionary of keyword-weight pairs from phrasescorelist
    keyword_weights = {kw: weight for kw, weight in phrasescorelist}
    
    # Get missed keywords with their weights
    weighted_missed = []
    for kw in corrected_keywords:
        if kw not in missing_concepts:
            # Use the weight from phrasescorelist if available, else use 0
            weight = keyword_weights.get(kw, 0)
            weighted_missed.append((kw, weight))
    
    # Sort by weight in descending order and take top N
    sorted_keywords = sorted(weighted_missed, key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:n]], phrasescorelist

# Get the transcript from the joined corrected_transcript_keywords
transcript = " ".join(corrected_transcript_keywords)
missed_keywords, phrasescorelist = get_top_missed_keywords(corrected_transcript_keywords, missing_concepts, transcript)

# Print all the details
print(f"Answer Match: {answer_match}")
print(f"Key Phrases Missing in Student Answer Present in Trainer Answer: {', '.join(missing_concepts)}")
print(f"Key Phrases Mentioned in Student Answer Not in Trainer Answer: {', '.join(additional_concepts)}")
print(f"Missed Keywords: {', '.join(missed_keywords)}")
#print(f"Extra Keywords: {', '.join(extra_keywords)}")
print(f"Reasons: {reasons}")

print("\n" + "-"*50)
print("FINAL ANALYSIS")
print("-"*50)
print(f"Content Match Percentage: {answer_match}%")
print(f"Total Questions Found: {total_questions}")
print("\nTop 10 Missing Keywords by Weight:")
for i, keyword in enumerate(missed_keywords, 1):
    weight = dict(phrasescorelist).get(keyword, 0)
    print(f"{i}. {keyword} (weight: {weight:.2f})")
print("-"*50)

