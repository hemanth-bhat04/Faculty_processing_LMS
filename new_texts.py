import psycopg2


def fetch_videos_text(video_id: str, output_file: str = "transcripts_new_two.txt"):
    
    adhoc_db = psycopg2.connect(dbname="piruby_automation", user="postgres", host="164.52.194.25",
                                password="piruby@157", port="5432")
    
    adhoc_cursor = adhoc_db.cursor()
    
    adhoc_query = "SELECT text FROM new_vimeo_master_m WHERE video_id = %s"
    #adhoc_query = "SELECT text FROM cs_ee_5m_test WHERE video_id = %s order by _offset"
    adhoc_cursor.execute(adhoc_query, (video_id,))
    transcript_texts = adhoc_cursor.fetchall()

    # Write all non-empty texts to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for row in transcript_texts:
            text = row[0]
            if text:
                f.write(text.strip() + "\n")
                
    # Close connections
    adhoc_cursor.close()
    adhoc_db.close()

    print(f"Transcript saved to {output_file}")
    

if __name__ == "__main__":
    # Example usage
    video_id = "982394038"
    #video_id = "8yvBqhm92xA"
    fetch_videos_text(video_id)
    