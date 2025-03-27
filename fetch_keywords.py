import psycopg2

def fetch_keywords(video_id):
    adhoc_db = psycopg2.connect(dbname="piruby_automation", user="postgres", host="164.52.194.25",
                                password="piruby@157", port="5432")
    adhoc_cursor = adhoc_db.cursor()
    
    query = """SELECT critical_keywords FROM public.cs_ee_5m_test WHERE video_id = %s order by _offset;"""
    adhoc_cursor.execute(query, (video_id,))
    result = adhoc_cursor.fetchall()

    print("Critical keywords:", result)
    
    adhoc_cursor.close()
    adhoc_db.close()
    return result


keywords = fetch_keywords('Oy4duAOGdWQ')