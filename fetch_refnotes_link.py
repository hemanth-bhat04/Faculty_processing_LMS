import psycopg2


def fetchRefnotesLink(content_id):

    try:

        query = """
                   SELECT "refnotes_link"
                   FROM "public"."Lms_coursereferencenote"
                   WHERE "id" = %s;
                """
        cur_4.execute(query, (content_id,))
        result = cur_4.fetchall()
        # print(result)

        if result:
            return result[0][0]
        else:
            return None

    except psycopg2.Error as e:
         print("Psycopg2 error:", e)
    # Handle the error appropriately (e.g., rollback transaction, close connection, etc.)
    except Exception as e:
         print("Error:", e)
