import requests
import json
import re


def semantic_smart_answer(student_answer: str, question: str, answer: str, details: int = 0):
    headers = {'Content-Type': 'application/json'}

    # Optimized prompt
    prompt = (
        f"Question: \"{question}\". "
        f"Correct Answer: \"{answer}\". "
        f"Student's Answer: \"{student_answer}\". "
        "Analyze the semantic match between the student's answer and the correct answer. "
        "Focus on technical relevance and ensure that synonyms or paraphrased concepts are treated as matches. "
        "Treat identical content as a 100% match, and ignore minor differences such as phrasing, punctuation, or formatting. "
        "Provide the following details in JSON format: "
        "{"
        "\"answer_match\": \"<percentage value>\", "
        "\"missing_concepts\": [\"<list of key phrases or concepts missing in the student's answer that are technically relevant. Do not include synonyms or paraphrased concepts as missing.>\"], "
        "\"additional_concepts\": [\"<list of key phrases or concepts present in the student's answer but not in the correct answer. Include only technically relevant concepts.>\"], "
        "\"reasons\": \"<detailed explanation of the semantic match, including why certain concepts were considered missing or additional. Provide reasoning for the percentage match.>\""
        "}."
    )

    payload = json.dumps({"prompt": prompt})

    try:
        response = requests.post("http://164.52.212.233:8010/pi-chat-prod", data=payload, headers=headers, timeout=500)

        if response.status_code == 200:
            response_data = json.loads(response.text)

            # Post-process the response to ensure identical content gets 100% match
            if (
                response_data.get("missing_concepts") == [] and
                response_data.get("additional_concepts") == []
            ):
                response_data["answer_match"] = "100%"  # Force 100% match for identical content

            return json.dumps(response_data)  # Return the modified response as JSON

        else:
            return json.dumps({'error': f"Semantic Match Failed with status code {response.status_code}."})

    except requests.exceptions.RequestException as e:
        return json.dumps({'error': str(e)})


if __name__ == '__main__':
    # Example inputs
    stu_answer = "Describe in how many ways can"
    ques = "Describe in how many ways can you create a singleton pattern?"
    ans = "There are 'two ways' of creating a Singleton pattern. 1. Early Instantiation It is responsible for the 'creation of instance' at 'load time'. 2 Lazy Instantiation it is responsible for the creation of instance when 'required'."

    # Call the semantic_smart_answer function
    result = semantic_smart_answer(student_answer=stu_answer, question=ques, answer=ans)
    print(result)
