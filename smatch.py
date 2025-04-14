
import requests
import json
import re


def semantic_smart_answer(student_answer: str, question: str, answer: str, details: int = 0):
    headers = {'Content-Type': 'application/json'}

    # Updated prompt to explicitly request all required fields
    prompt = (
        f"Question: \"{question}\". "
        f"Correct Answer: \"{answer}\". "
        f"Student's Answer: \"{student_answer}\". "
        "Analyze the semantic match between the student's answer and the correct answer. "
        "Provide the following details in JSON format: "
        "{"
        "\"answer_match\": \"<percentage value>\", "
        "\"missing_concepts\": [\"<list of key phrases missing in the student's answer,provide only the concepts that are technically related, if synonyms are present do not include it as missing_concepts>\"], "
        "\"additional_concepts\": [\"<list of key phrases present in the student's answer but not in the correct answer>\"], "
        "\"reasons\": \"<explanation of the semantic match>\""
        "}."
    )

    payload = json.dumps({"prompt": prompt})

    try:
        response = requests.post("http://164.52.212.233:8010/pi-chat-prod", data=payload, headers=headers, timeout=500)

        if response.status_code == 200:
            return response.text  # Return the raw response text
        else:
            return json.dumps({'error': f"Semantic Match Failed with status code {response.status_code}."})

    except requests.exceptions.RequestException as e:
        return json.dumps({'error': str(e)})


if __name__ == '__main__':
    # stu_answer = "Hi. Hello. Java is a object oriented programming language. Play another one function. Java is object oriented programming language"
    # ques = "What is Java?"
    # ans = "Java is a high-level programming language. It is based on the principles of object-oriented programming and can be used to develop large-scale applications"
    stu_answer = "Describe in how many ways can"
    ques = "Describe in how many ways can you create a singleton pattern?"
    ans = "There are 'two ways' of creating a Singleton pattern. 1. Early Instantiation It is responsible for the 'creation of instance' at 'load time'. 2 Lazy Instantiation it is responsible for the creation of instance when 'required'."

    result = semantic_smart_answer(student_answer=stu_answer, question=ques, answer=ans)
    print(result)
