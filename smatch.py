import requests
import json
import re


def semantic_smart_answer(student_answer: str, question: str, answer: str, details: int = 0):

    headers = {'Content-Type': 'application/json'}

    # prompt = f"Given the question by trainer-{question} and the correct answer provided by trainer-{answer}, can you rate in percentage only with its symbol donot provide in float the answer match to the below student answer: {student_answer} provide the response in the format: Answer Match: <percentage value>, Key Phrases Missing in Student Answer (Present in Trainer Answer): keyword1, keyword2, Key Phrases Mentioned in Student Answer (Not in Trainer Answer): keywordA, keywordB, Reasons: <valid points about student_answer compared to trainer question and answer>.(Limit the number of keywords to 1-3 for each list.)"
    # prompt = f"The following question is given by a faculty - {question}. The correct answer to the above question provided by the faculty is - {answer}. Rate in percentage the match to the correct answer as above to the answer provided by the student as below -  {student_answer}. Provide the answer match as integer only. Provide the response in the format: Answer Match: <percentage value>, Key Phrases Missing in Student Answer Present in Trainer Answer: keyword1, keyword2, Key Phrases Mentioned in Student Answer Not in Trainer Answer: keywordA, keywordB, Reasons: <valid points about student_answer compared to trainer question and answer>. Limit the number of keywords to 1-3 for each list. In Key headings it should not contain any brackets in the headings."
    # prompt = f"The following question is given by a faculty - {question}. The correct answer to the above question provided by the faculty is - {answer}. Perform a Semantic Comparison and provide in percentage how close the student answer provided below is close to the correct answer above -  {student_answer}. Provide the response in the format: Answer Match: <percentage value>. , Key Phrases Missing in Student Answer Present in Trainer Answer: keyword1, keyword2, Key Phrases Mentioned in Student Answer Not in Trainer Answer: keywordA, keywordB, Reasons: <valid points about student_answer compared to trainer question and answer>. Limit the number of keywords to 1-3 for each list. Headings should be only Answer Match, Key Phrases Missing in Student Answer Present in Trainer Answer, Key Phrases Mentioned in Student Answer Not in Trainer Answer and Reasons. In Key headings it should not contain any brackets in the headings. Headings should not be in bold. Provide only Answer Match and do not provide any other explanations."
    # prompt = f"Question: \"{question}\". Correct Answer: \"{answer}\". Student's Answer: \"{student_answer}\". Calculate the semantic match score between the Student's Answer and Correct Answer expressed as a percentage. The score should measure how similar the student_answer is to correct_answer.  If less similar words provide less score. Provide the response in the format: Answer Match: <percentage value>."
    prompt = f"Question: \"{question}\". Correct Answer: \"{answer}\". Student's Answer: \"{student_answer}\". Calculate the semantic match score between student_answer and correct_amswer keeping in context the question, all specified above.  If less similar words provide less score. Don't compare with Question. Compare only student_answer and correct_answer. Provide Less Score If it partially matched between student_answer and correct_answer. Provide the response in the format: Answer Match: <percentage value>. Do not provide any explanations. Provide only the response in the format: Answer Match: <percentage value>."
    # print("Prompt:", prompt)

    payload = json.dumps({"prompt": prompt})

    try:
        response = requests.post("http://164.52.212.233:8010/pi-chat-prod", data=payload, headers=headers, timeout=500)
        # print(response.text)

        if response.status_code == 200:
            # Assuming the response contains the course description directly in the 'content' field
            outer_layer = json.loads(response.text)  # Parse the outer JSON object
            smart_answer = outer_layer.get('content', 'No content available')  # Get the content

            # Regular expression patterns
            percentage_pattern = r"(\d+)"
            # concepts_missing_pattern = r"Key Phrases Missing in Student Answer \(Present in Trainer Answer\)|Key Phrases Missing in Student Answer Present in Trainer Answer:\s*([\s\S]*?),\s*\n"
            # concepts_additional_pattern = r"Key Phrases Mentioned in Student Answer \(Not in Trainer Answer\)|Key Phrases Mentioned in Student Answer Not in Trainer Answer:\s*([\s\S]*?),\s*\n"
            concepts_missing_pattern = r"Key Phrases Missing in Student Answer \(Present in Trainer Answer\)|Key Phrases Missing in Student Answer Present in Trainer Answer:\s*([\s\S]*?)(?:,|$)"
            concepts_additional_pattern = r"Key Phrases Mentioned in Student Answer \(Not in Trainer Answer\)|Key Phrases Mentioned in Student Answer Not in Trainer Answer:\s*([\s\S]*?)(?:,|$)"
            reasons_pattern = r"Reasons:\s*([\s\S]*?)$"

            # Extract percentage rating
            match_percentage = re.search(percentage_pattern, smart_answer)
            # rating_student_ans = int(match_percentage.group(1)) if match_percentage else 0
            rating_student_ans = f"{match_percentage.group(1)}%" if match_percentage else "0%"

            # Extract missing concepts
            match_missing_concepts = re.search(concepts_missing_pattern, smart_answer)
            missing_concepts = [phrase.strip() for phrase in match_missing_concepts.group(1).split(',') if phrase.strip()] if match_missing_concepts else []

            # Extract additional concepts
            match_additional_concepts = re.search(concepts_additional_pattern, smart_answer)
            additional_concepts = [phrase.strip() for phrase in match_additional_concepts.group(1).split(',') if phrase.strip()] if match_additional_concepts else []

            # Extract reasons
            match_reasons = re.search(reasons_pattern, smart_answer)
            reasons = match_reasons.group(1) if match_reasons else ""

            if details == 0:
                return json.dumps({'content': [{'answer_match': rating_student_ans}]})
            elif details == 1:
                return json.dumps({'content': [{'answer_match': rating_student_ans, 'missing_concepts': missing_concepts, 'additional_concepts': additional_concepts, 'reasons': reasons}]})

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
