import re
import openai
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import tiktoken
from ollama import chat
from ollama import ChatResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]


API_KEY=""

'''
This method loads a context from file.
Args:
    - path: path to the file containing the context
Return:
    - returns the context
'''
def load_context(context: str) -> str:
  if context == "":
    return ""

  allowed_contexts = {
    "nonexpert": "nonexpert.txt",
    "expert": "expert.txt"
  }

  if context not in allowed_contexts:
    raise ValueError("Unsupported context type")

  context_path = BASE_DIR / "contexts" / allowed_contexts[context]

  with open(context_path) as f:
    return f.read()


'''
This method generates a formatted prompt to input the model
Args:
    - message: main message to be included in the prompt
Return:
    - returns the formatted prompt
'''
def get_prompt(
    context: str,
    system:bool,
    question:str
) -> str:
  if system:
    prompt = (
        f'You are a helpful medical assistant.\n'
        f'{context}\n'
        f'Q: {question} A: \n'
    )
  else:
    prompt = (
        f'{context}\n'
        # f'Q: Will wearing an ankle brace help heal achilles tendonitis?\n'
        # f'A: No.\n'
        # f'Q: Does yoga improve the management of asthma?\n'
        # f'A: Yes.\n'
        # f'Q: Is starving a fever effective?\n'
        # f'A: No.\n'
        f'{question} \n'
    )

  return prompt


'''
This method loads the ground truth values for health questions from file.
Args:
    - path: path to the file containing the ground truth questions
Return:
    - returns a dictionary with the questions as keys and the answers as values
'''
def load_answers(
  year:int
)-> dict:
  answers = {}

  allowed_datasets = {
    "2020": "misinfo-resources-2020/topics/misinfo-2020-topics.xml",
    "2021": "misinfo-resources-2021/topics/misinfo-2021-topics.xml",
    "2022": "misinfo-resources-2022/topics/misinfo-2022-topics.xml"
  }

  year = str(year)

  if year not in allowed_datasets:
    raise ValueError("Unsupported dataset year")

  dataset_path = BASE_DIR / "evaluation" / allowed_datasets[year]

  root = ET.parse(dataset_path).getroot()
  for topic in root.findall('topic'):
    if str(year)=="2022":
      query = topic.find("question").text 
      answer = topic.find("answer").text 
    elif str(year)=="2021":
      query = topic.find("description").text 
      answer = topic.find("stance").text
      if answer=="helpful":
        answer = 'yes'
      else:
        answer = 'no'
    elif str(year)=="2020":
      query = topic.find("description").text 
      answer = topic.find("answer").text
    answers[query.rstrip()] = answer
  return answers


'''
This method evaluates model's answer against ground truth values
Args:
    - response: model's answer to the question
    - answer: ground truth value
Return:
    - returns 1 if there is a math and 0 otherwise
'''
def evaluate(
    response:str,
    answer:str
)-> int:
    hit=0
    response = response.lower()

    m = re.search(r'\byes\b|\bno\b', response)
    if m and m.group().strip()==answer:
        hit=1
    return hit

'''
This method asks a LLM for the answer to a suite of health-related questions
Args:
    - model: model_name
    - prompt: context
    - answers: dict containing the questions plus their answers
Return:
    - returns the accuracy value for all the answers
'''
def ask(
  model:str,
  eval:dict,
  context:str,
  expert: str, 
  year: int,
  syst:bool=False,
)-> float:
  number_questions = len(eval)
  
  allowed_files = {
    ("", "2020"): "2020",
    ("", "2021"): "2021",
    ("", "2022"): "2022",
    ("nonexpert", "2020"): "nonexpert2020",
    ("nonexpert", "2021"): "nonexpert2021",
    ("nonexpert", "2022"): "nonexpert2022",
    ("expert", "2020"): "expert2020",
    ("expert", "2021"): "expert2021",
    ("expert", "2022"): "expert2022",
  }

  key = (expert, str(year))

  if key not in allowed_files:
    raise ValueError("Unsupported context or dataset year")

  outputfile = allowed_files[key] + (
    "_s.txt" if syst else ".txt"
  )
  import os
  outdir = Path(os.environ.get(
    "OUTDIR",
    str(REPO_DIR / "reproduction" / "outputs" / "llama3")
  ))
  outdir = outdir.resolve()

  outdir.relative_to(
    (REPO_DIR / "reproduction" / "outputs" / "llama3").resolve()
  )

  output_path = (outdir / outputfile).resolve()

  if output_path.parent != outdir:
    raise ValueError("Invalid output file path")

  outdir.mkdir(parents=True, exist_ok=True)

  with open(output_path, 'x') as f:
    hits = 0
    for k, v in eval.items():
        prompt = get_prompt(context, syst, k)
        print(prompt)
        f.write(prompt+'\n')

        response: ChatResponse = chat(model=model, messages=[
          {
            'role': 'user',
            'content': prompt + '\nYou are only allowed to respond with either \'yes\' or \'no\'. Do not provide any other text or explanation.',
          },
          
        ],options={'temperature': 0, 'num_ctx': 8192})  # reproduction: num_ctx set explicitly

        result = response['message']['content']

        f.write(result+'\n')
        print(result)
        hit = evaluate(result,v)
        if hit!=0:
            hits+=hit
            f.write(str(hits)+'\n')
            print(hits)

    accuracy =  hits/number_questions
    f.write("Accuracy:"+ str(accuracy))
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs='?', default="thewindmom/llama3-med42-8b") #  llama3.1:8b-instruct-q8_0
    parser.add_argument(
        "context",
        nargs="?",
        default="",
        choices=["", "nonexpert", "expert"]
    )

    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        default=2020,
        choices=[2020, 2021, 2022]
    )
    args = parser.parse_args()
    context = load_context(args.context)
    eval = load_answers(args.year)
    print(ask(args.model, eval, context, args.context, args.year))
