
import argparse
import re
import time
import torch
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = "google/flan-t5-xl"
DEVICE = torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parent.parent
LLM_DIR = ROOT_DIR / "src" / "llm"

CONTEXT_DIR = LLM_DIR / "contexts"
EVALUATION_DIR = LLM_DIR / "evaluation"

OUTPUT_DIR = (
    ROOT_DIR / "reproduction" / "outputs" / "flan-t5"
)


# --------------------------------------------------
# 2. Load original context
# --------------------------------------------------

def load_context(context: str) -> str:

    if context:
        context_file = CONTEXT_DIR / f"{context}.txt"

        with open(context_file, "r", encoding="utf-8") as f:
            context = f.read()

    return context


# --------------------------------------------------
# 3. Generate zero-shot prompt
# --------------------------------------------------

def get_prompt(context: str, question: str) -> str:

    if context:
        return f"{context}\n{question}"

    return question


# --------------------------------------------------
# 4. Load 2022 questions and ground-truth answers
# --------------------------------------------------

def load_answers(year: str) -> dict:

    answers = {}

    input_file = (
        EVALUATION_DIR
        / f"misinfo-resources-{year}"
        / "topics"
        / f"misinfo-{year}-topics.xml"
    )

    print(f"Loading questions from: {input_file}")

    root = ET.parse(input_file).getroot()

    for topic in root.findall("topic"):

        query = topic.find("question").text
        answer = topic.find("answer").text

        answers[query.rstrip()] = answer

    return answers


# --------------------------------------------------
# 5. Prediction and evaluation
# --------------------------------------------------

def predict(
    context: str,
    answers: dict,
    prompt_type: str,
    year: str,
    model,
    tokenizer,
    limit: int = None
) -> float:

    if limit is not None:
        answers = dict(list(answers.items())[:limit])

    n = len(answers)
    hits = 0

    if n == 0:
        raise ValueError("No questions found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_type = f"test_{limit}" if limit else "full"

    outputfile = (
        OUTPUT_DIR
        / f"{prompt_type}_{year}_{run_type}.txt"
    )

    print(f"Output file: {outputfile}")
    print(f"Number of questions: {n}")

    # Prevent overwriting previous results
    if outputfile.exists():
        raise FileExistsError(
            f"Output file already exists: {outputfile}"
        )

    start_time = time.perf_counter()

    with open(outputfile, "x", encoding="utf-8") as f:

        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("Precision: FP32\n")
        f.write("Decoding: Greedy\n")
        f.write(f"Prompt type: {prompt_type}\n")
        f.write(f"Number of questions: {n}\n")
        f.write(f"Generation config: {model.generation_config}\n")
        f.write("do_sample=False, num_beams=1\n\n")

        for i, (question, answer) in enumerate(
            answers.items(), start=1
        ):

            prompt = get_prompt(context, question)

            print(f"\nQuestion {i}/{n}")
            print(prompt)

            f.write(f"\nQuestion {i}/{n}\n")
            f.write(prompt + "\n")

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.inference_mode():

                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1
                )

            response = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            response = response[0].lower()

            print(f"Response: {response}")
            print(f"Ground truth: {answer}")

            f.write(f"Response: {response}\n")
            f.write(f"Ground truth: {answer}\n")

            # Original authors' evaluation method
            m = re.search(r'\byes\b|\bno\b', response)

            if m and m.group().strip() == answer:
                hits += 1
                f.write("Correct: Yes\n")
            else:
                f.write("Correct: No\n")

            print(f"Correct answers so far: {hits}/{i}")

            f.flush()

        accuracy = hits / n
        runtime = time.perf_counter() - start_time

        f.write("\n--- Final results ---\n")
        f.write(f"Correct answers: {hits}/{n}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Runtime (seconds): {runtime:.2f}\n")
        f.write(f"Runtime (minutes): {runtime / 60:.2f}\n")

    print("\n--- Final results ---")
    print(f"Correct answers: {hits}/{n}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Runtime: {runtime / 60:.2f} minutes")
    print(f"Results saved to: {outputfile}")

    return accuracy


# --------------------------------------------------
# 6. Main experiment
# --------------------------------------------------

if __name__ == "__main__":

    total_start = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context",
        choices=["no_context", "nonexpert", "expert"],
        default="no_context"
    )

    parser.add_argument(
        "--year",
        choices=["2022"],
        default="2022"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    if args.limit is not None and not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50.")

    context_name = (
        "" if args.context == "no_context"
        else args.context
    )

    context = load_context(context_name)
    answers = load_answers(args.year)

    print(f"Total questions loaded: {len(answers)}")

    if len(answers) != 50:
        raise ValueError(
            f"Expected 50 questions, found {len(answers)}."
        )

    run_type = f"test_{args.limit}" if args.limit else "full"

    outputfile = (
        OUTPUT_DIR
        / f"{args.context}_{args.year}_{run_type}.txt"
    )

    print(f"Planned output file: {outputfile}")

    if outputfile.exists():
        raise FileExistsError(
            f"Output file already exists: {outputfile}"
        )

    print(f"\nLoading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("Precision: FP32")

    model_start = time.perf_counter()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.eval()

    print(f"Actual model precision: {model.dtype}")
    print(f"Actual model device: {next(model.parameters()).device}")

    model_runtime = time.perf_counter() - model_start

    print(f"Model loading time: {model_runtime / 60:.2f} minutes")

    predict(
        context=context,
        answers=answers,
        prompt_type=args.context,
        year=args.year,
        model=model,
        tokenizer=tokenizer,
        limit=args.limit
    )

    total_runtime = time.perf_counter() - total_start

    with open(outputfile, "a", encoding="utf-8") as f:
        f.write(
            f"Model loading time (seconds): {model_runtime:.2f}\n"
        )
        f.write(
            f"Total runtime (seconds): {total_runtime:.2f}\n"
        )

    print(f"Total runtime: {total_runtime / 60:.2f} minutes")
