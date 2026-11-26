import re
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOPICS_FILE = (
    ROOT / "src/llm/evaluation/misinfo-resources-2022"
    / "topics/misinfo-2022-topics.xml"
)

AUTHORS_DIR = ROOT / "src/llm/outputs/zero-shot/flan-t5"
OUR_DIR = ROOT / "reproduction/outputs/flan-t5"


def load_topics():
    topics = {}

    for topic in ET.parse(TOPICS_FILE).getroot().findall("topic"):
        question = topic.find("question").text.rstrip()
        answer = topic.find("answer").text.strip().lower()
        topics[question] = answer

    return topics


def extract_prediction(response):
    match = re.search(r"\byes\b|\bno\b", response.lower())
    return match.group() if match else "NONE"


def parse_authors(path, topics):
    lines = path.read_text(encoding="utf-8").splitlines()
    responses = {}

    for i, line in enumerate(lines):
        question = line.strip()

        if question in topics:
            response_lines = []

            for next_line in lines[i + 1:]:
                if next_line.startswith("Accuracy:"):
                    break

                response_lines.append(next_line)

            # Remove the cumulative correct-answer counter
            if response_lines and response_lines[-1].strip().isdigit():
                response_lines.pop()

            if question in responses:
                raise ValueError(f"Duplicate question: {question}")

            responses[question] = "\n".join(response_lines).strip()

    return responses


def parse_reproduction(path, topics):
    lines = path.read_text(encoding="utf-8").splitlines()
    responses = {}

    for i, line in enumerate(lines):
        question = line.strip()

        if question in topics:
            if i + 1 >= len(lines):
                raise ValueError(f"Missing response: {question}")

            if not lines[i + 1].startswith("Response:"):
                raise ValueError(f"Unexpected format: {question}")

            response_lines = [
                lines[i + 1].removeprefix("Response:").strip()
            ]

            for next_line in lines[i + 2:]:
                if next_line.startswith("Ground truth:"):
                    break

                response_lines.append(next_line)
            else:
                raise ValueError(f"Missing response boundary: {question}")

            if question in responses:
                raise ValueError(f"Duplicate question: {question}")

            responses[question] = "\n".join(response_lines).strip()

    return responses


def compare(context):
    filenames = {
        "no_context": "2022.txt",
        "nonexpert": "nonexpert2022.txt",
        "expert": "expert2022.txt"
    }

    topics = load_topics()

    authors = parse_authors(
        AUTHORS_DIR / filenames[context], topics
    )

    ours = parse_reproduction(
        OUR_DIR / f"{context}_2022_full.txt", topics
    )

    n = len(topics)

    # Validate completeness before calculating metrics
    if n != 50 or set(authors) != set(topics) or set(ours) != set(topics):
        raise ValueError(
            f"Incomplete data: topics={n}, "
            f"authors={len(authors)}, ours={len(ours)}"
        )

    exact_matches = 0
    prediction_matches = 0
    correctness_matches = 0

    authors_correct = 0
    ours_correct = 0

    authors_invalid = 0
    ours_invalid = 0

    differences = []

    for i, (question, truth) in enumerate(topics.items(), 1):

        author_response = authors[question]
        our_response = ours[question]

        author_pred = extract_prediction(author_response)
        our_pred = extract_prediction(our_response)

        exact_matches += author_response == our_response
        prediction_matches += author_pred == our_pred

        authors_correct += author_pred == truth
        ours_correct += our_pred == truth

        correctness_matches += (
            (author_pred == truth) == (our_pred == truth)
        )

        authors_invalid += author_pred == "NONE"
        ours_invalid += our_pred == "NONE"

        if author_response != our_response:
            differences.append(
                (i, question, author_response, our_response,
                 author_pred, our_pred, truth)
            )

    print(f"\n--- Comparison: {context} ---")
    print(f"Questions: {n}")

    print("\n--- Accuracy ---")
    print(f"Authors: {authors_correct}/{n} ({authors_correct/n:.2%})")
    print(f"Ours: {ours_correct}/{n} ({ours_correct/n:.2%})")
    print(f"Difference: {(ours_correct-authors_correct)/n*100:.2f} pp")

    print("\n--- Agreement ---")
    print(f"Exact responses: {exact_matches}/{n} ({exact_matches/n:.2%})")
    print(f"Predictions: {prediction_matches}/{n} ({prediction_matches/n:.2%})")
    print(f"Correctness: {correctness_matches}/{n} ({correctness_matches/n:.2%})")

    print("\n--- Invalid responses ---")
    print(f"Authors: {authors_invalid}")
    print(f"Ours: {ours_invalid}")

    print("\n--- Response differences ---")

    if not differences:
        print("No differences found.")

    for i, question, a, o, ap, op, truth in differences:
        print(f"\nQuestion {i}: {question}")
        print(f"Authors: {a}")
        print(f"Ours: {o}")
        print(f"Predictions: authors={ap}, ours={op}")
        print(f"Ground truth: {truth}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context",
        choices=["no_context", "nonexpert", "expert"],
        required=True
    )

    args = parser.parse_args()
    compare(args.context)