"""
Run replacement models (university API) for the retired OpenAI models in
Fernandez-Pichel et al. (npj Digital Medicine, 2025).

Replacements:
  GPT-4          -> GPT OSS 120b      (open-weight)
  GPT-3.5-turbo  -> ChatGPT 5.4-mini  (closed)

Prompt: identical to the authors' src/llm/openai_models.py
  - system message "You are a chatbot"
  - user prompt "{context}\n[3 few-shot examples]\nQ: {question} A:\n"
  - --shots 0 removes the three examples, everything else stays the same

Necessary deviations from the original (document these in the report):
  - The original forced a one-token answer with logit_bias (+50 for "yes"/"no")
    and max_tokens=1 at temperature 0. The university API models are reasoning
    models: they do not accept temperature, and a 1-token limit leaves no room
    for reasoning. So we let the model answer freely and extract the first
    "yes"/"no" with the authors' own regex (same as their evaluate()).
  - No temperature control, so we run several repeats and report the spread.
  - Current OpenAI Python SDK (>=1.0) instead of the retired openai.ChatCompletion.

Setup (never commit your key):
  export UNI_API_BASE_URL="https://.../v1"
  export UNI_API_KEY="..."

Usage (from the repo root):
  python reproduction/run_api_models.py --model <api-model-id> --label gpt-oss-120b --context expert --shots 3 --repeat 1
  python reproduction/run_api_models.py --model <api-model-id> --label gpt-oss-120b --context expert --shots 3 --limit 3   # quick test
"""
import argparse, csv, json, os, platform, re, sys, time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]
LLM = REPO / "src" / "llm"

FEW_SHOT = (
    "Q: Will wearing an ankle brace help heal achilles tendonitis?\n"
    "A: No.\n"
    "Q: Does yoga improve the management of asthma?\n"
    "A: Yes.\n"
    "Q: Is starving a fever effective?\n"
    "A: No.\n"
)


def load_context(name):
    if not name:
        return ""
    return (LLM / "contexts" / f"{name}.txt").read_text(encoding="utf-8")


# Exact sentence the authors append in src/llm/llama3.py (their method for models without logit_bias)
STRICT = "\nYou are only allowed to respond with either 'yes' or 'no'. Do not provide any other text or explanation."


def get_prompt(context, question, shots):
    # Same structure as get_prompt(system=False) in openai_models.py
    examples = FEW_SHOT if shots == 3 else ""
    return f"{context}\n{examples}Q: {question} A:\n"


def load_answers(year):
    root = ET.parse(LLM / "evaluation" / f"misinfo-resources-{year}" / "topics" / f"misinfo-{year}-topics.xml").getroot()
    answers = {}
    for t in root.findall("topic"):
        if year == 2022:
            q, a = t.find("question").text, t.find("answer").text
        elif year == 2021:
            q, a = t.find("description").text, ("yes" if t.find("stance").text == "helpful" else "no")
        else:
            q, a = t.find("description").text, t.find("answer").text
        answers[q.rstrip()] = a.strip().lower()   # dict de-duplicates, same as the authors
    return answers


def extract(response):
    # Same regex as evaluate() in openai_models.py
    m = re.search(r"\byes\b|\bno\b", response.lower())
    return m.group() if m else "NONE"


TOKEN_PARAM = {"name": "max_tokens"}  # GPT-5 models may require max_completion_tokens instead


def ask(client, model, prompt, max_tokens, retries=5):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a chatbot"},
                          {"role": "user", "content": prompt}],
                **{TOKEN_PARAM["name"]: max_tokens},
            )
            return (r.choices[0].message.content or ""), getattr(r, "model", model)
        except Exception as e:
            msg = str(e)
            if "max_completion_tokens" in msg and TOKEN_PARAM["name"] == "max_tokens":
                print("  API wants max_completion_tokens, switching", flush=True)
                TOKEN_PARAM["name"] = "max_completion_tokens"
                continue
            if getattr(e, "status_code", None) in (400, 401, 403, 404):
                raise SystemExit(f"API rejected the request ({e.status_code}): {msg[:300]}")
            wait = 2 ** attempt * 5  # rate limits, timeouts, server errors
            print(f"  API error ({type(e).__name__}: {msg[:120]}), retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("API failed after retries")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model ID exactly as the university API expects it")
    p.add_argument("--label", required=True, help="folder name, e.g. gpt-oss-120b or gpt-5.4-mini")
    p.add_argument("--context", default="", choices=["", "nonexpert", "expert"])
    p.add_argument("--shots", type=int, default=0, choices=[0, 3])
    p.add_argument("--year", type=int, default=2022, choices=[2020, 2021, 2022])
    p.add_argument("--repeat", type=int, default=1, help="repeat number (1, 2, 3, ...)")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (for testing)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--strict", action="store_true", help="append the authors' yes/no instruction from llama3.py")
    a = p.parse_args()

    base_url, key = os.environ.get("UNI_API_BASE_URL"), os.environ.get("UNI_API_KEY")
    if not base_url or not key:
        sys.exit("Set UNI_API_BASE_URL and UNI_API_KEY first.")
    client = OpenAI(base_url=base_url, api_key=key, timeout=180)

    context = load_context(a.context)
    answers = load_answers(a.year)
    items = list(answers.items())[: a.limit] if a.limit else list(answers.items())

    prompt_name = a.context or "nocontext"
    run = f"{prompt_name}{a.year}_{a.shots}shot_rep{a.repeat}" + ("_strict" if a.strict else "") + (f"_test{a.limit}" if a.limit else "")
    outdir = REPO / "reproduction" / "outputs" / "api" / a.label
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path, log_path = outdir / f"{run}.csv", outdir / f"{run}_log.json"
    if csv_path.exists():
        sys.exit(f"{csv_path} already exists. Use another --repeat number.")

    start = time.time()
    hits = noans = 0
    served_model = a.model
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "question", "ground_truth", "prediction", "correct", "seconds", "response"])
        for i, (q, gt) in enumerate(items, 1):
            t0 = time.time()
            response, served_model = ask(client, a.model, get_prompt(context, q, a.shots) + (STRICT if a.strict else ""), a.max_tokens)
            pred = extract(response)
            ok = int(pred == gt)
            hits += ok
            noans += pred == "NONE"
            w.writerow([i, q, gt, pred, ok, round(time.time() - t0, 2), response.strip()])
            f.flush()
            print(f"{i:2}/{len(items)} gt={gt:3} pred={pred:4} {'OK ' if ok else '   '} | {q[:60]}", flush=True)

    acc = hits / len(items)
    log = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model_requested": a.model, "model_served": served_model, "label": a.label,
        "base_url": base_url, "prompt": prompt_name, "shots": a.shots, "year": a.year,
        "repeat": a.repeat, "questions": len(items), "correct": hits, "no_yes_no_answer": noans,
        "accuracy": round(acc, 4), "runtime_seconds": round(time.time() - start, 1),
        "max_tokens": a.max_tokens, "system_message": "You are a chatbot", "strict_instruction": a.strict,
        "deviations": ["no logit_bias", "no max_tokens=1", "no temperature control (unsupported)", "openai SDK >= 1.0"],
        "python": platform.python_version(),
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nAccuracy: {acc:.2f} ({hits}/{len(items)}), no yes/no: {noans}, runtime {log['runtime_seconds']}s")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
