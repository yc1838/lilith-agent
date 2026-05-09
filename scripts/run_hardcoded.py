"""Minimal runner for hardcoded GAIA Level 1 questions - no HuggingFace dependency."""
from __future__ import annotations
import json, logging, os, sys, dataclasses
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=root_dir / ".env", override=True)
except ImportError:
    pass

from lilith_agent.config import Config
from lilith_agent.runner import run_agent_on_questions

LEVEL1_QUESTIONS = [
    {"task_id":"8e867cd7-cff9-4e6c-867a-ff5ddc2550be","question":"How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.","expected_answer":"3","Level":"1","file_name":""},
    {"task_id":"2d83110e-a098-4ebb-9987-066c06fa42d0","question":".rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI","expected_answer":"right","Level":"1","file_name":""},
    {"task_id":"4fc2f1ae-8625-45b5-ab34-ad4433bc21f8","question":"Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?","expected_answer":"FunkMonk","Level":"1","file_name":""},
    {"task_id":"6f37996b-2ac7-44b0-8e68-6d28256631b4","question":"Given this table defining * on the set S = {a, b, c, d, e}\n\n|*|a|b|c|d|e|\n|---|---|---|---|---|---|\n|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\nprovide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order.","expected_answer":"a, b, c, d, e","Level":"1","file_name":""},
    {"task_id":"cabe07ed-9eca-40ea-8ead-410ef5e83f91","question":"What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?","expected_answer":"Swanson","Level":"1","file_name":""},
    {"task_id":"305ac316-eef6-4446-960a-92d80d542f82","question":"Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.","expected_answer":"Kuba","Level":"1","file_name":""},
    {"task_id":"f918266a-b3e0-4914-865d-4faa564f1aef","question":"What is the final numeric output from the attached Python code?","expected_answer":"0","Level":"1","file_name":"f918266a-b3e0-4914-865d-4faa564f1aef.py"},
    {"task_id":"3f57289b-8c60-48be-bd80-01f8099ca449","question":"How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?","expected_answer":"519","Level":"1","file_name":""},
    {"task_id":"840bfca7-4f7b-481a-8794-c560c340185d","question":"On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?","expected_answer":"80GSFC21M0002","Level":"1","file_name":""},
    {"task_id":"bda648d7-d618-4883-88f4-3466eabd860e","question":"Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.","expected_answer":"Saint Petersburg","Level":"1","file_name":""},
    {"task_id":"cf106601-ab4f-4af9-b045-5295fe67b37d","question":"What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.","expected_answer":"CUB","Level":"1","file_name":""},
    {"task_id":"a0c07678-e491-4bbc-8f0b-07405144218f","question":"Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.","expected_answer":"Uehara, Matsui","Level":"1","file_name":""},
    {"task_id":"7bd855d8-463d-4ed5-93ca-5fe35145f733","question":"The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.","expected_answer":"$89706.00","Level":"1","file_name":"7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"},
    {"task_id":"5a0c1adf-205e-4841-a666-7c3ef95def9d","question":"What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?","expected_answer":"Dimitris","Level":"1","file_name":""},
    {"task_id":"3cef3a44-215e-4aed-8e3b-b1e3f08063b7","question":"I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes to categorizing things. I need to add different foods to different categories on the grocery list, but if I make a mistake, she won't buy anything inserted in the wrong category. Here's the list I have so far:\n\nmilk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\nI need to make headings for the fruits and vegetables. Could you please create a list of just the vegetables from my list? If you could do that, then I can figure out how to categorize the rest of the list into the appropriate categories. But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the vegetable list, or she won't get them when she's at the store. Please alphabetize the list of vegetables, and place each item in a comma separated list.","expected_answer":"broccoli, celery, lettuce, sweet potatoes","Level":"1","file_name":""},
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--task-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "langchain_core", "openai", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = Config.from_env()
    
    if args.task_id:
        target_ids = [t.strip() for t in args.task_id.split(",")]
        questions = [q for q in LEVEL1_QUESTIONS if q["task_id"] in target_ids]
    else:
        questions = LEVEL1_QUESTIONS[:args.limit]

    if args.force:
        cp_root = Path(cfg.checkpoint_dir)
        for q in questions:
            cp = cp_root / f"{q['task_id']}.json"
            if cp.exists():
                print(f"Forcing rerun: deleting checkpoint {cp}")
                cp.unlink()

    from lilith_agent.app import build_react_agent
    graph = build_react_agent(cfg)
    
    print(f"Running agent on {len(questions)} question(s)...")
    answers = run_agent_on_questions(graph, questions, cfg.checkpoint_dir)
    answers_by_id = {a["task_id"]: a["submitted_answer"] for a in answers}

    correct = 0
    failed_ids = []
    def _norm(s): return str(s).strip().lower().replace("-", " ")
    
    for q in questions:
        tid = q["task_id"]
        got = answers_by_id.get(tid, "")
        expected = q.get("expected_answer", "")
        ok = _norm(got) == _norm(expected)
        if ok: correct += 1
        else: failed_ids.append(tid)
        print("-" * 70)
        print(f"task_id : {tid}")
        print(f"Q       : {q['question'][:120]}")
        print(f"expected: {expected}")
        print(f"got     : {got}")
        print(f"match   : {ok}")
    
    print("=" * 70)
    print(f"Score: {correct}/{len(questions)}")
    if failed_ids:
        ids_str = ",".join(failed_ids)
        print(f"Failed: {ids_str}")
        Path(".last_failures.txt").write_text(ids_str)

if __name__ == "__main__":
    main()
