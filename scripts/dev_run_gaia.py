"""Run Lilith Agent against a small slice of real GAIA questions.

Runs locally OR inside an HF Space. Uses the gated `gaia-benchmark/GAIA`
dataset, which is accessible without extra config when running inside an
authenticated HF Space (and with $HF_TOKEN locally).

Usage:
    python scripts/dev_run_gaia.py --limit 3 --level 1
    python scripts/dev_run_gaia.py --limit 1 --task-id c61d22de-5f6c-4958-a7f6-5e9707bd3466
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import json
from pathlib import Path

# Make `app.py` and `src` importable whether invoked from repo root or inside the Space.
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

try:
    from dotenv import load_dotenv  # noqa: E402
    
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    
except ImportError:
    pass

from lilith_agent.config import Config  # noqa: E402
from lilith_agent.gaia_dataset import GaiaDatasetClient  # noqa: E402
from lilith_agent.runner import run_agent_on_questions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to run. Set to -1 for all.")
    parser.add_argument("--level", type=str, default="1")
    parser.add_argument("--verbose", action="store_true", help="Enable LangChain debug logging")
    parser.add_argument("--config", type=str, default="2023_all")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--local", action="store_true", help="Run with local model defined in .env")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Model to run when not using --local (defaults to gemini provider)")
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run specific task_id(s) (comma-separated, ignores --limit/--level)",
    )
    parser.add_argument("--rerun-failed", action="store_true", help="Automatically rerun tasks that failed in the last run (from .last_failures.txt)")
    parser.add_argument("--force", action="store_true", help="Delete existing checkpoints for selected tasks before running")
    parser.add_argument("--cavemen", action="store_true", help="Enable Caveman Mode (ultra-terse communication)")
    parser.add_argument("--caveman-mode", type=str, default="full", choices=["lite", "full", "ultra"], help="Caveman Mode intensity level")
    parser.add_argument("--gemma4", action=argparse.BooleanOptionalAction, default=None, help="Use gemma-4-31b-it for all model tiers (completely free on Gemini API). CLI flag overrides environment variables.")
    parser.add_argument("--gemini3", action=argparse.BooleanOptionalAction, default=None, help="Use gemini-3-flash-preview for all model tiers. CLI flag overrides environment variables.")
    parser.add_argument("--hardcoded", action="store_true", help="Use hardcoded GAIA questions for testing instead of importing from Hub")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # suppress noisy third-party loggers unless verbose
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "langchain_core", "openai", "urllib3", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")

    from lilith_agent.app import build_react_agent  # noqa: WPS433

    cfg = Config.from_env()

    # Default to local-env behavior. The --model flag will only be used if GAIA_STRONG_MODEL is not set.
    if args.model != "gemini-3-flash-preview" or not cfg.strong_model:
        cfg = dataclasses.replace(
            cfg,
            cheap_provider="google",
            cheap_model=args.model,
            strong_provider="google",
            strong_model=args.model
        )

    if args.local:
        cfg = Config.from_env()

    if args.gemma4 is not None:
        use_gemma4 = args.gemma4
        source = "CLI: --gemma4" if args.gemma4 else "CLI: --no-gemma4"
    else:
        use_gemma4 = os.getenv("USE_GEMMA_4", "false").lower() == "true"
        source = "ENV: USE_GEMMA_4=TRUE" if use_gemma4 else None

    if args.gemini3 is not None:
        use_gemini3 = args.gemini3
        source_gemini3 = "CLI: --gemini3" if args.gemini3 else "CLI: --no-gemini3"
    else:
        use_gemini3 = os.getenv("USE_GEMINI_3", "false").lower() == "true"
        source_gemini3 = "ENV: USE_GEMINI_3=TRUE" if use_gemini3 else None

    if use_gemma4:
        banner_source = source or "DEFAULT: OFF"
        print("\n" + "!"*80, flush=True)
        print(f"!!! OVERRIDING ALL TIERED MODELS TO gemma-4-31b-it ({source}) !!!".center(80), flush=True)
        print("!"*80 + "\n", flush=True)
        cfg = dataclasses.replace(
            cfg,
            cheap_provider="google",
            cheap_model="gemma-4-31b-it",
            strong_provider="google",
            strong_model="gemma-4-31b-it",
            extra_strong_provider="google",
            extra_strong_model="gemma-4-31b-it"
        )
    elif use_gemini3:
        banner_source = source_gemini3 or "DEFAULT: OFF"
        print("\n" + "!"*80, flush=True)
        print(f"!!! OVERRIDING ALL TIERED MODELS TO gemini-3-flash-preview ({source_gemini3}) !!!".center(80), flush=True)
        print("!"*80 + "\n", flush=True)
        cfg = dataclasses.replace(
            cfg,
            cheap_provider="google",
            cheap_model="gemini-3-flash-preview",
            strong_provider="google",
            strong_model="gemini-3-flash-preview",
            extra_strong_provider="google",
            extra_strong_model="gemini-3-flash-preview"
        )

    cfg = dataclasses.replace(
        cfg,
        caveman=args.cavemen if args.cavemen else cfg.caveman,
        caveman_mode=args.caveman_mode if args.caveman_mode != "full" else cfg.caveman_mode
    )
    limit = args.limit if args.limit > 0 else None
    
    target_ids = []
    if args.rerun_failed:
        fail_file = Path(".last_failures.txt")
        if fail_file.exists():
            target_ids = fail_file.read_text().strip().split(",")
            target_ids = [tid.strip() for tid in target_ids if tid.strip()]
            print(f"Loaded {len(target_ids)} failed IDs from {fail_file}")

    if args.task_id:
        target_ids.extend([tid.strip() for tid in args.task_id.split(",") if tid.strip()])

    client = GaiaDatasetClient(
        config=args.config,
        split=args.split,
        level=None if target_ids else args.level,
        limit=None if target_ids else limit,
        token=token,
    )

    if args.hardcoded:
        # Load the hardcoded chunk specified by user previously
        questions = [{"task_id":"8e867cd7-cff9-4e6c-867a-ff5ddc2550be","question":"How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.","Level":"1","file_name":""},{"task_id":"a1e91b78-d3d8-4675-bb8d-62741b4b68a6","question":"In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?","Level":"1","file_name":""},{"task_id":"2d83110e-a098-4ebb-9987-066c06fa42d0","question":".rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI","Level":"1","file_name":""},{"task_id":"cca530fc-4052-43b2-b130-b30968d8aa44","question":"Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.","Level":"1","file_name":"cca530fc-4052-43b2-b130-b30968d8aa44.png"},{"task_id":"4fc2f1ae-8625-45b5-ab34-ad4433bc21f8","question":"Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?","Level":"1","file_name":""},{"task_id":"6f37996b-2ac7-44b0-8e68-6d28256631b4","question":"Given this table defining * on the set S = {a, b, c, d, e}\n\n|*|a|b|c|d|e|\n|---|---|---|---|---|---|\n|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\nprovide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order.","Level":"1","file_name":""},{"task_id":"9d191bce-651d-4746-be2d-7ef8ecadb9c2","question":"Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.\n\nWhat does Teal'c say in response to the question \"Isn't that hot?\"","Level":"1","file_name":""},{"task_id":"cabe07ed-9eca-40ea-8ead-410ef5e83f91","question":"What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?","Level":"1","file_name":""},{"task_id":"3cef3a44-215e-4aed-8e3b-b1e3f08063b7","question":"I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes to categorizing things. I need to add different foods to different categories on the grocery list, but if I make a mistake, she won't buy anything inserted in the wrong category. Here's the list I have so far:\n\nmilk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\nI need to make headings for the fruits and vegetables. Could you please create a list of just the vegetables from my list? If you could do that, then I can figure out how to categorize the rest of the list into the appropriate categories. But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the vegetable list, or she won't get them when she's at the store. Please alphabetize the list of vegetables, and place each item in a comma separated list.","Level":"1","file_name":""},{"task_id":"99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3","question":"Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3.\n\nIn your response, please only list the ingredients, not any measurements. So if the recipe calls for \"a pinch of salt\" or \"two cups of ripe strawberries\" the ingredients on the list would be \"salt\" and \"ripe strawberries\".\n\nPlease format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients.","Level":"1","file_name":"99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"},{"task_id":"305ac316-eef6-4446-960a-92d80d542f82","question":"Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.","Level":"1","file_name":""},{"task_id":"f918266a-b3e0-4914-865d-4faa564f1aef","question":"What is the final numeric output from the attached Python code?","Level":"1","file_name":"f918266a-b3e0-4914-865d-4faa564f1aef.py"},{"task_id":"3f57289b-8c60-48be-bd80-01f8099ca449","question":"How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?","Level":"1","file_name":""},{"task_id":"1f975693-876d-457b-a649-393859e79bf3","question":"Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order.","Level":"1","file_name":"1f975693-876d-457b-a649-393859e79bf3.mp3"},{"task_id":"840bfca7-4f7b-481a-8794-c560c340185d","question":"On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?","Level":"1","file_name":""},{"task_id":"bda648d7-d618-4883-88f4-3466eabd860e","question":"Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.","Level":"1","file_name":""},{"task_id":"cf106601-ab4f-4af9-b045-5295fe67b37d","question":"What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.","Level":"1","file_name":""},{"task_id":"a0c07678-e491-4bbc-8f0b-07405144218f","question":"Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.","Level":"1","file_name":""},{"task_id":"7bd855d8-463d-4ed5-93ca-5fe35145f733","question":"The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.","Level":"1","file_name":"7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"},{"task_id":"5a0c1adf-205e-4841-a666-7c3ef95def9d","question":"What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?","Level":"1","file_name":""}]
    else:
        questions = client.get_questions()
        if target_ids:
            questions = [q for q in questions if q["task_id"] in target_ids]

    if not questions:
        print("No matching questions.")
        return

    if args.force:
        checkpoint_root = Path(cfg.checkpoint_dir)
        for q in questions:
            checkpoint_path = checkpoint_root / f"{q['task_id']}.json"
            if checkpoint_path.exists():
                print(f"Forcing rerun: deleting checkpoint {checkpoint_path}")
                checkpoint_path.unlink()

    graph = build_react_agent(cfg)

    print(f"Running agent on {len(questions)} question(s)...")
    answers = run_agent_on_questions(graph, questions, cfg.checkpoint_dir, client=client)
    answers_by_id = {a["task_id"]: a["submitted_answer"] for a in answers}

    correct = 0
    failed_ids = []

    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("-", " ")

    for q in questions:
        tid = q["task_id"]
        got = answers_by_id.get(tid, "")
        expected = q.get("expected_answer", "")
        ok = _norm(got) == _norm(expected) if expected else None
        if ok:
            correct += 1
        elif ok is False:
            failed_ids.append(tid)

        print("-" * 70)
        print(f"task_id : {tid}")
        print(f"level   : {q.get('Level')}")
        print(f"Q       : {q['question'][:200]}")
        if expected:
            print(f"expected: {expected}")
            print(f"got     : {got}")
            print(f"match   : {ok}")
        else:
            print(f"got     : {got}")

    print("=" * 70)
    if any(q.get("expected_answer") for q in questions):
        print(f"Score: {correct}/{len(questions)}")
        if failed_ids:
            ids_str = ",".join(failed_ids)
            print(f"\nFailed Task IDs (for easy rerun):")
            print(ids_str)
            Path(".last_failures.txt").write_text(ids_str)
            print(f"\nWritten to .last_failures.txt")
        elif target_ids:
            if Path(".last_failures.txt").exists():
                 Path(".last_failures.txt").unlink()
                 print("\nCleared .last_failures.txt as all requested tasks passed.")
    else:
        print(f"Completed {len(questions)} items. No expected answers provided in the selected data.")


if __name__ == "__main__":
    main()
