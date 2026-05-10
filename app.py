import os
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from lilith_agent.runtime_env import apply_safe_thread_env

apply_safe_thread_env()

import gradio as gr
import pandas as pd
import requests

from lilith_agent.app import build_react_agent
from lilith_agent.config import Config
from lilith_agent.runner import run_agent_on_questions
from lilith_agent.scoring_client import DEFAULT_API_URL, ScoringApiClient


class LilithAgent:
    """ReAct agent from lilith_agent.app, wired for the scoring-API flow."""

    def __init__(self, cfg: Config | None = None, client: ScoringApiClient | None = None) -> None:
        self.cfg = cfg or Config.from_env()
        self.client = client or ScoringApiClient()
        self.graph = build_react_agent(self.cfg)
        print(f"LilithAgent initialized (caveman={self.cfg.caveman}/{self.cfg.caveman_mode}).", flush=True)
        print(
            "[config] "
            f"cheap={self.cfg.cheap_provider}/{self.cfg.cheap_model} "
            f"strong={self.cfg.strong_provider}/{self.cfg.strong_model} "
            f"extra={self.cfg.extra_strong_provider}/{self.cfg.extra_strong_model} "
            f"vision={self.cfg.vision_provider}/{self.cfg.vision_model} "
            f"recursion_limit={self.cfg.recursion_limit} "
            f"budget_warn_at={self.cfg.budget_warn_at} "
            f"budget_hard_cap={self.cfg.budget_hard_cap} "
            f"checkpoint_dir={self.cfg.checkpoint_dir}",
            flush=True,
        )


def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}", flush=True)
    else:
        print("User not logged in.", flush=True)
        return "Please Login to Hugging Face with the button.", None

    submit_url = f"{DEFAULT_API_URL}/submit"

    try:
        agent = LilithAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}", flush=True)
        traceback.print_exc()
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code, flush=True)

    print("Fetching questions from scoring API...", flush=True)
    try:
        questions_data = agent.client.get_questions()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}", flush=True)
        traceback.print_exc()
        return f"Error fetching questions: {e}", None
    if agent.client.last_warning:
        print(agent.client.last_warning, flush=True)

    if not questions_data:
        return "Fetched questions list is empty or invalid format.", None
    print(f"Fetched {len(questions_data)} questions.", flush=True)

    print(f"Running agent on {len(questions_data)} questions...", flush=True)
    try:
        answers_payload = run_agent_on_questions(
            agent.graph,
            questions_data,
            agent.cfg.checkpoint_dir,
            client=agent.client,
        )
    except Exception as e:
        print(f"[app] runner failed type={type(e).__name__} error={e}", flush=True)
        traceback.print_exc()
        return f"Agent runner failed: {type(e).__name__}: {e}", None
    print(f"[app] runner produced {len(answers_payload)} answers", flush=True)
    answers_by_id = {a["task_id"]: a["submitted_answer"] for a in answers_payload}
    results_log = [
        {
            "Task ID": item.get("task_id"),
            "Question": item.get("question"),
            "Submitted Answer": answers_by_id.get(item.get("task_id"), ""),
        }
        for item in questions_data
        if item.get("task_id") and item.get("question") is not None
    ]

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}", flush=True)

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        print(f"[submit] status_code={response.status_code}", flush=True)
        response.raise_for_status()
        result_data = response.json()
        print(
            "[submit] success "
            f"score={result_data.get('score', 'N/A')} "
            f"correct={result_data.get('correct_count', '?')} "
            f"attempted={result_data.get('total_attempted', '?')}",
            flush=True,
        )
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        print(f"[submit] http_error {error_detail}", flush=True)
        return f"Submission Failed: {error_detail}", pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        print("[submit] timeout", flush=True)
        return "Submission Failed: The request timed out.", pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        print(f"[submit] network_error {e}", flush=True)
        traceback.print_exc()
        return f"Submission Failed: Network error - {e}", pd.DataFrame(results_log)
    except Exception as e:
        print(f"[submit] unexpected_error type={type(e).__name__} error={e}", flush=True)
        traceback.print_exc()
        return f"An unexpected error occurred during submission: {e}", pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# 🦋 Lilith Agent — GAIA Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. Log in to your Hugging Face account using the button below. Your HF username is used for submission.
        2. Click **Run Evaluation & Submit All Answers** to fetch questions, run Lilith, submit answers, and see the score.

        ---
        Running the full GAIA set takes a while — Lilith plans, calls tools, and verifies each answer.
        Per-question checkpoints are cached so reruns skip already-answered questions.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30, flush=True)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}", flush=True)
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space", flush=True)
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).", flush=True)

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}", flush=True)
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}", flush=True)
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main", flush=True)
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.", flush=True)

    print("-" * (60 + len(" App Starting ")) + "\n", flush=True)

    print("Launching Gradio Interface for Lilith Agent Evaluation...", flush=True)
    demo.launch(debug=True, share=False)
