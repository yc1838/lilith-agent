from lilith_agent.config import Config
from lilith_agent.gaia_dataset import GaiaDatasetClient
from lilith_agent.tools.vision import inspect_visual_content
import os
import json

def test_fal_vision():
    os.environ["GAIA_VISION_PROVIDER"] = "fal"
    os.environ["GAIA_VISION_MODEL"] = "llava"
    cfg = Config.from_env()

    client = GaiaDatasetClient(config="2023_all", split="validation")
    task_id = "8f80e01c-1296-4371-9486-bb3d68651a60"
    
    # Let's find the question
    q = next((q for q in client.get_questions() if q["task_id"] == task_id), None)
    if not q:
        print(f"Task {task_id} not found in validation split!")
        return

    print("Task:", json.dumps(q, indent=2))
    
    if q.get("file_name"):
        file_path = client.download_file(task_id, dest_dir=".checkpoints/files")
        if file_path:
            print(f"Downloaded file to: {file_path}")
            print("Invoking inspect_visual_content with FAL...")
            result = inspect_visual_content(str(file_path), q["question"], cfg)
            print("-" * 50)
            print("VISION RESULT:")
            print(result)
        else:
            print("Failed to download file.")
    else:
        print("This task does not have an attached file.")

if __name__ == "__main__":
    test_fal_vision()
