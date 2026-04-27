import os
import pytest
from dotenv import load_dotenv

# Load .env before evaluating anything
load_dotenv(override=True)

from lilith_agent.config import Config
from lilith_agent.gaia_dataset import GaiaDatasetClient
from lilith_agent.tools.vision import inspect_visual_content

def test_fal_vision_integration(tmp_path):
    """
    Test the FAL vision integration using GAIA task 8f80e01c-1296-4371-9486-bb3d68651a60.
    This test runs an actual API call to ensure payload and authentication are correct.
    """
    # Force use of FAL provider and Llava model
    os.environ["GAIA_VISION_PROVIDER"] = "fal"
    os.environ["GAIA_VISION_MODEL"] = "llava"  # which remaps to fal-ai/moondream-next internally
    
    cfg = Config.from_env()

    # Use a reliable random image from Picsum to satisfy Gemini's minimum dimension constraint
    # while completely bypassing Wikimedia's aggressive IP blocks.
    file_path_or_url = "https://picsum.photos/200"
    prompt = "Is there an image here? say YES."
    
    print(f"Invoking inspect_visual_content with FAL on URL: {file_path_or_url}")
    result = inspect_visual_content(file_path_or_url, prompt, cfg)
    
    assert "ERROR:" not in result, f"Vision API returned an error: {result}"
    assert "File Preparation Failed" not in result, f"Failed to download image: {result}"
    assert "All Vision Attempts Failed" not in result, "Vision multi-tier fallback failed completely"
    assert len(result.strip()) >= 3, "Vision result is suspiciously empty"
    print(f"Vision API responded successfully:\n{result}")
