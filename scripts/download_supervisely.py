import os
import supervisely as sly

SERVER = "https://app.supervisely.com"
TOKEN = "7R6RII8AfkIeSbpW2GFzGckuemDc4VOu0C85wxyaCrHvXqLeDsJBejOfLvXg6Yh7XIfaiuNeM8j0i0xmd7fQRM8kuKEQreEGhfHtKe7hDGJqqgDOdypTFXEUTnMFE6JO"

PROJECT_ID = 328010

OUT_DIR = r"E:\3d_visual\ml\exports\supervisely_sdk"

api = sly.Api(SERVER, TOKEN)

project_info = api.project.get_info_by_id(PROJECT_ID)

print(f"Downloading project: {project_info.name}")

sly.download_project(
    api=api,
    project_id=PROJECT_ID,
    dest_dir=OUT_DIR,
    dataset_ids=None,
    log_progress=True,
    save_images=True,
)

print("DONE")
