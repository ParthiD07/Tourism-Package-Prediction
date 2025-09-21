from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="Parthi07/Tourism-Package-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
)
