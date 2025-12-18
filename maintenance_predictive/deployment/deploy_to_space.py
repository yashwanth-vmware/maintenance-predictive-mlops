"""
Deploy Hugging Face Space files (Dockerfile, requirements.txt, app.py) from this repo.

Usage (Colab):
    !python maintenance_predictive/deployment/deploy_to_space.py

Prereqs:
- Add HF_TOKEN in Colab Secrets (key: HF_TOKEN)
- Optional: set HF_SPACE_ID env var (default provided below)
"""

import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError


def get_hf_token() -> str:
    """
    Get HF_TOKEN from environment, or from Colab Secrets if running in Colab.
    """
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    # Colab-only fallback
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN") or ""
        if token:
            os.environ["HF_TOKEN"] = token  # export so other code can reuse it
            return token
    except Exception:
        pass

    raise EnvironmentError(
        "❌ HF_TOKEN not found. Set os.environ['HF_TOKEN'] or add 'HF_TOKEN' to Colab Secrets and restart runtime."
    )


def ensure_space_exists(api: HfApi, space_id: str, token: str, private: bool = False) -> None:
    """
    Ensure the Hugging Face Space repo exists; create it if missing.
    """
    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        print(f"✅ Space exists: {space_id}")
        return
    except Exception:
        print(f"ℹ️ Space not found. Creating: {space_id}")

    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            private=private,
            token=token,
            exist_ok=True,  # safe even if it appears during race conditions
        )
        print(f"✅ Created Space: {space_id}")
    except HfHubHTTPError as e:
        raise RuntimeError(f"❌ Unable to create/access Space '{space_id}': {e}") from e


def upload_files_to_space(
    api: HfApi,
    space_id: str,
    deploy_dir: Path,
    files: List[Tuple[str, str]],
) -> None:
    """
    Upload local files to Hugging Face Space repo root.
    files: [(local_filename, path_in_repo), ...]
    """
    for local_name, path_in_repo in files:
        local_fp = deploy_dir / local_name
        if not local_fp.exists():
            raise FileNotFoundError(f"❌ Local file missing: {local_fp}")

        print(f"📤 Uploading {local_fp} -> {space_id}/{path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local_fp),
            path_in_repo=path_in_repo,
            repo_id=space_id,
            repo_type="space",
        )


def main() -> None:
    # ---- Config ----
    HF_SPACE_ID = os.getenv("HF_SPACE_ID", "Yashwanthsairam/engine-predictive-maintenance")
    DEPLOY_DIR = Path(os.getenv("DEPLOY_DIR", "maintenance_predictive/deployment"))

    # Docker Spaces expect Dockerfile at repo root
    files_to_upload = [
        ("Dockerfile", "Dockerfile"),
        ("requirements.txt", "requirements.txt"),
        ("app.py", "app.py"),
    ]

    # ---- Auth + Client ----
    token = get_hf_token()
    api = HfApi(token=token)

    # ---- Ensure Space exists ----
    ensure_space_exists(api, HF_SPACE_ID, token, private=False)

    # ---- Upload ----
    upload_files_to_space(api, HF_SPACE_ID, DEPLOY_DIR, files_to_upload)

    print("✅ Deployment files uploaded. Open your Space and wait for the build to complete.")


if __name__ == "__main__":
    main()

