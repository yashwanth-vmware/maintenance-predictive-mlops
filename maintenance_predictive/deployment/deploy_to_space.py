"""
Deploy Hugging Face Space files (Dockerfile, requirements.txt, app.py) to a Hugging Face Space.

✅ Fixes included:
- Handles empty HF_SPACE_ID from GitHub Actions (vars.HF_SPACE_ID may be empty)
- Works in both Colab and GitHub Actions
- Clear validation + logs
- Uploads Docker deployment files to Space repo root (required by Docker Spaces)

Usage:
  # Local / Colab:
  python maintenance_predictive/deployment/deploy_to_space.py

  # GitHub Actions:
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
    HF_SPACE_ID: ${{ vars.HF_SPACE_ID }}   # optional (can be empty); default used if empty
"""

import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, create_repo

# Some hub versions raise HfHubHTTPError, others raise generic Exception subclasses
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:  # pragma: no cover
    class HfHubHTTPError(Exception):
        pass


DEFAULT_SPACE_ID = "Yashwanthsairam/maintenance-predictive-mlops"
DEPLOY_DIR_DEFAULT = "maintenance_predictive/deployment"


def get_hf_token() -> str:
    """
    Get HF_TOKEN from environment, or from Colab Secrets if running in Colab.
    """
    token = os.getenv("HF_TOKEN")
    if token and token.strip():
        return token.strip()

    # Colab fallback (optional)
    try:
        from google.colab import userdata  # type: ignore
        token = (userdata.get("HF_TOKEN") or "").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception:
        pass

    raise EnvironmentError(
        "❌ HF_TOKEN not found. Add it to GitHub Secrets (Actions) as HF_TOKEN "
        "or set os.environ['HF_TOKEN'] locally/Colab and rerun."
    )


def get_space_id() -> str:
    """
    Get HF_SPACE_ID from env; if missing or empty, fall back to DEFAULT_SPACE_ID.
    IMPORTANT: GitHub Actions vars.HF_SPACE_ID can be empty -> must use `or`.
    """
    space_id = (os.getenv("HF_SPACE_ID") or "").strip()
    return space_id if space_id else DEFAULT_SPACE_ID


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
            exist_ok=True,  # safe if created concurrently
        )
        print(f"✅ Created Space: {space_id}")
    except HfHubHTTPError as e:
        raise RuntimeError(f"❌ Unable to create/access Space '{space_id}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"❌ Unable to create/access Space '{space_id}': {e}") from e


def upload_files(
    api: HfApi,
    space_id: str,
    deploy_dir: Path,
    files: List[Tuple[str, str]],
) -> None:
    """
    Upload local files to Space root.
    Docker Spaces expect Dockerfile at the repo root.
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
    token = get_hf_token()
    api = HfApi(token=token)

    space_id = get_space_id()
    if not space_id.strip():
        raise ValueError("❌ HF_SPACE_ID resolved to empty string. Set a valid Space id like 'user/space-name'.")

    deploy_dir = Path(os.getenv("DEPLOY_DIR", DEPLOY_DIR_DEFAULT))

    # Files expected for Docker Space deployment
    files_to_upload = [
        ("Dockerfile", "Dockerfile"),
        ("requirements.txt", "requirements.txt"),
        ("app.py", "app.py"),
    ]

    print("----------")
    print(f"HF_SPACE_ID: {space_id}")
    print(f"DEPLOY_DIR : {deploy_dir.resolve()}")
    print("----------")

    ensure_space_exists(api, space_id, token, private=False)
    upload_files(api, space_id, deploy_dir, files_to_upload)

    print("✅ Deployment files uploaded. Open your Space and wait for build to complete.")


if __name__ == "__main__":
    main()

