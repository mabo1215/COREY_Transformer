from __future__ import annotations

import argparse
import json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a Hugging Face snapshot into the local HF cache.")
    parser.add_argument("repo_id")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--token")
    parser.add_argument("--local-dir")
    parser.add_argument("--resume-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import snapshot_download

    args = _parse_args()
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        local_dir=args.local_dir,
        resume_download=args.resume_download,
        force_download=args.force_download,
    )
    print(json.dumps({"repo_id": args.repo_id, "snapshot_path": path}, indent=2))


if __name__ == "__main__":
    main()