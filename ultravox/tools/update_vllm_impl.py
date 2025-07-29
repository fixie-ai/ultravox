import dataclasses
import json
import os
import sys
from typing import List, Optional

import requests
import simple_parsing


@dataclasses.dataclass
class UpdateVLLMArgs:
    """Updates specified files in a local vLLM installation from a GitHub branch,
    either by specifying files explicitly or by comparing against a parent commit."""

    # GitHub branch to pull files from (e.g., 'main', 'farzad-llama4')
    branch: str = simple_parsing.field(alias="-b")
    # Optional: Parent commit SHA. If provided, script will fetch files changed between this commit and the branch HEAD.
    parent_commit: Optional[str] = simple_parsing.field(default=None, alias="-c")
    # Optional: List of relative file paths within the vLLM repo to update. Use this OR parent_commit.
    # These paths should be relative to the root of the vllm package directory.
    files: Optional[List[str]] = simple_parsing.field(default=None, alias="-f")
    # GitHub repository owner/name
    repo: str = simple_parsing.field(default="fixie-ai/vllm", alias="-r")


def get_vllm_base_path():
    """Finds the base path of the installed vllm package."""
    try:
        import vllm  # type: ignore

        # __file__ points to vllm/__init__.py, so we need the parent directory
        return os.path.dirname(vllm.__file__)
    except ImportError:
        print("Error: Could not import the 'vllm' package.", file=sys.stderr)
        print(
            "Please ensure vLLM is installed in your Python environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error determining vLLM path: {e}", file=sys.stderr)
        sys.exit(1)


def get_changed_files_from_github(repo: str, base: str, head: str) -> List[str]:
    """Gets list of changed files between two commits/branches using GitHub API."""
    compare_url = f"https://api.github.com/repos/{repo}/compare/{base}...{head}"
    print(f"  Fetching diff from GitHub API: {compare_url}")
    try:
        # Consider adding authentication (e.g., using a token) for private repos or to avoid rate limits
        # headers = {'Authorization': f'token YOUR_GITHUB_TOKEN'}
        # response = requests.get(compare_url, headers=headers)
        response = requests.get(compare_url)
        response.raise_for_status()
        data = response.json()

        if "files" not in data:
            # Handle cases like comparing identical commits or unexpected API response format
            print(
                f"  Warning: 'files' key not found or empty in GitHub API response for {base}...{head}.",
                file=sys.stderr,
            )
            return []

        # Filter out removed files as we can't download them
        changed_files = [
            f["filename"] for f in data.get("files", []) if f.get("status") != "removed"
        ]

        if not changed_files:
            print(
                f"  No changed (added/modified) files found between {base} and {head}."
            )
        else:
            print(f"  Found {len(changed_files)} changed files to potentially update.")
        return changed_files

    except requests.exceptions.HTTPError as e:
        print(
            f"  HTTP Error calling GitHub API ({compare_url}): {e.response.status_code} {e.response.reason}",
            file=sys.stderr,
        )
        # Attempt to parse error message from GitHub response
        try:
            error_data = response.json()
            if "message" in error_data:
                print(
                    f"  GitHub API Error Message: {error_data['message']}",
                    file=sys.stderr,
                )
        except (json.JSONDecodeError, AttributeError):
            pass  # Ignore if response is not JSON or doesn't have details
        print(
            "  Check if the repo, branch, or commit SHA are correct and you have access.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(
            f"  Network Error calling GitHub API ({compare_url}): {e}", file=sys.stderr
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"  Error parsing GitHub API response: {e}", file=sys.stderr)
        print(
            f"  Response text: {response.text[:500]}...", file=sys.stderr
        )  # Log beginning of response
        sys.exit(1)
    except Exception as e:
        print(
            f"  An unexpected error occurred while fetching diff: {e}", file=sys.stderr
        )
        sys.exit(1)


def main(args: UpdateVLLMArgs):
    # --- Argument validation ---
    if args.files and args.parent_commit:
        print(
            "Error: Provide either --files (-f) or --parent-commit (-c), not both.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.files and not args.parent_commit:
        print(
            "Error: Provide either --files (-f) or --parent-commit (-c).",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Determine files to update ---
    files_to_update: List[str] = []
    if args.parent_commit:
        print(
            f"Finding changed files between commit '{args.parent_commit}' and branch '{args.branch}' in repo '{args.repo}'..."
        )
        files_to_update = get_changed_files_from_github(
            repo=args.repo, base=args.parent_commit, head=args.branch
        )
        if not files_to_update:
            print("No files to update based on the diff. Exiting.")
            sys.exit(0)
    elif args.files:  # Use elif to be explicit, args.files must be non-None here
        files_to_update = args.files
        print(f"Using explicitly provided file list: {files_to_update}")
    # Else case should not happen due to validation above

    # --- Proceed with download/update ---
    vllm_base_path = get_vllm_base_path()
    print(f"Detected vLLM installation path: {vllm_base_path}")

    # Base URL for raw file content uses the target branch
    base_raw_url = f"https://raw.githubusercontent.com/{args.repo}/{args.branch}"

    print(f"Updating files from branch '{args.branch}' in repo '{args.repo}'...")
    updated_count = 0
    error_count = 0

    for relative_file_path in files_to_update:
        # Construct the download URL using the relative path as is
        updated_url = f"{base_raw_url}/{relative_file_path}"
        # Construct the target file path using the detected vllm base path and the relative path
        path_inside_package = relative_file_path
        if path_inside_package.startswith("vllm/"):
            path_inside_package = path_inside_package[len("vllm/") :]

        target_file = os.path.join(
            vllm_base_path, os.path.normpath(path_inside_package)
        )

        print(f"  Attempting to update {relative_file_path}")
        print(f"    Source URL: {updated_url}")
        print(f"    Target Path: {target_file}")

        try:
            response = requests.get(updated_url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        except requests.exceptions.HTTPError as e:
            # Specifically handle 404 for files that might be in the diff but not downloadable (e.g., submodules)
            if e.response.status_code == 404:
                print(
                    f"    Warning: Skipping file {relative_file_path} - Not found at {updated_url} (status 404). Might be a submodule or deleted after diff.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"    Error downloading {updated_url}: HTTP {e.response.status_code}",
                    file=sys.stderr,
                )
            error_count += 1
            continue  # Skip to the next file
        except requests.exceptions.RequestException as e:
            print(f"    Error downloading {updated_url}: {e}")
            error_count += 1
            continue  # Skip to the next file

        target_dir = os.path.dirname(target_file)
        if not os.path.exists(target_dir):
            print(f"    Creating directory {target_dir}...")
            try:
                os.makedirs(
                    target_dir, exist_ok=True
                )  # exist_ok=True handles race conditions
            except OSError as e:
                print(f"    Error creating directory {target_dir}: {e}")
                error_count += 1
                continue  # Skip to the next file

        print(f"    Saving to {target_file}...")
        try:
            with open(target_file, "w", encoding="utf-8") as f:  # Specify encoding
                f.write(response.text)
            print(f"    Successfully updated {target_file}")
            updated_count += 1
        except IOError as e:
            print(f"    Error writing file {target_file}: {e}")
            error_count += 1
        except Exception as e:  # Catch potential encoding errors, etc.
            print(f"    Unexpected error saving file {target_file}: {e}")
            error_count += 1

    print("\nUpdate process finished.")
    print(f"Summary: {updated_count} files updated, {error_count} errors encountered.")
    if error_count > 0:
        sys.exit(1)  # Exit with error code if any errors occurred


if __name__ == "__main__":
    # Add dash variants for convenience (e.g., --branch instead of -branch)
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(UpdateVLLMArgs, dest="update_args")
    args = parser.parse_args()
    main(args.update_args)
