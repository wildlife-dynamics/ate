import os
import hashlib
import zipfile
from pathlib import Path
from pydantic import Field
from urllib.parse import urlparse
from ecoscope.io import download_file
from typing import Annotated, Optional

from ecoscope_workflows_core.decorators import task


def normalize_file_url(path: str) -> str:
    """Convert file:// URL to local path, handling malformed Windows URLs."""
    if not path.startswith("file://"):
        return path

    path = path[7:]
    
    if os.name == 'nt':
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith('/') and len(path) > 2 and path[2] in (':', '|'):
            path = path[1:]

        path = path.replace('/', '\\')
        path = path.replace('|', ':')
    else:
        if not path.startswith('/'):
            path = '/' + path
    
    return path

@task
def download_file_and_persist(
    url: Annotated[str, Field(description="URL to download the file from")],
    output_path: Annotated[Optional[str], Field(description="Path to save the downloaded file or directory. Defaults to current working directory")] = None,
    retries: Annotated[int, Field(description="Number of retries on failure", ge=0)] = 3,
    overwrite_existing: Annotated[bool, Field(description="Whether to overwrite existing files")] = False,
    unzip: Annotated[bool, Field(description="Whether to unzip the file if it's a zip archive")] = False,
) -> str:
    """
    Downloads a file from the provided URL and persists it locally.
    If output_path is not specified, saves to the current working directory.
    Returns the full path to the downloaded file, or if unzipped, the path to the extracted directory.
    """
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = normalize_file_url(output_path)
    looks_like_dir = (
        output_path.endswith(os.sep)
        or output_path.endswith("/")
        or output_path.endswith("\\")
        or os.path.isdir(output_path)
    )

    if looks_like_dir:
        # ensure directory exists
        os.makedirs(output_path, exist_ok=True)

        # determine filename from Content-Disposition or URL
        import requests, email
        try:
            s = requests.Session()
            r = s.head(url, allow_redirects=True, timeout=10)
            cd = r.headers.get("content-disposition", "")
            filename = None
            if cd:
                # parse content-disposition safely
                m = email.message.Message()
                m["content-disposition"] = cd
                filename = m.get_param("filename")
            if not filename:
                filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"
        except Exception:
            filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"

        target_path = os.path.join(output_path, filename)
    else:
        target_path = output_path

    if not target_path or str(target_path).strip() == "":
        raise ValueError("Computed download target path is empty. Check 'output_path' argument.")

    # Store the parent directory to check for extracted content
    parent_dir = os.path.dirname(target_path)
    before_extraction = set()
    if unzip:
        if os.path.exists(parent_dir):
            before_extraction = set(os.listdir(parent_dir))

    # Do the download and bubble up useful context on failure
    try:
        download_file(
            url=url,
            path=target_path,
            retries=retries,
            overwrite_existing=overwrite_existing,
            unzip=unzip,
        )
    except Exception as e:
        # include debug info so callers can see what was attempted
        raise RuntimeError(
            f"download_file failed for url={url!r} path={target_path!r} retries={retries}. "
            f"Original error: {e}"
        ) from e

    # Determine the final persisted path
    if unzip and zipfile.is_zipfile(target_path):
        after_extraction = set(os.listdir(parent_dir))
        new_items = after_extraction - before_extraction
        zip_filename = os.path.basename(target_path)
        new_items.discard(zip_filename)
        
        if len(new_items) == 1:
            new_item = new_items.pop()
            new_item_path = os.path.join(parent_dir, new_item)
            if os.path.isdir(new_item_path):
                persisted_path = str(Path(new_item_path).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
        elif len(new_items) > 1:
            persisted_path = str(Path(parent_dir).resolve())
        else:
            extracted_dir = target_path.rsplit('.zip', 1)[0]
            if os.path.isdir(extracted_dir):
                persisted_path = str(Path(extracted_dir).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
    else:
        persisted_path = str(Path(target_path).resolve())

    if not os.path.exists(persisted_path):
        parent = os.path.dirname(persisted_path)
        if os.path.exists(parent):
            actual_files = os.listdir(parent)
            raise FileNotFoundError(
                f"Download failed — {persisted_path} not found after execution. "
                f"Files in {parent}: {actual_files}"
            )
        else:
            raise FileNotFoundError(
                f"Download failed — {persisted_path}. Parent dir missing: {parent}"
            )
    return persisted_path