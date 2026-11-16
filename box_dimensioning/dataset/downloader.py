"""Dataset downloader and manager for Google Research Objectron dataset (F1, F2, F3, F4).
Handles downloading index files, geometry, annotations, and sample frames.
"""

from typing import List, Dict, Optional
import os
import urllib.request
import concurrent.futures

OBJECTRON_BASE_URL = "https://storage.googleapis.com/objectron"
GITHUB_INDEX_URL = "https://raw.githubusercontent.com/google-research-datasets/Objectron/master/index"

INDEX_NAMES = {
    "cereal_box_train": "cereal_box_annotations_train",
    "cereal_box_test": "cereal_box_annotations_test",
    "book_train": "book_annotations_train",
    "book_test": "book_annotations_test",
}


def download_index_file(index_key: str, cache_dir: str = "data/indices") -> List[str]:
    """Downloads official Objectron index file and returns list of video IDs."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = INDEX_NAMES.get(index_key, index_key)
    filepath = os.path.join(cache_dir, filename)

    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        url = f"{GITHUB_INDEX_URL}/{filename}"
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            # Fallback to storage url
            alt_url = f"{OBJECTRON_BASE_URL}/v1/index/{filename}"
            urllib.request.urlretrieve(alt_url, filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    return video_ids


def download_sequence_metadata(video_id: str, output_base_dir: str = "data/objectron") -> Dict[str, str]:
    """Downloads annotation.pbdata and geometry.pbdata for a specific Objectron video ID."""
    seq_dir = os.path.join(output_base_dir, video_id)
    os.makedirs(seq_dir, exist_ok=True)

    annot_path = os.path.join(seq_dir, "annotation.pbdata")
    geom_path = os.path.join(seq_dir, "geometry.pbdata")

    annot_url = f"{OBJECTRON_BASE_URL}/annotations/{video_id}.pbdata"
    geom_url = f"{OBJECTRON_BASE_URL}/videos/{video_id}/geometry.pbdata"

    if not os.path.isfile(annot_path) or os.path.getsize(annot_path) == 0:
        urllib.request.urlretrieve(annot_url, annot_path)

    if not os.path.isfile(geom_path) or os.path.getsize(geom_path) == 0:
        urllib.request.urlretrieve(geom_url, geom_path)

    return {
        "video_id": video_id,
        "sequence_dir": seq_dir,
        "annotation_path": annot_path,
        "geometry_path": geom_path,
    }


def download_dataset_batch(
    video_ids: List[str],
    output_base_dir: str = "data/objectron",
    max_workers: int = 4
) -> List[Dict[str, str]]:
    """Downloads a batch of sequences concurrently."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(download_sequence_metadata, vid, output_base_dir): vid
            for vid in video_ids
        }
        for future in concurrent.futures.as_completed(future_to_id):
            vid = future_to_id[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Failed to download {vid}: {e}")
    return results
