import os
from pathlib import Path

from utils import ensure_dir, log


def download_video(url, output_dir):
    """Download a video from *url* into *output_dir* using yt-dlp.

    Returns the local file path of the downloaded video.
    """
    ensure_dir(output_dir)

    try:
        import yt_dlp
    except ImportError:
        log.error(
            "yt-dlp is required for downloading videos. "
            "Install it with: pip install yt-dlp"
        )
        raise

    # yt-dlp will append its own extension; we use a template so the
    # output filename is predictable.
    out_template = os.path.join(output_dir, "%(title).100s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        log.info("Downloading: %s", url)
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # yt-dlp may change the extension (e.g. mkv → mp4 after merging);
        # use the actual file on disk.
        base = Path(filename).stem
        candidates = list(Path(output_dir).glob(f"{base}.*"))
        if not candidates:
            # Fall back to the prepared filename
            candidates = [Path(filename)]
        local_path = str(candidates[0])

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Download did not produce a file: {local_path}")

    log.info("Downloaded to: %s", local_path)
    return local_path
