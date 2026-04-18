"""
Download ETHUSDT U-margin futures data from Binance Data Portal.

Data:
  - klines: 1m / 5m / 15m / 1h / 4h
  - fundingRate

Period: 2021-01 ~ present (auto-detects latest available month)
Storage: data/raw/klines/{interval}/ and data/raw/funding_rate/
"""

import os
import sys
import zipfile
import hashlib
import logging
import argparse
from io import BytesIO
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
DATA_ROOT = Path(__file__).parent.parent / "data" / "raw"

KLINE_INTERVALS = ["1m", "5m", "15m", "1h", "4h"]
START_YEAR, START_MONTH = 2021, 1

MAX_WORKERS = 5          # concurrent downloads
CHUNK_SIZE = 1024 * 256  # 256 KB per read chunk
REQUEST_TIMEOUT = 60     # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_months(start_year: int, start_month: int):
    """Yield (year, month) from start up to last completed month."""
    today = date.today()
    # last fully-completed month
    if today.month == 1:
        end_year, end_month = today.year - 1, 12
    else:
        end_year, end_month = today.year, today.month - 1

    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def kline_url(interval: str, year: int, month: int) -> str:
    return f"{BASE_URL}/klines/ETHUSDT/{interval}/ETHUSDT-{interval}-{year}-{month:02d}.zip"


def funding_url(year: int, month: int) -> str:
    return f"{BASE_URL}/fundingRate/ETHUSDT/ETHUSDT-fundingRate-{year}-{month:02d}.zip"


def checksum_url(data_url: str) -> str:
    return data_url + ".CHECKSUM"


def verify_checksum(data: bytes, checksum_text: str) -> bool:
    """Verify SHA256 from Binance CHECKSUM file (format: '<sha256>  <filename>')."""
    try:
        expected_hash = checksum_text.strip().split()[0].lower()
        actual_hash = hashlib.sha256(data).hexdigest().lower()
        return expected_hash == actual_hash
    except Exception:
        return False


def download_and_extract(url: str, dest_dir: Path, verify: bool = True) -> str:
    """
    Download a zip from `url`, extract the CSV into `dest_dir`.
    Returns the path of the extracted CSV, or raises on error.
    Skips download if CSV already exists.
    """
    zip_name = url.split("/")[-1]           # e.g. ETHUSDT-1m-2021-01.zip
    csv_name = zip_name.replace(".zip", ".csv")
    csv_path = dest_dir / csv_name

    if csv_path.exists():
        return f"skip:{csv_path}"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download zip
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    if resp.status_code == 404:
        return f"404:{url}"
    resp.raise_for_status()

    raw = BytesIO()
    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
        raw.write(chunk)
    zip_bytes = raw.getvalue()

    # Optional checksum verification
    if verify:
        try:
            ck_resp = requests.get(checksum_url(url), timeout=REQUEST_TIMEOUT)
            if ck_resp.status_code == 200:
                if not verify_checksum(zip_bytes, ck_resp.text):
                    raise ValueError(f"Checksum mismatch: {url}")
        except ValueError:
            raise
        except Exception:
            pass  # checksum file unavailable — skip silently

    # Extract CSV
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        csv_members = [n for n in names if n.endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV in zip: {url}")
        zf.extract(csv_members[0], dest_dir)
        # Rename if extracted name differs from expected
        extracted = dest_dir / csv_members[0]
        if extracted != csv_path:
            extracted.rename(csv_path)

    return f"ok:{csv_path}"


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def build_tasks(intervals=None, start_year=START_YEAR, start_month=START_MONTH):
    """Return list of (url, dest_dir) tuples."""
    if intervals is None:
        intervals = KLINE_INTERVALS

    tasks = []

    # klines
    for interval in intervals:
        dest = DATA_ROOT / "klines" / interval
        for y, m in iter_months(start_year, start_month):
            tasks.append((kline_url(interval, y, m), dest))

    # fundingRate
    dest = DATA_ROOT / "funding_rate"
    for y, m in iter_months(start_year, start_month):
        tasks.append((funding_url(y, m), dest))

    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(tasks: list, verify: bool = True, workers: int = MAX_WORKERS):
    skipped = ok = failed = 0
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_and_extract, url, dest, verify): url
            for url, dest in tasks
        }
        with tqdm(total=len(tasks), unit="file", ncols=80) as bar:
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    result = fut.result()
                    if result.startswith("skip:"):
                        skipped += 1
                        bar.set_postfix(ok=ok, skip=skipped, fail=failed)
                    elif result.startswith("404:"):
                        # Month not yet published — not an error
                        skipped += 1
                        bar.set_postfix(ok=ok, skip=skipped, fail=failed)
                    else:
                        ok += 1
                        bar.set_postfix(ok=ok, skip=skipped, fail=failed)
                except Exception as exc:
                    failed += 1
                    errors.append((url, str(exc)))
                    bar.set_postfix(ok=ok, skip=skipped, fail=failed)
                    log.warning("FAIL %s — %s", url.split("/")[-1], exc)
                bar.update(1)

    log.info("Done. downloaded=%d  skipped=%d  failed=%d", ok, skipped, failed)
    if errors:
        log.error("Failed files:")
        for url, msg in errors:
            log.error("  %s  →  %s", url.split("/")[-1], msg)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Binance ETHUSDT futures data")
    parser.add_argument(
        "--intervals", nargs="+", default=KLINE_INTERVALS,
        help=f"Kline intervals to download (default: {KLINE_INTERVALS})"
    )
    parser.add_argument(
        "--start", default=f"{START_YEAR}-{START_MONTH:02d}",
        metavar="YYYY-MM",
        help="Start month inclusive (default: 2021-01)"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip checksum verification"
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Concurrent downloads (default: {MAX_WORKERS})"
    )
    args = parser.parse_args()

    try:
        sy, sm = map(int, args.start.split("-"))
    except ValueError:
        parser.error("--start must be in YYYY-MM format")

    tasks = build_tasks(intervals=args.intervals, start_year=sy, start_month=sm)
    log.info(
        "Total tasks: %d  (intervals=%s, start=%s, workers=%d, verify=%s)",
        len(tasks), args.intervals, args.start, args.workers, not args.no_verify
    )
    run(tasks, verify=not args.no_verify, workers=args.workers)


if __name__ == "__main__":
    main()
