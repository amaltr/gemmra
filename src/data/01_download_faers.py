"""
Step 1: Download FAERS data from FDA.
Run this on CPU (no GPU needed).

Downloads FAERS quarterly data extracts from 2019 Q1 – 2026 Q1.
Total: ~29 quarters × ~60 MB = ~1.7 GB compressed, ~9 GB uncompressed.

Why 2019Q1 start (not 2014Q1):
  - While AllDeletedCases.txt (2019Q1) + incremental delete files DO cover
    pre-2019 deletions, adding 2014-2018 data (20 extra quarters) provides
    no practical benefit: we sample ~21K pairs from ~7M reports (0.3% rate).
    Adding 5M more reports does not change the output distribution.
  - 2019Q1+ avoids schema transition issues (2014Q3 added PROD_AI, changed
    GNDR_COD→SEX) and folder structure inconsistencies in older quarters.
  - 29 quarters (~7M raw reports) is more than sufficient for ~21K training pairs.
  - Fewer quarters = faster download/processing (hackathon time constraint).

FDA ZIP folder structure (varies by quarter):
  <quarter>.zip/
  ├── ASCII/    (or Ascii/ or ascii/)  ← data files live HERE, not at root
  │   ├── DEMO19Q1.txt (or demo19q1.txt)
  │   ├── DRUG19Q1.txt
  │   └── ...
  ├── Deleted/  (or deleted/ or DELETED/) ← only present from 2019Q1+
  │   ├── ADR19Q1DeletedCases.txt (naming varies: ADR*, DELETE*, AllDeleted*)
  │   └── AllDeletedCases.txt (only in 2019Q1 — cumulative historical deletes)
  ├── FAQs.pdf
  └── Readme.pdf

Storage strategy:
  - Download to /workspace/ (3.1 TB ephemeral) — NOT /workspace/shared/ (28 GB persistent)
  - After preprocessing, only the final parquet (~200-500 MB) goes to /workspace/shared/
  - Raw files are re-downloadable and don't need persistence
"""

import os
import zipfile
import urllib.request
import sys
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# Generate all FAERS URLs dynamically
# ============================================================
# FDA URL pattern: https://fis.fda.gov/content/Exports/faers_ascii_{year}{Qq}{quarter}.zip
# Case of Q varies (Q3 vs q3) — we try lowercase first, then uppercase

BASE_URL = "https://fis.fda.gov/content/Exports"

def generate_faers_urls(start_year: int = 2019, end_year: int = 2026, end_quarter: int = 1) -> dict:
    """Generate download URLs for all FAERS quarters in the given range.
    
    Default range: 2019Q1–2026Q1 (29 quarters).
    2019Q1 start is pragmatic: 29 quarters (~7M reports) is more than sufficient
    for ~21K training pairs. Older quarters add schema complexity (2014Q3 changes),
    folder structure inconsistencies, and processing time without practical benefit.
    Note: AllDeletedCases.txt from 2019Q1 + incremental files DO cover pre-2019
    deletions, so data quality is not the blocker — sufficiency is.
    """
    urls = {}
    for year in range(start_year, end_year + 1):
        max_q = 4 if year < end_year else end_quarter
        for q in range(1, max_q + 1):
            quarter_key = f"{year}Q{q}"
            # FDA uses inconsistent casing — try lowercase q first (more common in recent years)
            urls[quarter_key] = f"{BASE_URL}/faers_ascii_{year}q{q}.zip"
    return urls

# Generate all URLs from 2019 Q1 to 2026 Q1
# See docstring above for rationale on why we start at 2019, not 2014.
FAERS_URLS = generate_faers_urls(2019, 2026, 1)

RAW_DIR = Path("data/raw")
EXPECTED_TABLES = ["DEMO", "DRUG", "REAC", "OUTC", "THER", "INDI", "RPSR"]


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress indication.
    
    Validates existing files using ZIP integrity check to detect
    partially downloaded or corrupted files.
    Uses chunked download with browser User-Agent for maximum speed.
    """
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        if size_mb > 1:  # Skip only if file is non-trivially sized
            # Verify the existing file is a valid ZIP archive.
            # Partially downloaded files pass the size check but fail extraction.
            if zipfile.is_zipfile(str(dest)):
                print(f"  ⏭️  Already downloaded: {dest.name} ({size_mb:.0f} MB)")
                return True
            else:
                print(f"  ⚠️  Corrupted/partial file detected: {dest.name} ({size_mb:.0f} MB) — re-downloading")
                dest.unlink()  # Delete corrupt file
    
    print(f"  ⬇️  Downloading: {url}")
    
    def _fast_download(download_url: str, target: Path) -> bool:
        """Chunked download with browser User-Agent for FDA speed."""
        import time
        req = urllib.request.Request(download_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'identity',  # Don't request compression — FDA ZIPs are already compressed
        })
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get('Content-Length', 0))
            total_mb = total / 1e6 if total else 0
            
            downloaded = 0
            start_time = time.time()
            chunk_size = 1024 * 1024  # 1 MB chunks (vs urlretrieve's ~8KB default)
            
            with open(str(target), 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress display
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
                    dl_mb = downloaded / 1e6
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {dl_mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%) — {speed:.1f} MB/s", end='', flush=True)
                    else:
                        print(f"\r    {dl_mb:.0f} MB — {speed:.1f} MB/s", end='', flush=True)
            print()  # Newline after progress
        return True
    
    try:
        _fast_download(url, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"  ✅ Downloaded: {size_mb:.0f} MB")
        return True
    except Exception as e:
        # Try uppercase Q variant
        alt_url = url.replace(f"q{url[-5]}", f"Q{url[-5]}")
        if alt_url != url:
            try:
                _fast_download(alt_url, dest)
                size_mb = dest.stat().st_size / 1e6
                print(f"  ✅ Downloaded: {size_mb:.0f} MB (uppercase Q URL)")
                return True
            except:
                pass
        print(f"  ❌ Download failed: {e}")
        print(f"     → Download manually and place at: {dest}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a ZIP file."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  ⏭️  Already extracted: {extract_dir.name}")
        return True
    
    print(f"  📦 Extracting: {zip_path.name}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            z.extractall(str(extract_dir))
        print(f"  ✅ Extracted to: {extract_dir}")
        return True
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def verify_files(extract_dir: Path, quarter: str) -> dict:
    """Verify expected FAERS files exist.
    
    Uses truly case-insensitive matching via f.name.lower() comparisons.
    This handles all real-world FDA variations:
    - Data files: ASCII/DEMO19Q1.txt, Ascii/demo19q1.txt, ascii/DEMO19Q1.txt
    - Deleted files: Deleted/ADR19Q1DeletedCases.txt, DELETED/DELETE21Q4.txt,
      deleted/AllDeletedCases.txt, etc.
    
    Critical on Linux (AMD cloud) where the filesystem is case-sensitive.
    """
    found = {}
    quarter_short = quarter[2:]  # 2024Q3 → 24Q3, 2020Q1 → 20Q1
    
    # Collect all files once (rglob is expensive for repeated calls)
    all_files = [f for f in extract_dir.rglob("*") if f.is_file()]
    
    for table in EXPECTED_TABLES:
        pattern_lower = f"{table}{quarter_short}".lower()
        # Case-insensitive search: compare lowercase pattern against lowercase filename
        txt_matches = [
            f for f in all_files
            if pattern_lower in f.name.lower() and f.suffix.lower() == '.txt'
        ]
        if txt_matches:
            found[table] = txt_matches[0]
    
    # Check for DELETED files — truly case-insensitive.
    # Real-world naming varies wildly across quarters:
    #   2019Q1: ADR19Q1DeletedCases.txt, AllDeletedCases.txt  (folder: Deleted/)
    #   2019Q2: ADR19Q2DeletedCases.txt                       (folder: Deleted/)
    #   2020Q4: 20Q4DeletedCases.txt                          (folder: deleted/)
    #   2021Q4+: DELETE21Q4.txt                               (folder: DELETED/)
    # Common pattern: filename.lower() contains 'delet' (covers Delete, DELETED, deleted)
    deleted_files = [
        f for f in all_files
        if 'delet' in f.name.lower() and f.suffix.lower() == '.txt'
    ]
    if deleted_files:
        found['DELETED'] = deleted_files  # Store ALL deleted files (may be multiple)
    
    return found


def main():
    print("=" * 60)
    print("  FAERS Data Download — 2019Q1 to 2026Q1")
    print("=" * 60)
    print(f"  Output directory: {RAW_DIR.absolute()}")
    print(f"  Quarters to download: {len(FAERS_URLS)} ({min(FAERS_URLS.keys())} – {max(FAERS_URLS.keys())})")
    print(f"  Expected total: ~{len(FAERS_URLS) * 60 / 1000:.1f} GB compressed")
    print(f"  Rationale: 29 quarters is sufficient; older quarters add complexity without benefit")
    print()
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_files = {}
    success_count = 0
    fail_count = 0
    
    # Phase 1: PARALLEL DOWNLOAD — FDA throttles per-connection to ~0.3 MB/s,
    # but allows multiple simultaneous connections. 8 parallel = ~2.4 MB/s total.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL_WORKERS = 8  # 8 simultaneous connections
    
    download_tasks = {}
    for quarter, url in sorted(FAERS_URLS.items()):
        zip_path = RAW_DIR / f"faers_ascii_{quarter}.zip"
        download_tasks[quarter] = (url, zip_path)
    
    # Filter out already-downloaded files
    pending = {}
    for quarter, (url, zip_path) in download_tasks.items():
        if zip_path.exists() and zip_path.stat().st_size > 1e6 and zipfile.is_zipfile(str(zip_path)):
            print(f"  ⏭️  Already downloaded: {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB)")
        else:
            pending[quarter] = (url, zip_path)
    
    if pending:
        print(f"\n  📥 Downloading {len(pending)} quarters with {PARALLEL_WORKERS} parallel connections...")
        print(f"     (FDA limits ~0.3 MB/s per connection, {PARALLEL_WORKERS} parallel ≈ {PARALLEL_WORKERS * 0.3:.1f} MB/s total)\n")
        
        def _download_quarter(args):
            quarter, url, zip_path = args
            try:
                import time
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': '*/*',
                    'Accept-Encoding': 'identity',
                })
                try:
                    resp = urllib.request.urlopen(req, timeout=120)
                except:
                    # Try uppercase Q variant
                    alt_url = url.replace(f"q{url[-5]}", f"Q{url[-5]}")
                    req = urllib.request.Request(alt_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': '*/*',
                        'Accept-Encoding': 'identity',
                    })
                    resp = urllib.request.urlopen(req, timeout=120)
                
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                start = time.time()
                with open(str(zip_path), 'wb') as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                elapsed = time.time() - start
                speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
                return quarter, True, f"{downloaded / 1e6:.0f} MB @ {speed:.1f} MB/s"
            except Exception as e:
                return quarter, False, str(e)
        
        work_items = [(q, url, zp) for q, (url, zp) in pending.items()]
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_download_quarter, item): item[0] for item in work_items}
            for future in as_completed(futures):
                quarter, ok, msg = future.result()
                if ok:
                    print(f"  ✅ {quarter}: {msg}")
                else:
                    print(f"  ❌ {quarter}: {msg}")
    
    # Phase 2: SEQUENTIAL EXTRACT + VERIFY (disk-bound, no parallelism benefit)
    print(f"\n  📦 Extracting and verifying...")
    for quarter in sorted(FAERS_URLS.keys()):
        zip_path = RAW_DIR / f"faers_ascii_{quarter}.zip"
        extract_dir = RAW_DIR / quarter
        
        if not zip_path.exists():
            fail_count += 1
            continue
        
        # Extract
        if not extract_zip(zip_path, extract_dir):
            fail_count += 1
            continue
        
        # Verify
        files = verify_files(extract_dir, quarter)
        all_files[quarter] = files
        txt_count = len([t for t in EXPECTED_TABLES if t in files])
        deleted_count = len(files.get('DELETED', []))
        print(f"  📊 {quarter}: {txt_count}/{len(EXPECTED_TABLES)} data tables, {deleted_count} deleted file(s)")
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    
    print(f"  ✅ Successfully downloaded: {success_count}/{len(FAERS_URLS)} quarters")
    if fail_count:
        print(f"  ❌ Failed: {fail_count} quarters")
    
    total_tables = sum(len([t for t in EXPECTED_TABLES if t in files]) for files in all_files.values())
    print(f"  📊 Total data tables: {total_tables}")
    
    # Show which quarters succeeded
    for quarter in sorted(all_files.keys()):
        files = all_files[quarter]
        txt_count = len([t for t in EXPECTED_TABLES if t in files])
        has_deleted = '🗑️' if 'DELETED' in files else '  '
        status = "✅" if txt_count >= 4 else "⚠️"
        print(f"    {status} {quarter}: {txt_count}/7 tables {has_deleted}")
    
    if all_files:
        print(f"\n  ✅ Data ready! Next step:")
        print(f"     python src/data/02_preprocess.py")
    else:
        print(f"\n  ❌ No data downloaded. Download manually from:")
        print(f"     https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
