"""
Step 2: Preprocess FAERS data using DuckDB.
Run this on CPU (no GPU needed).

Loads raw FAERS files → filters DELETED cases → deduplicates → quality filters
→ master join (with OUTC/concomitant pre-aggregation) → saves.

Data quality fixes applied (see data_quality_audit.md):
  - DELETED.txt filtering (Issue #1)
  - OUTC pre-aggregation to avoid row fan-out (Issue #2)
  - Year 2020 file discovery fix (Issue #3)
  - rept_cod included in SELECT (Issue #4)
  - n_concomitant pre-computed in SQL (Issue #5)
"""

import duckdb
import pandas as pd
from pathlib import Path
import sys
import json

# Fix Windows terminal encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EXPECTED_TABLES = ["DEMO", "DRUG", "REAC", "OUTC", "THER", "INDI", "RPSR"]

def discover_quarters(raw_dir: Path) -> list[str]:
    """Auto-discover all downloaded FAERS quarters from the raw directory."""
    quarters = []
    if raw_dir.exists():
        for d in sorted(raw_dir.iterdir()):
            if d.is_dir() and len(d.name) == 6 and d.name[:4].isdigit() and 'Q' in d.name.upper():
                quarters.append(d.name)
    return quarters


def find_faers_file(raw_dir: Path, quarter: str, table: str) -> str | None:
    """Find the FAERS file for a given quarter and table."""
    quarter_short = quarter[2:]  # 2024Q3 → 24Q3, 2020Q1 → 20Q1
    quarter_dir = raw_dir / quarter
    
    if not quarter_dir.exists():
        return None
    
    # Search patterns
    for f in quarter_dir.rglob("*"):
        if (table.lower() in f.name.lower() and 
            quarter_short.lower() in f.name.lower() and 
            f.suffix.lower() == '.txt'):
            return str(f)
    
    return None


def collect_deleted_ids(raw_dir: Path, quarters: list[str]) -> set[str]:
    """Collect all case IDs from DELETED files across all quarters.
    
    FAERS includes DELETED files in some quarterly ZIPs listing retracted
    cases (duplicates, manufacturer retractions, data errors). These must
    be excluded before any processing.
    
    IMPORTANT: DELETED files contain CASEID values (6-8 digit integers),
    NOT primaryid values. A caseid identifies a case across all versions,
    so deleting a caseid removes ALL versions of that case.
    
    Deleted file naming varies wildly across quarters:
      2019Q1: ADR19Q1DeletedCases.txt + AllDeletedCases.txt  (folder: Deleted/)
      2019Q2: ADR19Q2DeletedCases.txt                        (folder: Deleted/)
      2020Q4: 20Q4DeletedCases.txt                           (folder: deleted/)
      2021Q4+: DELETE21Q4.txt                                (folder: DELETED/)
    
    File format: Headerless text files with one caseid per line.
    Some files may have a 'caseid' header line or trailing '$' characters.
    
    AllDeletedCases.txt (only in 2019Q1) is a cumulative list of ALL
    historically nullified cases — it covers deletions from before 2019.
    
    See: PROBLEM_STATEMENT.md Section 4 "Data Quality Considerations" point 1.
    """
    deleted_ids = set()
    
    for quarter in quarters:
        quarter_dir = raw_dir / quarter
        if not quarter_dir.exists():
            continue
        
        # Truly case-insensitive search: match any .txt file containing 'delet'
        # in its name (covers Delete, DELETED, deleted, AllDeletedCases, etc.)
        deleted_files = [
            f for f in quarter_dir.rglob("*")
            if f.is_file() and 'delet' in f.name.lower() and f.suffix.lower() == '.txt'
        ]
        
        for dfile in deleted_files:
            try:
                ids = _parse_deleted_file(dfile)
                if ids:
                    deleted_ids.update(ids)
                    print(f"  🗑️  {dfile.name}: {len(ids):,} deleted IDs")
                else:
                    print(f"  ⚠️  {dfile.name}: no IDs found")
            except Exception as e:
                print(f"  ⚠️  Could not parse DELETED file {dfile.name}: {e}")
    
    return deleted_ids


def _parse_deleted_file(filepath: Path) -> set[str]:
    """Parse a FAERS DELETED file and extract case IDs.
    
    DELETED files contain caseid values (NOT primaryid). Handles:
    - One caseid per line, headerless (most common)
    - Delimited by '$' (with or without headers like 'caseid')
    - Lines with trailing '$'
    - Optional 'caseid' header line (skipped automatically since it's not all-digits)
    """
    text = filepath.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return set()
    
    lines = text.splitlines()
    ids = set()
    for line in lines:
        parts = line.strip().split('$')
        for part in parts:
            val = part.strip()
            # If the part is non-empty and consists only of digits, it's a valid ID
            if val and val.isdigit():
                ids.add(val)
    return ids


def load_and_process(conn: duckdb.DuckDBPyConnection, quarters: list[str]):
    """Load all FAERS files into DuckDB views."""
    
    print("\n📂 Step 1: Loading FAERS files...")
    loaded = {}
    
    for quarter in quarters:
        for table in EXPECTED_TABLES:
            filepath = find_faers_file(RAW_DIR, quarter, table)
            if filepath:
                view_name = f"{table.lower()}_{quarter[2:]}"
                try:
                    conn.execute(f"""
                        CREATE OR REPLACE VIEW {view_name} AS
                        SELECT * FROM read_csv_auto(
                            '{filepath}', 
                            delim='$', 
                            header=True, 
                            ignore_errors=True,
                            all_varchar=True
                        )
                    """)
                    count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
                    print(f"  ✅ {view_name}: {count:,} rows")
                    loaded[f"{table}_{quarter}"] = count
                except Exception as e:
                    print(f"  ❌ {view_name}: {e}")
            else:
                print(f"  ⚠️  Not found: {table} for {quarter}")
    
    # Combine quarters
    print("\n🔗 Step 2: Combining quarters...")
    for table in EXPECTED_TABLES:
        table_lower = table.lower()
        views = []
        for quarter in quarters:
            view_name = f"{table_lower}_{quarter[2:]}"
            try:
                conn.execute(f"SELECT 1 FROM {view_name} LIMIT 1")
                views.append(view_name)
            except:
                pass
        
        if views:
            # Use UNION ALL BY NAME to handle schema evolution across quarters.
            # FDA added columns over time (e.g., prod_ai added to DRUG in 2014Q3).
            # Positional UNION ALL would crash or misalign columns; BY NAME fills
            # missing columns with NULL.
            union_sql = " UNION ALL BY NAME ".join(f"SELECT * FROM {v}" for v in views)
            conn.execute(f"CREATE OR REPLACE VIEW {table_lower} AS {union_sql}")
            count = conn.execute(f"SELECT COUNT(*) FROM {table_lower}").fetchone()[0]
            print(f"  ✅ {table_lower}: {count:,} total rows")
    
    return loaded


def deduplicate(conn: duckdb.DuckDBPyConnection):
    """Deduplicate by keeping latest caseversion per caseid.
    
    Falls back to fda_dt if caseversion is not available.
    """
    
    print("\n🔄 Step 3: Deduplicating...")
    
    before = conn.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    
    # Check which dedup column is available
    demo_cols = [c.lower() for c in conn.execute("SELECT * FROM demo LIMIT 0").df().columns]
    
    if 'caseversion' in demo_cols:
        order_expr = "CAST(caseversion AS INTEGER) DESC"
        print("  Using caseversion for dedup ordering")
    elif 'fda_dt' in demo_cols:
        order_expr = "fda_dt DESC"
        print("  ⚠️  caseversion missing — using fda_dt for dedup ordering")
    else:
        # No versioning column available — keep all, just in case
        order_expr = "primaryid DESC"
        print("  ⚠️  No caseversion or fda_dt — dedup by primaryid (may keep duplicates)")
    
    if 'caseid' in demo_cols:
        partition_col = "caseid"
    else:
        partition_col = "primaryid"
        print("  ⚠️  caseid missing — partitioning by primaryid")
    
    conn.execute(f"""
        CREATE OR REPLACE VIEW demo_dedup AS
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY {partition_col} 
                ORDER BY {order_expr}
            ) as rn
            FROM demo
        ) WHERE rn = 1
    """)
    
    after = conn.execute("SELECT COUNT(*) FROM demo_dedup").fetchone()[0]
    print(f"  Before dedup: {before:,}")
    print(f"  After dedup:  {after:,} (removed {before - after:,} duplicates)")


def filter_deleted(conn: duckdb.DuckDBPyConnection, deleted_ids: set[str]):
    """Filter out DELETED (retracted) cases from FAERS.
    
    FDA publishes DELETED files listing caseids that have been retracted
    (duplicates, manufacturer retractions, data errors). The deleted files
    contain CASEID values (6-8 digits), not primaryid values.
    
    We match against both caseid AND primaryid columns in demo_dedup to
    handle edge cases, since some older files may list primaryid-format IDs.
    """
    
    print("\n🗑️  Step 3b: Filtering DELETED (retracted) cases...")
    
    if not deleted_ids:
        print("  ℹ️  No DELETED files found — skipping")
        # Create pass-through view
        conn.execute("""
            CREATE OR REPLACE VIEW demo_clean AS
            SELECT * FROM demo_dedup
        """)
        return
    
    # Register deleted IDs as a DuckDB table for efficient filtering.
    # Using a TABLE (not view) + index allows DuckDB to use hash joins.
    deleted_df = pd.DataFrame({'deleted_id': list(deleted_ids)})
    conn.register('deleted_ids_df', deleted_df)
    
    conn.execute("""
        CREATE OR REPLACE TABLE deleted_ids AS 
        SELECT CAST(deleted_id AS VARCHAR) as deleted_id FROM deleted_ids_df
    """)
    
    before = conn.execute("SELECT COUNT(*) FROM demo_dedup").fetchone()[0]
    
    # Discover actual columns in demo_dedup
    demo_cols = [c.lower() for c in conn.execute("SELECT * FROM demo_dedup LIMIT 0").df().columns]
    
    # Show sample IDs for debugging
    sample_deleted = list(deleted_ids)[:5]
    print(f"  Sample deleted IDs: {sample_deleted}")
    
    if 'caseid' in demo_cols:
        # Show sample caseid/primaryid from demo for comparison
        sample = conn.execute("""
            SELECT CAST(primaryid AS VARCHAR) as pid, CAST(caseid AS VARCHAR) as cid 
            FROM demo_dedup LIMIT 3
        """).df()
        print(f"  Sample demo primaryid/caseid: {list(zip(sample['pid'], sample['cid']))}")
        
        # Pre-check: how many caseids overlap with deleted IDs?
        overlap = conn.execute("""
            SELECT COUNT(DISTINCT d.caseid) 
            FROM demo_dedup d 
            WHERE CAST(d.caseid AS VARCHAR) IN (SELECT deleted_id FROM deleted_ids)
        """).fetchone()[0]
        print(f"  Caseid matches found: {overlap:,}")
        
        overlap_pid = conn.execute("""
            SELECT COUNT(DISTINCT d.primaryid) 
            FROM demo_dedup d 
            WHERE CAST(d.primaryid AS VARCHAR) IN (SELECT deleted_id FROM deleted_ids)
        """).fetchone()[0]
        print(f"  Primaryid matches found: {overlap_pid:,}")
        
        # Use ANTI JOIN pattern: remove cases where caseid OR primaryid is in deleted set.
        # DELETED files contain caseid values, so caseid match is the primary path.
        conn.execute("""
            CREATE OR REPLACE VIEW demo_clean AS
            SELECT d.* FROM demo_dedup d
            WHERE CAST(d.caseid AS VARCHAR) NOT IN (SELECT deleted_id FROM deleted_ids)
              AND CAST(d.primaryid AS VARCHAR) NOT IN (SELECT deleted_id FROM deleted_ids)
        """)
        print("  ✅ Filtering on both caseid and primaryid")
    else:
        conn.execute("""
            CREATE OR REPLACE VIEW demo_clean AS
            SELECT d.* FROM demo_dedup d
            WHERE CAST(d.primaryid AS VARCHAR) NOT IN (SELECT deleted_id FROM deleted_ids)
        """)
        print("  ⚠️ caseid missing from demo_dedup — checking primaryid only")
        
    after = conn.execute("SELECT COUNT(*) FROM demo_clean").fetchone()[0]
    removed = before - after
    print(f"  DELETED IDs loaded: {len(deleted_ids):,}")
    print(f"  Cases removed: {removed:,}")
    print(f"  Cases remaining: {after:,}")


def quality_filter(conn: duckdb.DuckDBPyConnection):
    """Filter for high-quality reports.
    
    Keep reports from healthcare professionals (HP) and literature (LIT).
    We intentionally do NOT filter on rept_cod here — keeping both EXP
    (expedited/serious) and PER (periodic/non-serious) provides a natural
    class balance for T1 seriousness training. See data_quality_audit.md #4.
    """
    
    print("\n🔍 Step 4: Quality filtering...")
    
    # Keep reports from healthcare professionals and literature
    conn.execute("""
        CREATE OR REPLACE VIEW demo_filtered AS
        SELECT d.* FROM demo_clean d
        WHERE d.primaryid IN (
            SELECT DISTINCT primaryid FROM rpsr 
            WHERE rpsr_cod IN ('HP', 'LIT')
        )
    """)
    
    count = conn.execute("SELECT COUNT(*) FROM demo_filtered").fetchone()[0]
    print(f"  After HP/LIT filter: {count:,} cases")


def master_join(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join all tables into a single master dataframe.
    
    Key design decisions:
    - OUTC is pre-aggregated to a comma-separated string per case (avoids row fan-out)
    - rept_cod is included for downstream use
    - n_concomitant is pre-computed via subquery (avoids O(n²) runtime computation)
    - REAC is NOT aggregated (one row per reaction per case is correct for T2)
    - All column references are dynamically resolved to handle FAERS schema
      variations across quarters (e.g., gndr_cod → sex rename in 2014Q3,
      drug_seq presence varies in THER/INDI)
    """
    
    print("\n🔗 Step 5: Master join...")
    
    # Discover actual columns in each table.
    # FAERS schema varies across quarters — column names can differ.
    def _get_columns(table: str) -> list[str]:
        try:
            cols = conn.execute(f"SELECT * FROM {table} LIMIT 0").df().columns
            return [c.lower() for c in cols]
        except Exception:
            return []
    
    demo_cols = _get_columns('demo_filtered')
    drug_cols = _get_columns('drug')
    ther_cols = _get_columns('ther')
    indi_cols = _get_columns('indi')
    
    print(f"  📋 DEMO columns: {len(demo_cols)} found")
    print(f"  📋 DRUG columns: {len(drug_cols)} found")
    print(f"  📋 THER columns: {len(ther_cols)} found")
    print(f"  📋 INDI columns: {len(indi_cols)} found")
    
    # --- Resolve DEMO column variations ---
    # gndr_cod was renamed to 'sex' in 2014Q3. Since we use 2019+ data,
    # it should always be 'sex', but handle both for safety.
    if 'sex' in demo_cols:
        gender_select = "d.sex AS gndr_cod"
        print("  ✅ DEMO has 'sex' column (post-2014Q3 schema) — aliasing to gndr_cod")
    elif 'gndr_cod' in demo_cols:
        gender_select = "d.gndr_cod"
        print("  ✅ DEMO has 'gndr_cod' column (pre-2014Q3 schema)")
    else:
        gender_select = "NULL AS gndr_cod"
        print("  ⚠️  DEMO has neither 'sex' nor 'gndr_cod' — using NULL")
    
    # age_cod may also be named differently
    age_cod_select = "d.age_cod" if 'age_cod' in demo_cols else "NULL AS age_cod"
    
    # occp_cod (reporter occupation)
    occp_select = "d.occp_cod" if 'occp_cod' in demo_cols else "NULL AS occp_cod"
    
    # rept_cod (report type: EXP/PER)
    rept_select = "d.rept_cod" if 'rept_cod' in demo_cols else "NULL AS rept_cod"
    
    # event_dt (event date)
    event_dt_select = "d.event_dt" if 'event_dt' in demo_cols else "NULL AS event_dt"
    
    # --- Resolve DRUG column variations ---
    prod_ai_select = "dr.prod_ai" if 'prod_ai' in drug_cols else "NULL AS prod_ai"
    nda_num_select = "dr.nda_num" if 'nda_num' in drug_cols else "NULL AS nda_num"
    dechal_select = "dr.dechal" if 'dechal' in drug_cols else "NULL AS dechal"
    rechal_select = "dr.rechal" if 'rechal' in drug_cols else "NULL AS rechal"
    drug_seq_select = "dr.drug_seq" if 'drug_seq' in drug_cols else "NULL AS drug_seq"
    
    # --- Resolve THER/INDI join conditions ---
    drug_has_seq = 'drug_seq' in drug_cols
    ther_has_seq = 'drug_seq' in ther_cols
    indi_has_seq = 'drug_seq' in indi_cols
    
    if drug_has_seq and ther_has_seq:
        ther_join = "LEFT JOIN ther t ON d.primaryid = t.primaryid AND dr.drug_seq = t.drug_seq"
        print("  ✅ THER join: using drug_seq for precise drug-level matching")
    else:
        ther_join = "LEFT JOIN ther t ON d.primaryid = t.primaryid"
        print("  ⚠️  THER join: primaryid-only (drug_seq missing from THER or DRUG)")
    
    if drug_has_seq and indi_has_seq:
        indi_join = "LEFT JOIN indi i ON d.primaryid = i.primaryid AND dr.drug_seq = i.drug_seq"
        print("  ✅ INDI join: using drug_seq for precise drug-level matching")
    else:
        indi_join = "LEFT JOIN indi i ON d.primaryid = i.primaryid"
        print("  ⚠️  INDI join: primaryid-only (drug_seq missing from INDI or DRUG)")
    
    # --- Resolve THER column availability ---
    start_dt_select = "t.start_dt" if 'start_dt' in ther_cols else "NULL AS start_dt"
    end_dt_select = "t.end_dt" if 'end_dt' in ther_cols else "NULL AS end_dt"
    
    query = f"""
        SELECT 
            d.primaryid, 
            d.caseid, 
            d.age, 
            {age_cod_select}, 
            {gender_select}, 
            {event_dt_select},
            {occp_select},
            {rept_select},
            dr.drugname, 
            {prod_ai_select}, 
            dr.role_cod, 
            {nda_num_select}, 
            {dechal_select}, 
            {rechal_select}, 
            {drug_seq_select},
            r.pt AS meddra_pt,
            outc_agg.outc_codes AS outc_cod,
            {start_dt_select}, 
            {end_dt_select},
            i.indi_pt,
            COALESCE(con.n_concomitant, 0) AS n_concomitant
        FROM demo_filtered d
        LEFT JOIN drug dr ON d.primaryid = dr.primaryid AND dr.role_cod = 'PS'
        LEFT JOIN reac r ON d.primaryid = r.primaryid
        LEFT JOIN (
            -- Pre-aggregate OUTC: one row per case with comma-separated outcome codes
            SELECT primaryid, STRING_AGG(DISTINCT outc_cod, ',') AS outc_codes
            FROM outc
            WHERE outc_cod IS NOT NULL
            GROUP BY primaryid
        ) outc_agg ON d.primaryid = outc_agg.primaryid
        {ther_join}
        {indi_join}
        LEFT JOIN (
            -- Pre-compute concomitant drug count per case
            SELECT primaryid, COUNT(*) AS n_concomitant
            FROM drug
            WHERE role_cod = 'C'
            GROUP BY primaryid
        ) con ON d.primaryid = con.primaryid
        WHERE dr.drugname IS NOT NULL 
          AND r.pt IS NOT NULL
    """
    
    df = conn.execute(query).df()
    
    print(f"  ✅ Master dataset: {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Quick stats
    print(f"\n  📊 Quick stats:")
    print(f"     Unique cases: {df['caseid'].nunique():,}")
    print(f"     Unique drugs: {df['drugname'].nunique():,}")
    print(f"     Unique PTs: {df['meddra_pt'].nunique():,}")
    print(f"     Has outcome: {df['outc_cod'].notna().sum():,}")
    if 'dechal' in df.columns and df['dechal'].notna().any():
        print(f"     Has dechallenge: {(df['dechal'].str.upper() == 'Y').sum():,}")
    if 'rept_cod' in df.columns and df['rept_cod'].notna().any():
        print(f"     Has rept_cod: {df['rept_cod'].notna().sum():,}")
    print(f"     Avg concomitant drugs: {df['n_concomitant'].mean():.1f}")
    
    return df


def main():
    print("=" * 60)
    print("  FAERS Preprocessing — Step 2 of Data Pipeline")
    print("=" * 60)
    
    # Auto-discover all downloaded quarters
    quarters = discover_quarters(RAW_DIR)
    if not quarters:
        print(f"  ❌ No FAERS quarter directories found in {RAW_DIR}")
        print(f"     Run first: python src/data/01_download_faers.py")
        return 1
    
    print(f"  📅 Discovered {len(quarters)} quarters: {', '.join(quarters)}")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect DELETED case IDs before loading into DuckDB
    deleted_ids = collect_deleted_ids(RAW_DIR, quarters)
    if deleted_ids:
        print(f"  🗑️  Found {len(deleted_ids):,} DELETED (retracted) case IDs to filter")
    else:
        print(f"  ℹ️  No DELETED files found in any quarter")
    
    # Use in-memory DuckDB
    conn = duckdb.connect()
    
    # Load
    load_and_process(conn, quarters)
    
    # Dedup
    deduplicate(conn)
    
    # Filter DELETED cases
    filter_deleted(conn, deleted_ids)
    
    # Quality filter
    quality_filter(conn)
    
    # Join
    df = master_join(conn)
    
    # Save — parquet preferred (smaller, faster), CSV as fallback
    output_path = PROCESSED_DIR / "faers_master.parquet"
    try:
        df.to_parquet(str(output_path), index=False)
        print(f"\n  💾 Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        
        # Also save a small sample for local testing
        sample_path = PROCESSED_DIR / "faers_sample_1000.parquet"
        df.sample(min(1000, len(df)), random_state=42).to_parquet(str(sample_path), index=False)
        print(f"  💾 Saved sample: {sample_path}")
    except ImportError:
        print("\n  ⚠️  pyarrow not installed — falling back to CSV export")
        print("     Install pyarrow: pip install pyarrow>=14.0.0")
        output_path = PROCESSED_DIR / "faers_master.csv"
        df.to_csv(str(output_path), index=False)
        print(f"  💾 Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        
        sample_path = PROCESSED_DIR / "faers_sample_1000.csv"
        df.sample(min(1000, len(df)), random_state=42).to_csv(str(sample_path), index=False)
        print(f"  💾 Saved sample: {sample_path}")
    
    conn.close()
    
    print(f"\n  ✅ Preprocessing complete! Next step:")
    print(f"     python src/data/03_build_training_data.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
