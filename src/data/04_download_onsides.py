"""
OnSIDES Database Processor — Ground truth for Task 3 (Labelling).

Builds a lookup table from OnSIDES database (adverse drug effects extracted
from FDA labels using PubMedBERT NLP).

Source: https://github.com/tatonetti-lab/onsides
Paper: Tatonetti Lab, Columbia University

Why OnSIDES instead of DailyMed string matching:
- FAERS uses MedDRA PTs (e.g., "Gastrointestinal haemorrhage")
- Drug labels use free text (e.g., "bleeding in the stomach")
- OnSIDES already extracted + mapped label text → MedDRA PTs via PubMedBERT
- Direct PT-to-PT lookup, no fuzzy matching needed

Supports two input modes:
  1. Auto-download from GitHub (attempts zip + CSV URLs)
  2. Local CSV files manually placed in data/external/
     Required files: product_adverse_effect.csv, product_label.csv,
     vocab_meddra_adverse_effect.csv

OnSIDES v3.1.1 uses a NORMALIZED schema with 7 tables:
  - product_label: label_id → application_number (NDA)
  - product_adverse_effect: label_id → meddra_id (link table)
  - vocab_meddra_adverse_effect: meddra_id → pt_meddra_term (names)
  These 3 must be joined to produce our lookup: NDA → PT name.

Output: data/external/onsides_lookup.parquet (or .csv if pyarrow missing)
  Columns: [nda_num, pt_meddra_term, label_section]
"""

import os
import sys
import zipfile
import io
import pandas as pd
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# Configuration
# ============================================================

# OnSIDES v3.1.1 release — zip containing all 8 CSV files
# Individual CSVs are LFS-tracked on GitHub and CANNOT be downloaded via raw URLs.
# The zip release is the ONLY reliable download method.
ONSIDES_RELEASE_URL = "https://github.com/tatonetti-lab/onsides/releases/download/v3.1.1/onsides-v3.1.1.zip"

OUTPUT_DIR = Path("data/external")
OUTPUT_FILE = OUTPUT_DIR / "onsides_lookup.parquet"


# ============================================================
# Drug Name Normalization
# ============================================================

import re

# Common dosage forms, routes, and noise words to strip from drug names
_DOSAGE_FORMS = re.compile(
    r'\b(tablet|tablets|capsule|capsules|injection|injectable|solution|'
    r'suspension|cream|ointment|gel|patch|patches|spray|inhaler|'
    r'suppository|drops|syrup|elixir|powder|granules|lotion|'
    r'film|wafer|chewable|extended.?release|delayed.?release|'
    r'sustained.?release|modified.?release|controlled.?release|'
    r'immediate.?release|oral|topical|intravenous|subcutaneous|'
    r'intramuscular|ophthalmic|nasal|rectal|vaginal|transdermal|'
    r'sublingual|buccal|inhalation|nebulizer|for\s+injection|'
    r'lyophilized|reconstituted|concentrate|prefilled|auto.?injector|'
    r'vial|syringe|kit|pack|blister|bottle|tube|ampule|ampoule)\b',
    re.IGNORECASE
)

_STRENGTH_PATTERN = re.compile(
    r'\b\d+[\.,]?\d*\s*(?:mg|mcg|g|ml|%|iu|units?|meq|mmol)\b',
    re.IGNORECASE
)

_NOISE = re.compile(
    r'[,/\(\)\[\]\{\}]|^\s+|\s+$'
)

def _normalize_drug_name(name: str) -> str:
    """Normalize a drug name for fuzzy matching.
    
    Strips dosage forms, strengths, routes, and punctuation:
      "ASPIRIN TABLETS, 325 MG"      → "aspirin"
      "BAYER ASPIRIN"                → "bayer aspirin"
      "METFORMIN HCL 500MG TABLETS"  → "metformin hcl"
    """
    if not name or name == 'nan':
        return ''
    
    name = name.lower().strip()
    name = _STRENGTH_PATTERN.sub('', name)
    name = _DOSAGE_FORMS.sub('', name)
    name = _NOISE.sub(' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


# ============================================================
# Strategy 1: Build from locally placed v3.1.1 CSV files
# ============================================================

def try_local_v3_csvs() -> pd.DataFrame | None:
    """Try to build lookup from manually placed OnSIDES v3.x CSV files.
    
    OnSIDES v3.1.1 uses a normalized schema with 7 CSV files.
    Actual column names (verified from user's data):
      - product_adverse_effect.csv: product_label_id, effect_meddra_id, label_section, ...
      - product_label.csv: label_id, source, source_product_name, source_product_id, ...
      - vocab_meddra_adverse_effect.csv: meddra_id, meddra_name, meddra_term_type
    
    IMPORTANT: product_label has NO NDA/application_number column.
    Instead, we build two lookup strategies:
      1. source_product_name → PT name (match by drug name)
      2. Use product_to_rxnorm + vocab_rxnorm_ingredient for active ingredient matching
    """
    pae_path = OUTPUT_DIR / "product_adverse_effect.csv"
    pl_path = OUTPUT_DIR / "product_label.csv"
    vocab_path = OUTPUT_DIR / "vocab_meddra_adverse_effect.csv"
    
    if not pae_path.exists() or not pl_path.exists() or not vocab_path.exists():
        return None
    
    print("\n  📁 Found local OnSIDES v3.x CSV files:")
    
    # Load the 3 required tables
    pae_df = pd.read_csv(pae_path)
    print(f"    product_adverse_effect.csv: {len(pae_df):,} rows")
    print(f"      Columns: {list(pae_df.columns)}")
    
    pl_df = pd.read_csv(pl_path)
    print(f"    product_label.csv: {len(pl_df):,} rows")
    print(f"      Columns: {list(pl_df.columns)}")
    
    vocab_df = pd.read_csv(vocab_path)
    print(f"    vocab_meddra_adverse_effect.csv: {len(vocab_df):,} rows")
    print(f"      Columns: {list(vocab_df.columns)}")
    
    # Lowercase all column names for uniformity
    pae_df.columns = [c.lower() for c in pae_df.columns]
    pl_df.columns = [c.lower() for c in pl_df.columns]
    vocab_df.columns = [c.lower() for c in vocab_df.columns]
    
    # ---- Identify column names ----
    
    # PAE: label ID
    label_id_col_pae = None
    for c in ['product_label_id', 'label_id']:
        if c in pae_df.columns:
            label_id_col_pae = c
            break
    
    # PAE: MedDRA ID (v3.1.1 uses 'effect_meddra_id')
    meddra_id_col = None
    for c in ['effect_meddra_id', 'pt_meddra_id', 'meddra_id', 'concept_id']:
        if c in pae_df.columns:
            meddra_id_col = c
            break
    
    # PAE: section
    section_col = None
    for c in ['label_section', 'section', 'section_name']:
        if c in pae_df.columns:
            section_col = c
            break
    
    # PL: label ID
    label_id_col_pl = None
    for c in ['label_id', 'product_label_id', 'id']:
        if c in pl_df.columns:
            label_id_col_pl = c
            break
    
    # PL: drug name (v3.1.1 has source_product_name, NO NDA column)
    drug_name_col = None
    for c in ['source_product_name', 'product_name', 'drug_name', 'name']:
        if c in pl_df.columns:
            drug_name_col = c
            break
    
    # PL: source_product_id (may contain NDA-like info)
    source_id_col = None
    for c in ['source_product_id', 'application_number', 'nda_num', 'nda', 'appl_no']:
        if c in pl_df.columns:
            source_id_col = c
            break
    
    # Vocab: MedDRA ID
    meddra_id_col_vocab = None
    for c in ['meddra_id', 'pt_meddra_id', 'concept_id', 'id']:
        if c in vocab_df.columns:
            meddra_id_col_vocab = c
            break
    
    # Vocab: PT name (v3.1.1 uses 'meddra_name')
    pt_name_col = None
    for c in ['meddra_name', 'pt_meddra_term', 'concept_name', 'meddra_term', 'term', 'name']:
        if c in vocab_df.columns:
            pt_name_col = c
            break
    
    print(f"\n    Identified join columns:")
    print(f"      PAE label_id: {label_id_col_pae}")
    print(f"      PAE meddra_id: {meddra_id_col}")
    print(f"      PAE section: {section_col}")
    print(f"      PL label_id: {label_id_col_pl}")
    print(f"      PL drug_name: {drug_name_col}")
    print(f"      PL source_id: {source_id_col}")
    print(f"      Vocab meddra_id: {meddra_id_col_vocab}")
    print(f"      Vocab PT name: {pt_name_col}")
    
    # Minimum required: PAE label+meddra, PL label+drugname, Vocab id+name
    if not all([label_id_col_pae, meddra_id_col, label_id_col_pl, 
                meddra_id_col_vocab, pt_name_col]):
        print("    ❌ Could not identify minimum required columns")
        print(f"       Missing: label_pae={label_id_col_pae}, meddra={meddra_id_col}, "
              f"label_pl={label_id_col_pl}, vocab_id={meddra_id_col_vocab}, pt_name={pt_name_col}")
        return None
    
    if not drug_name_col and not source_id_col:
        print("    ❌ product_label has no drug name or source ID column")
        return None
    
    # ---- 3-way JOIN ----
    print("\n    Performing 3-way join...")
    
    # Join 1: PAE + Product Label → get drug name
    pl_cols_to_use = [label_id_col_pl]
    if drug_name_col:
        pl_cols_to_use.append(drug_name_col)
    if source_id_col:
        pl_cols_to_use.append(source_id_col)
    
    merged = pae_df.merge(
        pl_df[pl_cols_to_use].drop_duplicates(),
        left_on=label_id_col_pae,
        right_on=label_id_col_pl,
        how='inner'
    )
    print(f"      After PAE ⋈ PL: {len(merged):,} rows")
    
    # Join 2: merged + Vocab → get PT name
    merged = merged.merge(
        vocab_df[[meddra_id_col_vocab, pt_name_col]].drop_duplicates(),
        left_on=meddra_id_col,
        right_on=meddra_id_col_vocab,
        how='inner'
    )
    print(f"      After ⋈ Vocab: {len(merged):,} rows")
    
    # ---- Try to also load RxNorm ingredient mapping for prod_ai matching ----
    rxnorm_path = OUTPUT_DIR / "product_to_rxnorm.csv"
    ingredient_prod_path = OUTPUT_DIR / "vocab_rxnorm_ingredient_to_product.csv"
    ingredient_path = OUTPUT_DIR / "vocab_rxnorm_ingredient.csv"
    
    ingredient_map = None
    if rxnorm_path.exists() and ingredient_path.exists():
        try:
            p2r = pd.read_csv(rxnorm_path)
            p2r.columns = [c.lower() for c in p2r.columns]
            ing_df = pd.read_csv(ingredient_path)
            ing_df.columns = [c.lower() for c in ing_df.columns]
            
            print(f"    Loading RxNorm ingredient mapping for prod_ai matching...")
            print(f"      product_to_rxnorm.csv: {len(p2r):,} rows, cols: {list(p2r.columns)}")
            print(f"      vocab_rxnorm_ingredient.csv: {len(ing_df):,} rows, cols: {list(ing_df.columns)}")
            
            # Try to find ingredient name column
            ing_name_col = None
            for c in ['ingredient_name', 'name', 'rxnorm_name']:
                if c in ing_df.columns:
                    ing_name_col = c
                    break
            
            if ing_name_col and ingredient_prod_path.exists():
                i2p = pd.read_csv(ingredient_prod_path)
                i2p.columns = [c.lower() for c in i2p.columns]
                print(f"      vocab_rxnorm_ingredient_to_product.csv: {len(i2p):,} rows, cols: {list(i2p.columns)}")
                
                # Find rxnorm product ID columns
                rx_prod_col_p2r = None
                for c in ['rxnorm_product_id', 'product_id', 'rxcui']:
                    if c in p2r.columns:
                        rx_prod_col_p2r = c
                        break
                
                label_col_p2r = None
                for c in ['product_label_id', 'label_id']:
                    if c in p2r.columns:
                        label_col_p2r = c
                        break
                
                # Find columns in i2p
                rx_prod_col_i2p = None
                for c in ['rxnorm_product_id', 'product_id']:
                    if c in i2p.columns:
                        rx_prod_col_i2p = c
                        break
                ing_id_col = None
                for c in ['ingredient_id', 'rxnorm_ingredient_id']:
                    if c in i2p.columns:
                        ing_id_col = c
                        break
                ing_id_col_vocab = None
                for c in ['ingredient_id', 'rxnorm_ingredient_id', 'rxnorm_id', 'id']:
                    if c in ing_df.columns:
                        ing_id_col_vocab = c
                        break
                
                print(f"      Join keys: p2r[{label_col_p2r}, {rx_prod_col_p2r}] "
                      f"→ i2p[{rx_prod_col_i2p}, {ing_id_col}] "
                      f"→ ing[{ing_id_col_vocab}, {ing_name_col}]")
                
                if rx_prod_col_p2r and label_col_p2r and rx_prod_col_i2p and ing_id_col and ing_id_col_vocab:
                    step1 = p2r[[label_col_p2r, rx_prod_col_p2r]].merge(
                        i2p[[rx_prod_col_i2p, ing_id_col]],
                        left_on=rx_prod_col_p2r, right_on=rx_prod_col_i2p, how='inner'
                    )
                    print(f"      Step 1 (p2r ⋈ i2p): {len(step1):,} rows")
                    
                    ingredient_map = step1.merge(
                        ing_df[[ing_id_col_vocab, ing_name_col]],
                        left_on=ing_id_col, right_on=ing_id_col_vocab, how='inner'
                    )[[label_col_p2r, ing_name_col]].drop_duplicates()
                    ingredient_map.columns = ['label_id', 'ingredient_name']
                    print(f"      ✅ Ingredient mapping: {len(ingredient_map):,} label→ingredient pairs")
                    if len(ingredient_map) > 0:
                        print(f"      Sample: {ingredient_map.head(3).to_string(index=False)}")
                else:
                    missing = []
                    if not rx_prod_col_p2r: missing.append("rx_prod_col (p2r)")
                    if not label_col_p2r: missing.append("label_col (p2r)")
                    if not rx_prod_col_i2p: missing.append("rx_prod_col (i2p)")
                    if not ing_id_col: missing.append("ing_id_col (i2p)")
                    if not ing_id_col_vocab: missing.append("ing_id_col (vocab)")
                    print(f"      ⚠️ Missing join keys: {missing}")
            else:
                if not ing_name_col:
                    print(f"      ⚠️ No ingredient name column found in {list(ing_df.columns)}")
                if not ingredient_prod_path.exists():
                    print(f"      ⚠️ Missing: {ingredient_prod_path}")
        except Exception as e:
            print(f"    ⚠️ Could not build ingredient mapping: {e}")
            import traceback
            traceback.print_exc()
    
    # ---- Build final lookup (vectorized — no iterrows on 6.9M rows!) ----
    
    lookup = pd.DataFrame()
    
    if drug_name_col and drug_name_col in merged.columns:
        lookup['drug_name'] = merged[drug_name_col].astype(str).str.strip().str.lower()
        # Normalized drug name: strip dosage forms, strengths for fuzzy matching
        lookup['drug_name_normalized'] = lookup['drug_name'].apply(_normalize_drug_name)
    
    if source_id_col and source_id_col in merged.columns:
        lookup['nda_num'] = merged[source_id_col].astype(str).str.strip()
    else:
        lookup['nda_num'] = ''
    
    lookup['pt_meddra_term'] = merged[pt_name_col].astype(str).str.strip().str.lower()
    
    if section_col and section_col in merged.columns:
        lookup['label_section'] = merged[section_col].values
    else:
        lookup['label_section'] = 'Adverse Reactions'
    
    # Add ingredient name via dictionary map (NOT left join — avoids row duplication)
    if ingredient_map is not None and label_id_col_pae in merged.columns:
        # Build label_id → ingredient_name dict (take first match per label)
        ing_dict = ingredient_map.drop_duplicates(subset='label_id').set_index('label_id')['ingredient_name'].to_dict()
        lookup['ingredient_name'] = merged[label_id_col_pae].map(ing_dict).str.strip().str.lower()
        n_with_ing = lookup['ingredient_name'].notna().sum()
        print(f"      Ingredient mapped: {n_with_ing:,} / {len(lookup):,} rows")
    
    return lookup


# ============================================================
# Strategy 2: Download from GitHub (old format)
# ============================================================

def download_and_extract_onsides() -> bool:
    """Download OnSIDES zip release and extract CSVs to data/external/.
    
    The zip release is the ONLY reliable download method for v3.1.1
    because individual CSVs are LFS-tracked on GitHub.
    
    Returns True if CSVs were extracted successfully.
    """
    import requests
    
    print(f"  📥 Downloading OnSIDES zip release...")
    print(f"     URL: {ONSIDES_RELEASE_URL}")
    
    try:
        resp = requests.get(ONSIDES_RELEASE_URL, timeout=600, allow_redirects=True,
                           stream=True)
        resp.raise_for_status()
        
        content = resp.content
        print(f"     Downloaded: {len(content) / 1024 / 1024:.1f} MB")
        
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            print(f"     Found {len(csv_files)} CSV files in zip:")
            
            for csv_file in csv_files:
                filename = Path(csv_file).name
                if not filename:
                    continue
                    
                target = OUTPUT_DIR / filename
                with zf.open(csv_file) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                print(f"       ✅ {filename}")
            
            print(f"     All CSVs extracted to {OUTPUT_DIR}/")
            return True
            
    except Exception as e:
        print(f"     ❌ Zip download failed: {e}")
        print(f"     💡 Manual fallback:")
        print(f"        1. Download from: https://github.com/tatonetti-lab/onsides/releases")
        print(f"        2. Extract the zip")
        print(f"        3. Copy CSV files from csv/ folder to data/external/")
    
    return False




def clean_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the lookup table for matching with FAERS."""
    
    # Clean NDA numbers if present
    if 'nda_num' in lookup.columns:
        lookup['nda_num'] = lookup['nda_num'].astype(str).str.strip()
        # Remove NDA/ANDA/BLA prefixes
        lookup['nda_num'] = lookup['nda_num'].str.replace(r'^(?:NDA|ANDA|BLA)\s*', '', regex=True)
        lookup['nda_num'] = lookup['nda_num'].str.replace(r'^0+', '', regex=True)
    
    # Normalize PT names
    if 'pt_meddra_term' in lookup.columns:
        lookup['pt_meddra_term'] = lookup['pt_meddra_term'].astype(str).str.strip().str.lower()
    
    # Normalize drug names if present
    if 'drug_name' in lookup.columns:
        lookup['drug_name'] = lookup['drug_name'].astype(str).str.strip().str.lower()
    
    # Normalize ingredient names if present
    if 'ingredient_name' in lookup.columns:
        lookup['ingredient_name'] = lookup['ingredient_name'].astype(str).str.strip().str.lower()
        # Replace 'nan' string with actual NaN
        lookup.loc[lookup['ingredient_name'] == 'nan', 'ingredient_name'] = pd.NA
    
    # Remove rows with missing PT
    lookup = lookup.dropna(subset=['pt_meddra_term'])
    lookup = lookup[lookup['pt_meddra_term'] != '']
    lookup = lookup[lookup['pt_meddra_term'] != 'nan']
    
    # Deduplicate — use drug_name+PT if available, otherwise nda_num+PT
    if 'drug_name' in lookup.columns:
        dedup_cols = ['drug_name', 'pt_meddra_term']
    else:
        dedup_cols = ['nda_num', 'pt_meddra_term']
    lookup = lookup.drop_duplicates(subset=dedup_cols)
    
    # Stats
    n_pts = lookup['pt_meddra_term'].nunique()
    n_pairs = len(lookup)
    
    print(f"\n    ✅ Lookup table built:")
    if 'drug_name' in lookup.columns:
        n_drugs = lookup['drug_name'].nunique()
        print(f"       Unique drugs (by name): {n_drugs:,}")
    if 'nda_num' in lookup.columns:
        valid_nda = lookup[lookup['nda_num'].notna() & (lookup['nda_num'] != '') & (lookup['nda_num'] != 'nan')]
        print(f"       Unique NDAs: {valid_nda['nda_num'].nunique():,}")
    if 'ingredient_name' in lookup.columns:
        valid_ing = lookup[lookup['ingredient_name'].notna()]
        print(f"       Unique ingredients: {valid_ing['ingredient_name'].nunique():,}")
    print(f"       Unique PTs: {n_pts:,}")
    print(f"       Total drug-PT pairs: {n_pairs:,}")
    
    return lookup


def main():
    print("=" * 60)
    print("  OnSIDES Database — T3 Ground Truth")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if output already exists
    csv_output = OUTPUT_DIR / "onsides_lookup.csv"
    if OUTPUT_FILE.exists():
        try:
            existing = pd.read_parquet(OUTPUT_FILE)
            print(f"\n  ℹ️ OnSIDES lookup already exists: {len(existing):,} rows")
            print(f"     To re-build, delete {OUTPUT_FILE}")
            return 0
        except ImportError:
            pass
    if csv_output.exists():
        existing = pd.read_csv(csv_output)
        print(f"\n  ℹ️ OnSIDES lookup already exists: {len(existing):,} rows")
        print(f"     To re-build, delete {csv_output}")
        return 0
    
    # Strategy 1: Try local v3.x CSV files (user manually downloaded)
    lookup = try_local_v3_csvs()
    
    if lookup is None:
        # Strategy 2: Try downloading from GitHub, then process locally
        print("\n  No local v3.x CSV files found, trying download...")
        downloaded = download_and_extract_onsides()
        
        if downloaded:
            # Re-try local processing with freshly extracted CSVs
            lookup = try_local_v3_csvs()
        
        if lookup is None:
            print(f"\n  ❌ Failed to build OnSIDES lookup")
            print(f"  💡 You can manually download from:")
            print(f"     https://github.com/tatonetti-lab/onsides/releases")
            print(f"     Extract the zip and copy CSV files to {OUTPUT_DIR}/")
            print(f"     Required: product_adverse_effect.csv, product_label.csv,")
            print(f"               vocab_meddra_adverse_effect.csv")
            print(f"\n  ⚠️ T3 will fall back to frequency heuristic without OnSIDES.")
            return 1
    
    # Clean and normalize
    lookup = clean_lookup(lookup)
    
    if len(lookup) == 0:
        print("  ❌ Lookup table is empty. T3 will use frequency heuristic.")
        return 1
    
    # Save
    try:
        lookup.to_parquet(OUTPUT_FILE, index=False)
        print(f"\n  ✅ Saved to: {OUTPUT_FILE}")
    except ImportError:
        lookup.to_csv(csv_output, index=False)
        print(f"\n  ⚠️  pyarrow not installed — saved CSV instead: {csv_output}")
        print(f"     Install pyarrow: pip install pyarrow>=14.0.0")
    
    print(f"\n  Next: Run python src/data/03_build_training_data.py")
    print(f"        T3 will automatically use OnSIDES for labelling ground truth.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
