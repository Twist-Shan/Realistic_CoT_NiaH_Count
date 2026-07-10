# Paul Graham haystack sources

## Direct haystack files (copied)
- `gkamradt_founders.txt`: copied from
  `https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/needlehaystack/PaulGrahamEssays/founders.txt`
- `gkamradt_worked.txt`: copied from
  `https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/needlehaystack/PaulGrahamEssays/worked.txt`

## RULER reference source (copied)
- `ruler_paulgraham_urls.txt`: copied from
  `https://github.com/NVIDIA/RULER/blob/main/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt`

RULER uses this URL list to construct essay haystacks in its data prep scripts.


## Full essay sync (May 20, 2026)
- Added `scripts/sync_paul_graham_essays.py` to fetch all `.txt` files from
  `gkamradt/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays`
  into this directory.
- Generated `index.jsonl` with source URL and SHA256 per file for provenance.
