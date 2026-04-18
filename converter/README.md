# PDF Converter

Converts reference papers (PDF) to structured Markdown & JSON using [OpenDataLoader-PDF](https://opendataloader.org/) for easier reading and analysis.

## Prerequisites

| Dependency | Version |
|------------|---------|
| Python     | ≥ 3.10  |
| Java       | ≥ 11    |

## Install

```bash
pip install -U opendataloader-pdf
```

## Usage

1. Place PDF files in `../paper/`
2. Run the converter:

```bash
cd converter
python3 parse.py
```

3. Output files will be generated in `../ref/`:
   - `*.md` — Markdown (human-readable, good for AI context)
   - `*.json` — Structured JSON (programmatic access)

## Directory Structure

```
ProtoSAM+Amodal/
├── paper/          # Input PDFs (git-ignored)
├── ref/            # Parsed output (git-ignored)
└── converter/
    ├── parse.py    # Conversion script
    └── README.md
```

## Notes

- Each `convert()` call spawns a JVM process, so batch multiple files in a single call for speed.
- Unicode glyph warnings (e.g. math symbols) are normal and can be ignored.
