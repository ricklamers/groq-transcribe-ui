## Groq Transcribe UI (Qt)

A PySide6 desktop app to transcribe large audio files with Groq STT, handling chunking, parallel uploads, and LLM-based stitching. Stores and reuses your GROQ_API_KEY.

### Setup

1. Create a virtualenv and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure `ffmpeg` is installed (pydub uses it under the hood):
```bash
# macOS (brew)
brew install ffmpeg
```

### Run

```bash
python app.py
```

### Notes
- API key is stored in your user config directory. You can update it from within the app.
- Chunking targets ~10MB per chunk with 1s overlap; uploads run in parallel for speed.
- Final transcript is stitched with an LLM to remove boundary duplication and improve flow.
