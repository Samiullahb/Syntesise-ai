# SynThesisAI

A Gemini-powered multi-agent content synthesis and quality-checking pipeline.

## Architecture

- **Engineer Agent** — turns a topic/domain request into a high-quality problem/task.
- **Checker Agent** — independently validates correctness, clarity, relevance, and completeness.
- **Target Agent** — produces the final answer/solution after reviewing the checker feedback.
- **Gemini client** — one reusable API layer built on Google's `google-genai` SDK.
- **Batch runner** — concurrent processing with configurable worker limits.
- **JSON output** — reproducible batch results saved under `results/`.

The project uses Gemini models only. No OpenAI API is required.

## Default model

All three agents use **`gemini-3.1-pro-preview`** by default:

- Engineer Agent: Gemini 3.1 Pro Preview
- Checker Agent: Gemini 3.1 Pro Preview
- Target Agent: Gemini 3.1 Pro Preview

The application also protects against the retired Gemini 2.5 model IDs by automatically replacing those legacy IDs with Gemini 3.1 Pro Preview.

## Requirements

- Python 3.11+
- A Gemini API key with access to Gemini 3.1 Pro Preview

Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example` and set:

```env
GEMINI_API_KEY=your_new_key_here
```

Never commit `.env` or an API key. The application reads `GEMINI_API_KEY` from the environment.

## Run

Single request:

```bash
python -m synthesise --topic "Algebra" --domain mathematics
```

Batch:

```bash
python -m synthesise --topic "Create 10 probability problems" --domain mathematics --count 10
```

The CLI defaults to `gemini-3.1-pro-preview`. Advanced users can override the model explicitly:

```bash
python -m synthesise --topic "Cloud security" --domain technology --engineer-model gemini-3.1-pro-preview --checker-model gemini-3.1-pro-preview --target-model gemini-3.1-pro-preview
```

## Configuration

Edit `config/settings.yaml` to control models, concurrency, output paths, and quality thresholds.

## Testing

The test suite mocks the Gemini API, so it does not require a live key:

```bash
pytest -q
```

## Security

- `.env` is ignored by Git.
- Only `.env.example` is committed.
- API keys are never written to results.
- Do not paste API keys into source files, README files, issues, or commits.

## License

MIT
