# Gemini DeepSearch (Test Project)

> **Test project** — this repo is an experimental “deep research” CLI that recursively explores a topic using **Google Gemini** (with the **Google Search** tool), tracks progress as a tree, extracts learnings, stores sources, and generates a final Markdown report.

## What this does

Given a starting query, the script:

1. **Chooses research breadth + depth** (how many parallel sub-queries and how many recursion levels) using Gemini.
2. **Searches the web via Gemini + Google Search tool** for each query.
3. **Extracts learnings** and generates **follow-up questions** from the search result text.
4. **Recursively explores** follow-up questions up to the configured depth.
5. **Tracks everything** in a `ResearchProgress` structure:
   - query tree (parent/child relationships)
   - completion status
   - learnings per query
   - sources per query (URL + title)
6. **Outputs**
   - a JSON research tree file (`research_tree_<...>.json`)
   - an optional Markdown final report (`--output report.md`)
   - console progress updates

## Modes

The CLI supports 3 modes:

- `fast` — smaller exploration, fewer learnings, quicker run
- `balanced` — default tradeoff
- `comprehensive` — more exhaustive exploration and extraction

Modes influence:
- depth/breadth determination prompt
- number of learnings extracted per query
- follow-up expansion behavior

## Requirements

- Python 3.10+ recommended
- A Google Gemini API key available as:
  - `GOOGLE_API_KEY` environment variable **or**
  - passed via `--api-key`

### Python dependencies

At minimum, the script expects:

- `python-dotenv`
- `pydantic`
- Google Gemini SDK packages used by your imports (`google-genai` / `google-generativeai` depending on your environment)

> Note: Your code imports both `from google import genai` and `from google.generative_ai ...`.
> Make sure your environment matches the library versions you’re using.

## Setup

1. Create a virtual env (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
````

2. Install dependencies (example):

   ```bash
   pip install -U python-dotenv pydantic
   # plus the relevant Google Gemini SDK packages for your environment
   ```

3. Add `.env` (optional) with:

   ```bash
   GOOGLE_API_KEY="YOUR_KEY_HERE"
   ```

## Usage

Basic run:

```bash
python deep_search.py "What are the latest trends in retrieval augmented generation?"
```

Choose a mode:

```bash
python deep_search.py "India fintech regulation overview" --mode comprehensive
```

Provide API key directly:

```bash
python deep_search.py "Open-source vector databases comparison" --api-key "YOUR_KEY"
```

Save the final report to a Markdown file:

```bash
python deep_search.py "Climate risk disclosure standards" --output report.md
```

## Outputs

### 1) Progress logs (console)

The script prints progress events such as:

* started query
* learnings added
* sources added
* completed query
* overall progress %

### 2) Research tree JSON

At the end, it writes a file like:

* `research_tree_<query>_<timestamp>.json`

This includes:

* query nodes (with UUIDs)
* parent/child links
* status
* learnings
* sources

### 3) Final report (Markdown)

The script generates a narrative report synthesizing all learnings and appends a source list.

If you pass `--output`, it saves the report to that path.

## How it works (high level)

### Key components

* **`ResearchProgress`**

  * Tracks each query’s status, learnings, sources, parent/child relationships
  * Builds a hierarchical tree structure for reporting/export

* **`DeepSearch`**

  * Gemini client wrapper + configuration
  * Determines initial depth/breadth
  * Runs recursive search + extraction (`_research_recursive`)
  * Calls Gemini with optional JSON schema parsing via Pydantic models

### Schema-backed model outputs

The script uses Pydantic models to validate structured JSON responses for:

* initial research parameters
* generated query lists
* extracted learnings and follow-up questions
* similarity checks (optional)

## Notes / limitations

* **Experimental (test project):** expect rough edges.
* **API costs & rate limits:** recursion can trigger many calls depending on depth/breadth.
* **Event loop caveat:** `determine_research_breadth_and_depth()` uses `asyncio.run()` internally; if you ever call it from an already-running async loop (e.g., inside another async app), you’ll need to refactor that part to be fully async.
* **Search sources extraction:** source extraction relies on the shape of Gemini grounding metadata. This can vary by SDK/version.

## Security

* Your API key should never be committed to git.
* Prefer using `.env` (ignored by git) or CI secrets.
