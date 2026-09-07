# Movie Recommender

A small movie website that figures out what you like **while you browse it** — no sign-up, no ratings to fill in, no "tell us your favourite genres" questionnaire.

You search for something, you click on a couple of movies, and the suggestions quietly rearrange themselves to match. Reload the page and it starts over from a neutral "Trending Now" list.

Everything runs on your own laptop. There is no paid API, no cloud account, and no API key needed to try it.

---

## Table of contents

- [What it actually does](#what-it-actually-does)
- [How it works, in plain English](#how-it-works-in-plain-english)
- [Installation](#installation)
- [Running it](#running-it)
- [Try it out](#try-it-out)
- [Project layout](#project-layout)
- [The API](#the-api)
- [Optional extras](#optional-extras)
- [Troubleshooting](#troubleshooting)
- [Notes for anyone reusing this](#notes-for-anyone-reusing-this)

---

## What it actually does

### 1. Search that understands meaning, not just words

Ordinary search looks for your words inside the text. This one compares *meaning*, so it finds the right film even when none of your words appear anywhere in its description.

Type **"assassin who falls in love with his target"** and you get the right movies back — even though the word "assassin" may not be in a single one of their plot summaries.

### 2. Recommendations that change as you browse

The site keeps a running picture of what you seem interested in, built from two things:

- **what you search for** — the words you type
- **what you click on** — the movies you open

Every search and every click updates that picture immediately. Your most recent three actions count double, so if you change direction mid-session the suggestions follow you rather than being stuck on what you did ten minutes ago.

### 3. It never shows you an empty box

The section at the top always has something in it, and the heading always tells you honestly which kind of list you're looking at:

| Heading | When you see it | What's in it |
|---|---|---|
| **Trending Now** | You've just arrived, no history yet | The highest-rated movies in the catalog — the same for everybody |
| **Recommended for you** | You've searched or clicked something | Movies matched to your activity, with a line underneath saying why |

That subtitle is the honest bit. It says things like *"Based on your search for "space adventure" and 2 movies you viewed"* — so you can always see what the suggestions were built from.

---

## How it works, in plain English

The trick is turning both **movies** and **your behaviour** into the same kind of thing: a list of 384 numbers. Once they're the same kind of thing, "what should we recommend?" becomes a simple question of which numbers are closest together.

```
                     ┌─────────────────────────────────────────────┐
  SET UP ONCE        │                                             │
                     │   399 movies                                │
                     │       ↓                                     │
                     │   Mash each one's genre, director, cast     │
                     │   and plot into a single block of text      │
                     │       ↓                                     │
                     │   An AI model reads that text and turns     │
                     │   it into 384 numbers ("an embedding")      │
                     │       ↓                                     │
                     │   Store all 399 in a local database         │
                     │                                             │
                     └─────────────────────────────────────────────┘

                     ┌─────────────────────────────────────────────┐
  EVERY TIME YOU     │                                             │
  DO SOMETHING       │   You search "space adventure"              │
                     │       → same model, → 384 numbers           │
                     │                                             │
                     │   You click a movie                         │
                     │       → reuse that movie's stored numbers   │
                     │                                             │
                     │       ↓                                     │
                     │   Average them all together, with your      │
                     │   last 3 actions counted double             │
                     │       ↓                                     │
                     │   = your "interest vector"                  │
                     │       ↓                                     │
                     │   Find the movies whose numbers sit         │
                     │   closest to it → your recommendations      │
                     │                                             │
                     └─────────────────────────────────────────────┘
```

A few details worth knowing:

- **Nothing is re-computed unnecessarily.** When you click a movie, its numbers are already stored, so they're looked up rather than recalculated.
- **Your history is remembered per browser**, using an anonymous session cookie. No name, no email, no account.
- **A movie you just clicked is never recommended back to you** — it would trivially be the closest match to itself, which isn't a recommendation.
- **If the search index is missing or broken, the site still works.** It quietly falls back to "Trending Now" instead of showing an error.

### The pieces it's built from

| Piece | What it's for |
|---|---|
| **FastAPI** | The web server and the JSON API |
| **all-MiniLM-L6-v2** | The AI model that turns text into numbers. Small (~90 MB), runs locally, free |
| **ChromaDB** | A local database built for searching by "closeness" instead of by keyword |
| **SQLite** | Stores your searches and clicks in a single `events.db` file |
| **Plain HTML, CSS and JavaScript** | The frontend. No build step, no npm, no framework |

---

## Installation

### Before you start

| You need | Notes |
|---|---|
| **Python 3.10 or newer** | Check with `python3 --version` |
| **About 2 GB of free disk space** | Most of it is PyTorch, which the AI model needs |
| **An internet connection for setup only** | To install packages and download the 90 MB model. After that it runs fully offline |
| **An API key** | **Not needed.** The movie catalog is already included in this repo |

### Step 1 — Get the code

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### Step 2 — Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your system. It's a folder called `venv` that you can safely delete later.

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked because your prompt now starts with `(venv)`.

> You need to run the `activate` line again each time you open a new terminal.

### Step 3 — Install the packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This is the slow step — **expect 3 to 10 minutes** and a fairly large download, because PyTorch comes along with the AI model.

<details>
<summary>Want the exact versions this was developed with?</summary>

`requirements.txt` uses minimum versions so it installs on any Python 3.10+. If you'd rather have the precise set of versions the project was built and tested against, use the lock file instead:

```bash
pip install -r requirements-lock.txt
```

Be aware it's pinned against Python 3.14, so it may not resolve on older Python versions.

</details>

### Step 4 — Build the search index

Three commands. They read the included movie catalog and produce the database the recommendations are looked up in.

```bash
python build_soup.py --save
python embed_movies.py
python load_chroma.py
```

What each one is doing:

| Command | What it does | Roughly how long |
|---|---|---|
| `build_soup.py --save` | Combines each movie's genre, director, cast and plot into one block of text | A second |
| `embed_movies.py` | Turns all 399 blocks of text into numbers. **Downloads the ~90 MB model the first time** | 1–2 seconds, plus the download |
| `load_chroma.py` | Loads those numbers into the local ChromaDB database | A few seconds |

Each script prints a summary with `PASS` checks at the end. If you see `PASS` lines, that step worked.

> You only ever do Step 4 once. The index is saved to disk and reused.

### Step 5 — Start it

```bash
uvicorn app:app --reload --port 8000
```

Then open **<http://localhost:8000>** in your browser.

Press `Ctrl+C` in the terminal to stop the server.

---

## Running it

Once installed, starting the site is just two lines:

```bash
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
uvicorn app:app --reload --port 8000
```

The `--reload` flag restarts the server automatically when you edit a file, which is handy while developing. Leave it off if you just want to use the site.

> **Heads up:** the very first recommendation after starting the server takes a few seconds, because the AI model is being loaded into memory. Every request after that is fast.

---

## Try it out

This is the quickest way to see the whole thing work:

1. **Open <http://localhost:8000>.**
   The top section says **"Trending Now"** and is filled with the highest-rated movies. Nothing is personalised yet, and the label says so.

2. **Search for `space adventure`.**
   The heading changes to **"Recommended for you"**, the list fills with space films, and a line appears underneath: *"Based on your search for "space adventure""*.

3. **Click on a movie or two** — try a couple of romances.
   The suggestions shift towards what you clicked, and the subtitle updates to mention the movies you viewed. The space films slide down but don't vanish entirely: it's blending your whole session, weighted towards what you did most recently.

4. **Click "start a new session"** in the small grey line under the title.
   Your history is wiped, you get a fresh anonymous session, and the heading drops back to **"Trending Now"** — ready to test again from scratch.

Want to see the machinery? These two URLs open in any browser:

- **<http://localhost:8000/api/debug/events>** — every search and click recorded for your session
- **<http://localhost:8000/api/debug/interest>** — what your interest vector was built from, and the weight given to each action

---

## Project layout

```
├── app.py                  the web server and all the API endpoints
├── interest.py             builds your interest vector, finds nearest movies
├── tracking.py             records searches and clicks into events.db
│
├── static/
│   ├── index.html          the page
│   ├── app.js              all the frontend behaviour
│   └── style.css           the styling
│
├── movies.json             the movie catalog (included — 399 movies)
│
│   ── setup pipeline, run once ──
├── build_soup.py           combines each movie's fields into one text block
├── embed_movies.py         turns those text blocks into numbers
├── load_chroma.py          loads the numbers into ChromaDB
│
│   ── optional extras ──
├── fetch_movies.py         rebuild movies.json from OMDb (needs a free key)
├── vectorize.py            the older keyword-based (TF-IDF) approach
├── semantic_test.py        proves the AI model beats keyword matching
├── recommend.py            search the catalog from the command line
├── test_setup.py           checks your environment and OMDb key
│
├── requirements.txt        the packages you need
└── requirements-lock.txt   exact versions, for reproducible installs
```

Files created when you run things, which are **not** in this repo (they're all rebuildable):

| Created | By | What it is |
|---|---|---|
| `movies_with_soup.json` | `build_soup.py --save` | The text blocks |
| `model/embeddings.pkl` | `embed_movies.py` | The numbers |
| `chroma_db/` | `load_chroma.py` | The search database |
| `events.db` | the app, as you browse | Your searches and clicks |

---

## The API

Every endpoint returns JSON, and you can open the `GET` ones directly in a browser.

| Endpoint | What it gives you |
|---|---|
| `GET /api/recommendations` | Your recommendations, plus the heading and subtitle that describe them |
| `GET /api/movies` | One page of the catalog. Supports `?page=`, `?per_page=`, `?language=`, `?genre=` |
| `GET /api/movies/{imdbID}` | Everything about one movie |
| `GET /api/facets` | The available languages and genres, with counts, for the dropdowns |
| `GET /api/search?q=...` | Search the catalog, and record that you searched |
| `POST /api/event` | Record that you viewed a movie — `{"type": "view", "imdbID": "tt0314331"}` |
| `POST /api/session/reset` | Clear your history and start a fresh session. Add `?purge=false` to keep the old records |
| `GET /api/debug/events` | Everything recorded for your session |
| `GET /api/debug/interest` | How your interest vector was built, action by action |
| `GET /api/health` | A simple "is it running?" check |

`GET /api/recommendations` is the interesting one. It always tells you which mode you're in:

```json
{
  "mode": "personalized",
  "title": "Recommended for you",
  "subtitle": "Based on your search for “space adventure” and 2 movies you viewed",
  "movies": [
    { "imdbID": "tt0356910", "title": "The Family Stone", "score": 0.598, "...": "..." }
  ]
}
```

`mode` is either `"trending"` or `"personalized"`. The frontend renders `title` and `subtitle` exactly as given, so the label is always decided by the same code that decided what's in the list — the page can never mislabel it.

Interactive API docs are built in: **<http://localhost:8000/docs>**

---

## Optional extras

None of these are needed to run the site.

### See for yourself that meaning-matching beats keyword-matching

```bash
python vectorize.py        # build the old keyword-based model to compare against
python semantic_test.py    # run both against the same questions
```

`semantic_test.py` asks things like *"a boy uses a modified car to visit the era when his parents were young"* and checks whether each approach can find **Back to the Future** — a film whose description contains almost none of those words.

### Search from the command line

```bash
python recommend.py "heist gone wrong"
python recommend.py --interactive
```

### Inspect a session's interest vector

```bash
python interest.py             # the most recent session
python interest.py <session_id>
```

Prints each action, the weight it was given, and the resulting recommendations.

### Build your own movie catalog

The included `movies.json` has 399 movies (1934–2026, 57 languages, 22 genres). To build your own instead:

1. Get a free API key from [omdbapi.com](https://www.omdbapi.com/apikey.aspx) — it arrives by email in a minute.
2. Create a `.env` file. There's a template to copy:
   ```bash
   cp .env.example .env
   ```
   Then put your key in it:
   ```
   OMDB_API_KEY=your_key_here
   ```
3. Check it's working, then fetch:
   ```bash
   python test_setup.py
   python fetch_movies.py --refresh
   ```
4. Re-run Step 4 of the installation to rebuild the index from your new catalog.

> `.env` is listed in `.gitignore`, so your key won't be committed. Keep it that way.

---

## Troubleshooting

**`chroma_db/ not found — run load_chroma.py first`**
Step 4 of the installation hasn't been done, or was only partly done. Run all three commands in order.

**The recommendations never change from "Trending Now"**
Your session probably has no history the app can use. Open `/api/debug/interest` to see what it recorded. If `has_vector` is `false` and `events_used` is `0`, your searches and clicks aren't being logged — check your browser isn't blocking cookies for `localhost`.

**The first recommendation takes several seconds**
Expected. The AI model loads into memory on first use. Everything after that is fast.

**`Address already in use`**
Something is already on port 8000. Either use another port with `--port 8001`, or stop the other process:
```bash
kill $(lsof -ti:8000)          # macOS / Linux
```

**`pip install` fails or takes forever**
It's downloading PyTorch, which is large. If it fails outright, upgrade pip first (`pip install --upgrade pip`) and confirm you're on Python 3.10+ with `python3 --version`.

**I want to wipe everything and start clean**
```bash
rm -rf chroma_db model events.db movies_with_soup.json
```
Then re-run Step 4. Nothing irreplaceable is in those — they're all rebuildable from `movies.json`.

**Old test data is polluting my results**
Click **"start a new session"** on the page. That clears your session's history and gives you a fresh one without touching anyone else's.

---

## Notes for anyone reusing this

- **No secrets are in this repository.** `.env` is gitignored; the app itself doesn't need an API key at all.
- **`events.db` is not committed.** It's created locally as you browse, and holds only anonymous session ids, search terms, and movie ids.
- **The movie data came from the [OMDb API](https://www.omdbapi.com/).** It's included here so the project runs without a key. If you plan to redistribute it more widely, have a look at OMDb's terms first.
- **The AI model** is [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), downloaded from Hugging Face on first use and cached in your home directory.
- **This is a learning project, not production software.** In particular: the debug endpoints expose session data and should be removed before any real deployment, the session cookie isn't `httponly` (the frontend reads it), and CORS is open to any `localhost` port.
