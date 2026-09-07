# Movie Recommender

A small movie website that figures out what you like **while you browse it** — no sign-up, no ratings to fill in, no "tell us your favourite genres" questionnaire.

You search for something, you click on a couple of movies, and the suggestions quietly rearrange themselves to match. Reload the page and it starts over from a neutral "Trending Now" list.

Everything runs on your own laptop. There is no paid API, no cloud account, and no API key needed to try it.

## Install

You need **Python 3.10+** and about **2 GB of free disk space** (most of it is PyTorch, which the AI model needs).

**1. Get the code**

```bash
git clone https://github.com/sravanipamu/movie-recommender.git
cd movie-recommender
```

**2. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`. Run this line again in any new terminal.

**3. Install the packages** — the slow step, 3–10 minutes

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Want the exact versions this was developed with? Use `requirements-lock.txt` instead. It's pinned to Python 3.14, so it won't resolve on older versions.

**4. Build the search index** — you only ever do this once

```bash
python build_soup.py --save     # combine each movie's genre, cast and plot into one text block
python embed_movies.py          # turn those blocks into numbers (downloads a ~90 MB model)
python load_chroma.py           # load the numbers into a local database
```

Each script prints `PASS` checks when it finishes. If you see them, that step worked.

**5. Start it**

```bash
uvicorn app:app --reload --port 8000
```

Open **<http://localhost:8000>**. Press `Ctrl+C` to stop.

> The first recommendation takes a few seconds while the AI model loads into memory. Everything after that is fast.

## Try it

1. **Open the site.** The top section says **"Trending Now"** — the highest-rated movies, same for everybody. Nothing is personalised yet, and the label says so.

2. **Search for `space adventure`.** The heading changes to **"Recommended for you"**, space films fill the list, and a line appears underneath: *"Based on your search for "space adventure""*.

3. **Click a couple of movies** — try some romances. The suggestions shift towards what you clicked. The space films slide down but don't vanish: it blends your whole session, weighted towards what you did most recently.

4. **Click "start a new session"** in the small grey line under the title. Your history is wiped and the heading drops back to "Trending Now", ready to test again from scratch.

## How it works

Both the movies and your behaviour get turned into the same thing — a list of 384 numbers. Once they're the same kind of thing, "what should we recommend?" is just a question of which numbers sit closest together.

**Set up once:** each movie's genre, director, cast and plot are mashed into one block of text, an AI model ([`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), ~90 MB, runs locally) turns that into 384 numbers, and all 399 are stored in a local ChromaDB database.

**Every time you do something:** a search gets run through the same model; a click reuses the numbers already stored for that movie. They're averaged into one "interest vector", with your **last 3 actions counted double** so the list follows you if you change direction. The movies closest to that vector are your recommendations.

A few details:

- Your history is per-browser, via an anonymous session cookie. No name, no email, no account.
- A movie you just clicked is never recommended back to you — it'd trivially be the closest match to itself.
- If the search index is missing or broken, the site falls back to "Trending Now" rather than showing an error.
- Two debug URLs show the machinery: `/api/debug/events` (what was recorded) and `/api/debug/interest` (how the vector was built, and the weight given to each action).

Built with FastAPI, ChromaDB, sentence-transformers and SQLite. The frontend is plain HTML, CSS and JavaScript — no build step, no npm.

## Troubleshooting

**`chroma_db/ not found — run load_chroma.py first`**
Install step 4 wasn't done, or only partly. Run all three commands in order.

**`Address already in use`**
Something's already on port 8000. Use `--port 8001`, or free it with `kill $(lsof -ti:8000)`.

**Recommendations never change from "Trending Now"**
Open `/api/debug/interest`. If `events_used` is `0`, your searches and clicks aren't being logged — check your browser isn't blocking cookies for `localhost`.

**Want to wipe everything and rebuild?**
`rm -rf chroma_db model events.db movies_with_soup.json`, then re-run install step 4. Nothing irreplaceable is in there.

---

The movie data came from the [OMDb API](https://www.omdbapi.com/) and is included here so the project runs without a key. To build your own catalog instead, get a free key, copy `.env.example` to `.env`, add your key, then run `python fetch_movies.py --refresh` and redo install step 4.

This is a learning project, not production software — the debug endpoints expose session data, and CORS is open to any localhost port.
