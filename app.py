"""Web Phase 3: catalog, session tracking, and live interest-vector recommendations.

Endpoints
    GET  /api/movies           paginated catalog (language/genre filters)
    GET  /api/movies/{imdbID}  full details for one movie
    GET  /api/facets           dropdown options
    GET  /api/search           substring search; logs a "search" event
    POST /api/event            logs a "view" event when a card is clicked
    POST /api/session/reset    clear this session and start a fresh one
    GET  /api/debug/events     this session's event history (testing aid)
    GET  /api/recommendations  {mode, title, subtitle, movies} - trending or
                               personalized, and it says which
    GET  /api/debug/interest   how that vector was built (testing aid)
    GET  /                     the single-page frontend

Run:
    ./venv/bin/uvicorn app:app --reload --port 8000
    open http://localhost:8000
"""

import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import interest
import tracking

HERE = Path(__file__).parent
MOVIES_FILE = HERE / "movies.json"
STATIC_DIR = HERE / "static"

DEFAULT_PER_PAGE = 20
MAX_PER_PAGE = 100

SESSION_COOKIE = "session_id"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

app = FastAPI(title="Movie Recommender", version="0.1.0")

# Browsers refuse to send cookies to a wildcard origin, so once sessions exist
# the allowed origins have to be named. This regex covers any localhost port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tracking.init_db()


@app.middleware("http")
async def ensure_session(request: Request, call_next):
    """Give every visitor a session_id cookie on their first request."""
    session_id = request.cookies.get(SESSION_COOKIE)
    is_new = not session_id
    if is_new:
        session_id = tracking.new_session_id()

    # Stash it so endpoints can read it without re-parsing cookies.
    request.state.session_id = session_id

    response = await call_next(request)
    if is_new:
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            max_age=SESSION_MAX_AGE,
            httponly=False,  # the frontend reads it to send in POST bodies
            samesite="lax",
        )
    return response


def session_of(request: Request) -> str:
    """The current visitor's session id, set by the middleware above."""
    return request.state.session_id


def na(value: object) -> str | None:
    """OMDb writes the string "N/A" for missing values; the API should send null."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if not value or value.upper() == "N/A" else value


def load_movies() -> list[dict]:
    """Read the catalog once at import and reshape it for the frontend."""
    if not MOVIES_FILE.exists():
        raise RuntimeError(f"{MOVIES_FILE.name} not found — run fetch_movies.py first")

    with MOVIES_FILE.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    return [
        {
            "imdbID": na(m.get("imdbID")),
            "title": na(m.get("Title")) or "Untitled",
            "year": na(m.get("Year")),
            "genre": na(m.get("Genre")),
            "director": na(m.get("Director")),
            "actors": na(m.get("Actors")),
            "plot": na(m.get("Plot")),
            "language": na(m.get("Language")),
            "poster": na(m.get("Poster")),
            "imdbRating": na(m.get("imdbRating")),
        }
        for m in raw
    ]


MOVIES = load_movies()

# imdbID -> movie, so the detail endpoint is a dict hit rather than a scan.
MOVIES_BY_ID = {m["imdbID"]: m for m in MOVIES if m.get("imdbID")}


def rating_of(movie: dict) -> float | None:
    """imdbRating arrives from OMDb as a string, and is None for unrated titles."""
    try:
        return float(movie["imdbRating"])
    except (KeyError, TypeError, ValueError):
        return None


def rank_by_rating() -> list[dict]:
    """The whole catalog sorted by rating, best first.

    Unrated titles are dropped rather than treated as 0.0 — "no rating" is
    missing data, not a bad movie. Ties break on title so the order is stable
    across restarts instead of depending on the catalog's file order.
    """
    rated = [(r, m) for m in MOVIES if (r := rating_of(m)) is not None]
    rated.sort(key=lambda pair: (-pair[0], pair[1]["title"].casefold()))
    return [m for _, m in rated]


# Trending is identical for every visitor and only changes when movies.json
# does, so the sort happens once here at import. get_trending_movies() slices
# this list — no per-request work, whatever n it is asked for.
TRENDING = rank_by_rating()

TRENDING_TITLE = "Trending Now"
PERSONALIZED_TITLE = "Recommended for you"


def get_trending_movies(n: int = 10) -> list[dict]:
    """Top n movies by imdbRating, descending. Fixed and non-personalized."""
    return TRENDING[:n]


def tokens_of(movie: dict, field: str) -> set[str]:
    """OMDb packs multi-values into one comma-separated string, e.g.
    "Hindi, Telugu, Tamil" or "Comedy, Drama". Split so any one can match."""
    raw = movie.get(field) or ""
    return {part.strip().casefold() for part in raw.split(",") if part.strip()}


def facet_values(field: str) -> list[dict]:
    """Every distinct value for a field, with counts, most common first."""
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for movie in MOVIES:
        raw = movie.get(field) or ""
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            key = part.casefold()
            counts[key] = counts.get(key, 0) + 1
            labels.setdefault(key, part)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], labels[kv[0]]))
    return [{"value": labels[k], "count": n} for k, n in ordered]


@app.get("/api/facets")
def get_facets() -> dict:
    """Dropdown options for the frontend filters."""
    return {"languages": facet_values("language"), "genres": facet_values("genre")}


@app.get("/api/movies")
def get_movies(
    page: int = Query(1, ge=1, description="1-based page number"),
    per_page: int = Query(DEFAULT_PER_PAGE, ge=1, le=MAX_PER_PAGE),
    language: str | None = Query(
        None, description='filter to movies listing this language, e.g. "Hindi"'
    ),
    genre: str | None = Query(
        None, description='filter to movies listing this genre, e.g. "Comedy"'
    ),
) -> dict:
    """Return one page of the catalog, optionally filtered.

    Filters combine with AND: language=Hindi&genre=Action means both.
    """
    movies = MOVIES
    if language:
        wanted = language.strip().casefold()
        movies = [m for m in movies if wanted in tokens_of(m, "language")]
    if genre:
        wanted = genre.strip().casefold()
        movies = [m for m in movies if wanted in tokens_of(m, "genre")]

    total = len(movies)
    total_pages = max(1, (total + per_page - 1) // per_page)

    if page > total_pages:
        raise HTTPException(404, f"page {page} is past the last page ({total_pages})")

    start = (page - 1) * per_page
    return {
        "language": language,
        "genre": genre,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "results": movies[start : start + per_page],
    }


@app.get("/api/movies/{imdb_id}")
def get_movie(imdb_id: str) -> dict:
    """Full details for one movie."""
    movie = MOVIES_BY_ID.get(imdb_id)
    if movie is None:
        raise HTTPException(404, f"no movie with imdbID {imdb_id!r}")
    return movie


def recommendation_subtitle(session_id: str) -> str:
    """A short "why am I seeing this" line, built from the session's own events.

    Names the most recent search because that is the strongest, most legible
    signal in the profile — the visitor typed it, so seeing it quoted back
    makes the list feel caused rather than magic.
    """
    events = tracking.events_for(session_id, limit=interest.HISTORY_LIMIT)

    # events is newest-first, so the first search hit is the latest one.
    latest_search = next(
        (
            e["query"].strip()
            for e in events
            if e["type"] == "search" and (e.get("query") or "").strip()
        ),
        None,
    )
    views = sum(1 for e in events if e["type"] == "view" and e.get("imdb_id"))

    parts = []
    if latest_search:
        parts.append(f'your search for \u201c{latest_search}\u201d')
    if views:
        parts.append(f"{views} movie{'' if views == 1 else 's'} you viewed")

    if not parts:
        return "Based on your recent activity"
    return "Based on " + " and ".join(parts)


def trending_payload(n: int) -> dict:
    """The non-personalized list, labelled honestly as such."""
    return {
        "mode": "trending",
        "title": TRENDING_TITLE,
        "subtitle": "Top rated across the catalog",
        "movies": get_trending_movies(n),
    }


@app.get("/api/recommendations")
def get_recommendations(
    request: Request,
    n: int = Query(12, ge=1, le=50),
    include_seen: bool = Query(False, description="keep movies already viewed"),
) -> dict:
    """This session's recommendations, plus the label that describes them.

    Two modes, and the response says which one the caller got:

        trending      no usable history yet — top rated, same for everyone
        personalized  nearest neighbours of the session's interest vector

    There is deliberately no empty state. A visitor on their first page load
    still gets a populated list; the title is what changes, so the label never
    over-promises personalization that has not happened yet.
    """
    session_id = session_of(request)

    # Chroma is only needed for the personalized branch. If the index is
    # missing or unreadable, trending still works, so degrade to it rather
    # than failing the request and blanking the section.
    vector = None
    if interest.is_ready():
        try:
            vector = interest.build_interest_vector(session_id)
        except Exception:
            vector = None  # /api/debug/interest surfaces the real error

    if vector is None:
        return trending_payload(n)

    # A movie the visitor just clicked is by definition the nearest thing to
    # its own vector, so handing it straight back is noise, not a recommendation.
    exclude: set[str] = set()
    if not include_seen:
        exclude = {
            e["imdb_id"]
            for e in tracking.events_for(session_id, limit=interest.HISTORY_LIMIT)
            if e["type"] == "view" and e.get("imdb_id")
        }

    try:
        hits = interest.recommend_for_session(
            session_id, n=n, exclude=exclude, vector=vector
        )
    except Exception:
        return trending_payload(n)

    if not hits:
        # A profile exists, but every neighbour was filtered out. Still better
        # to show trending than an empty grid under a personalized heading.
        return trending_payload(n)

    # Chroma only stores enough metadata to identify a match, so hydrate from
    # the catalog to give the cards posters and years like everywhere else.
    movies = []
    for hit in hits:
        movie = MOVIES_BY_ID.get(hit["imdbID"])
        movies.append({**movie, "score": hit["score"]} if movie else hit)

    return {
        "mode": "personalized",
        "title": PERSONALIZED_TITLE,
        "subtitle": recommendation_subtitle(session_id),
        "movies": movies,
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "movies": len(MOVIES)}


# --------------------------------------------------------------------------
# Search + event tracking
# --------------------------------------------------------------------------

SEARCH_FIELDS = ("title", "genre", "plot", "director", "actors")


class EventIn(BaseModel):
    """Body for POST /api/event."""

    type: str = Field(..., description='"view"')
    imdbID: str | None = Field(None, description="the movie that was clicked")
    # Accepted for compatibility, but the cookie wins — a client should not be
    # able to write events into someone else's session just by naming it.
    session_id: str | None = None


@app.get("/api/search")
def search(
    request: Request,
    q: str = Query(..., min_length=1, description="substring to look for"),
    limit: int = Query(40, ge=1, le=MAX_PER_PAGE),
) -> dict:
    """Substring search across title/genre/plot, logged against the session."""
    session_id = session_of(request)
    needle = q.strip().casefold()

    results = [
        m for m in MOVIES
        if any(needle in (m.get(f) or "").casefold() for f in SEARCH_FIELDS)
    ]

    # Log the query itself, not the results — Phase 3 cares about intent.
    tracking.log_event(session_id, "search", query=q.strip())

    return {
        "query": q.strip(),
        "session_id": session_id,
        "total": len(results),
        "results": results[:limit],
    }


@app.post("/api/event")
def post_event(request: Request, event: EventIn = Body(...)) -> dict:
    """Record a click on a movie card."""
    session_id = session_of(request)

    if event.type not in tracking.EVENT_TYPES:
        raise HTTPException(422, f"unknown event type: {event.type!r}")
    if event.type == "view" and not event.imdbID:
        raise HTTPException(422, "a view event needs an imdbID")

    event_id = tracking.log_event(
        session_id, event.type, imdb_id=event.imdbID
    )
    return {"ok": True, "event_id": event_id, "session_id": session_id}


@app.post("/api/session/reset")
def reset_session(
    request: Request,
    response: Response,
    purge: bool = Query(True, description="also delete the old session's events"),
) -> dict:
    """Start a clean session, so one test isn't coloured by the last one.

    Rotates the cookie to a brand-new session_id. Everything personalized keys
    off that id, so the recommendations section drops straight back to its
    trending cold start with nothing carried over.

    purge=true (the default) also deletes the old session's rows. Pass
    purge=false to rotate but keep the old history readable in
    /api/debug/events — useful for comparing two runs side by side.
    """
    previous_id = session_of(request)
    deleted = tracking.forget_session(previous_id) if purge else 0

    session_id = tracking.new_session_id()

    # Set the cookie here rather than leaving it to the middleware: that only
    # issues one when the request arrived without it, and this request by
    # definition arrived with one.
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=SESSION_MAX_AGE,
        httponly=False,
        samesite="lax",
    )
    # Keep request state in step, so anything later in this cycle sees the new id.
    request.state.session_id = session_id

    return {
        "ok": True,
        "session_id": session_id,
        "previous_session_id": previous_id,
        "purged": purge,
        "deleted_events": deleted,
    }


@app.get("/api/debug/events")
def debug_events(request: Request, limit: int = Query(200, ge=1, le=1000)) -> dict:
    """Everything logged for the current session. Testing aid, not for production."""
    session_id = session_of(request)
    events = tracking.events_for(session_id, limit=limit)

    # Attach titles so view events are readable without cross-referencing ids.
    by_id = {m["imdbID"]: m["title"] for m in MOVIES if m.get("imdbID")}
    for e in events:
        if e.get("imdb_id"):
            e["title"] = by_id.get(e["imdb_id"])

    return {
        "session_id": session_id,
        "count": len(events),
        **tracking.summarise(session_id),
        "events": events,
    }


@app.get("/api/debug/interest")
def debug_interest(request: Request) -> dict:
    """What the session's interest vector is built from, and how it is weighted."""
    session_id = session_of(request)
    detail = interest.explain(session_id)

    vector = interest.build_interest_vector(session_id) if interest.is_ready() else None
    detail["chroma_ready"] = interest.is_ready()
    detail["has_vector"] = vector is not None
    detail["dims"] = int(vector.shape[0]) if vector is not None else 0
    detail["preview"] = (
        [round(float(x), 4) for x in vector[:8]] if vector is not None else []
    )
    return detail


# Mounted last so the API routes above take precedence over the static files.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
