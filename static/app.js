// Same-origin by default. Point this at http://localhost:8000 if you ever serve
// the frontend from a different port — CORS on the backend already allows it.
const API = "";

const els = {
  catalog: document.getElementById("catalog"),
  language: document.getElementById("filter-language"),
  genre: document.getElementById("filter-genre"),
  clear: document.getElementById("clear-filters"),
  filterCount: document.getElementById("filter-count"),
  search: document.getElementById("search"),
  searchSection: document.getElementById("search-section"),
  searchResults: document.getElementById("search-results"),
  searchHeading: document.getElementById("search-heading"),
  catalogSection: document.getElementById("catalog-section"),
  sessionId: document.getElementById("session-id"),
  newSession: document.getElementById("new-session"),
  detail: document.getElementById("detail"),
  detailBody: document.getElementById("detail-body"),
  detailClose: document.getElementById("detail-close"),
  recommended: document.getElementById("recommended"),
  recSection: document.getElementById("recommended-section"),
  recHeading: document.getElementById("recommended-heading"),
  recWhy: document.getElementById("recommended-why"),
  heading: document.getElementById("catalog-heading"),
  status: document.getElementById("page-status"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
};

let page = 1;
const filters = { language: "", genre: "" };
let searchTimer = null;

/** Read the session cookie the backend set on our first request. */
function sessionId() {
  const match = document.cookie.match(/(?:^|;\s*)session_id=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

/** Escape text before putting it in innerHTML — titles contain quotes and &. */
function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function cardHTML(movie) {
  const poster = movie.poster
    ? `<img src="${esc(movie.poster)}" alt="" loading="lazy">`
    : `<img alt="" loading="lazy">`;

  const year = movie.year ? `<span>${esc(movie.year)}</span>` : "";
  const genre = movie.genre ? `<span>${esc(movie.genre)}</span>` : "";
  // Makes the trending claim checkable at a glance: the ratings descend.
  const rating = movie.imdbRating
    ? `<span class="rating">\u2605 ${esc(movie.imdbRating)}</span>`
    : "";

  return `
    <article class="card" data-imdbid="${esc(movie.imdbID)}">
      ${poster}
      <div class="card-body">
        <div class="title">${esc(movie.title)}</div>
        <div class="meta">${year}${rating}${genre}</div>
      </div>
    </article>`;
}

/** Build the query string from the current page and active filters. */
function moviesURL(targetPage) {
  const params = new URLSearchParams({ page: targetPage, per_page: 20 });
  if (filters.language) params.set("language", filters.language);
  if (filters.genre) params.set("genre", filters.genre);
  return `${API}/api/movies?${params}`;
}

async function loadMovies(targetPage) {
  els.status.textContent = "Loading…";
  els.prev.disabled = els.next.disabled = true;

  try {
    const res = await fetch(moviesURL(targetPage));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    page = data.page;
    const active = filters.language || filters.genre;
    els.catalog.innerHTML = data.results.length
      ? data.results.map(cardHTML).join("")
      : `<div class="empty">No movies match these filters.</div>`;
    els.heading.textContent = active ? "Movies" : `All movies (${data.total})`;
    els.filterCount.textContent = active ? `${data.total} match` : "";
    els.clear.hidden = !active;
    els.status.textContent = `Page ${data.page} of ${data.total_pages}`;
    els.prev.disabled = data.page <= 1;
    els.next.disabled = !data.has_next;
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    els.catalog.innerHTML =
      `<div class="error">Could not load movies: ${esc(err.message)}.
       Is the backend running on port 8000?</div>`;
    els.status.textContent = "";
  }
}

// Every search and view refreshes the list, and those can land out of order.
// Stamping each request lets a slow early reply be discarded instead of
// overwriting a fresher one.
let recsRequest = 0;

/**
 * Pull this session's recommendations and repaint the whole section.
 *
 * The heading text comes from the API rather than being hardcoded here: the
 * backend decides whether the list is trending or personalized, so it is also
 * the thing that knows what to call it. The frontend never guesses the label.
 */
async function loadRecommendations() {
  const stamp = ++recsRequest;
  els.recommended.classList.add("refreshing");

  try {
    const res = await fetch(`${API}/api/recommendations`, {
      credentials: "same-origin",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (stamp !== recsRequest) return; // a newer request already answered

    const movies = data.movies || [];

    // data-mode lets CSS distinguish the two states without the JS knowing
    // anything about how they look.
    els.recSection.dataset.mode = data.mode || "";
    if (data.title) els.recHeading.textContent = data.title;

    els.recWhy.textContent = data.subtitle || "";
    els.recWhy.hidden = !data.subtitle;

    if (!movies.length) {
      // Should not happen — trending is the floor — but never leave a heading
      // standing over nothing.
      els.recommended.className = "";
      els.recommended.innerHTML =
        `<div class="placeholder">Nothing to show yet.</div>`;
      return;
    }
    els.recommended.className = "grid";
    els.recommended.innerHTML = movies.map(cardHTML).join("");
  } catch {
    // Silent by design: recommendations are an enhancement, and shouting about
    // them would bury a working catalog under an error banner.
  } finally {
    if (stamp === recsRequest) els.recommended.classList.remove("refreshing");
  }
}

/** Fire-and-forget click log. Never blocks or breaks the UI. */
function logView(imdbID) {
  if (!imdbID) return;
  const body = JSON.stringify({
    type: "view",
    imdbID,
    session_id: sessionId(),
  });
  // keepalive lets the request survive if the click navigates away later.
  // Recommendations are refreshed only after the POST resolves — asking any
  // earlier would rebuild the interest vector without this click in it.
  fetch(`${API}/api/event`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body,
    keepalive: true,
  })
    .then(() => loadRecommendations())
    .catch(() => {});
}

function detailHTML(m) {
  const row = (label, value) =>
    value ? `<dt>${label}</dt><dd>${esc(value)}</dd>` : "";

  const poster = m.poster
    ? `<img src="${esc(m.poster)}" alt="Poster for ${esc(m.title)}">`
    : "";

  const sub = [m.year, m.imdbRating ? `IMDb ${m.imdbRating}` : null]
    .filter(Boolean).join(" · ");

  return `
    <div class="detail-grid">
      ${poster}
      <div class="detail-info">
        <h3 id="detail-title">${esc(m.title)}</h3>
        <div class="detail-sub">${esc(sub)}</div>
        <dl>
          ${row("Genre", m.genre)}
          ${row("Director", m.director)}
          ${row("Actors", m.actors)}
          ${row("Language", m.language)}
        </dl>
        ${m.plot ? `<div class="detail-plot">
          <h4>Plot</h4><p>${esc(m.plot)}</p></div>` : ""}
      </div>
    </div>`;
}

async function openDetail(imdbID) {
  if (!imdbID) return;
  els.detail.hidden = false;
  els.detailBody.innerHTML = `<div class="detail-loading">Loading…</div>`;
  document.body.style.overflow = "hidden"; // stop the page scrolling behind
  setParam("movie", imdbID);

  try {
    const res = await fetch(`${API}/api/movies/${encodeURIComponent(imdbID)}`, {
      credentials: "same-origin",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    els.detailBody.innerHTML = detailHTML(await res.json());
    els.detailClose.focus();
  } catch (err) {
    els.detailBody.innerHTML =
      `<div class="error">Could not load this movie: ${esc(err.message)}</div>`;
  }
}

function closeDetail() {
  els.detail.hidden = true;
  els.detailBody.innerHTML = "";
  document.body.style.overflow = "";
  setParam("movie", null);
}

els.detailClose.addEventListener("click", closeDetail);
els.detail.addEventListener("click", (ev) => {
  if (ev.target === els.detail) closeDetail(); // backdrop click only
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !els.detail.hidden) closeDetail();
});

// One listener on the page catches clicks in the grid AND in search results,
// including cards rendered after this ran.
document.addEventListener("click", (ev) => {
  const card = ev.target.closest(".card");
  if (!card) return;
  const { imdbid } = card.dataset;
  logView(imdbid);   // log first, exactly as before
  openDetail(imdbid); // then show the detail view
});

/**
 * Wipe this session's history and start a fresh one.
 *
 * Testing the personalization needs a clean slate — once a session has looked
 * at a few movies, every later result is coloured by that history and there is
 * no way back to the cold start from inside the page.
 */
async function startNewSession() {
  const label = els.newSession.textContent;
  els.newSession.disabled = true;
  els.newSession.textContent = "starting\u2026";

  try {
    const res = await fetch(`${API}/api/session/reset`, {
      method: "POST",
      credentials: "same-origin",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Clear the search UI too. Leaving results on screen would show them under
    // a session that never ran the search.
    clearTimeout(searchTimer);
    els.search.value = "";
    els.searchSection.hidden = true;
    els.catalogSection.hidden = false;
    els.searchResults.innerHTML = "";
    setParam("movie", null);

    // Read the cookie rather than the response body, so the label reflects what
    // the browser will actually send from here on.
    els.sessionId.textContent = (sessionId() || data.session_id).slice(0, 12);

    // Back to the cold start: this should relabel itself "Trending Now".
    await loadRecommendations();
    loadMovies(1);
  } catch (err) {
    els.newSession.textContent = "reset failed";
    setTimeout(() => { els.newSession.textContent = label; }, 1600);
    return;
  } finally {
    els.newSession.disabled = false;
  }
  els.newSession.textContent = label;
}

els.newSession.addEventListener("click", startNewSession);

async function runSearch(query) {
  const q = query.trim();

  if (!q) {
    // Empty box: hide results, show the full catalog again.
    els.searchSection.hidden = true;
    els.catalogSection.hidden = false;
    return;
  }

  els.searchSection.hidden = false;
  els.catalogSection.hidden = true;
  els.searchHeading.textContent = `Searching for "${q}"…`;

  try {
    const res = await fetch(
      `${API}/api/search?q=${encodeURIComponent(q)}`,
      { credentials: "same-origin" }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    els.searchHeading.textContent =
      `${data.total} result${data.total === 1 ? "" : "s"} for "${q}"`;
    els.searchResults.innerHTML = data.results.length
      ? data.results.map(cardHTML).join("")
      : `<div class="empty">Nothing matched "${esc(q)}".</div>`;

    // The search endpoint logged this query, so the interest vector has moved.
    // Note this fires even when nothing matched: "space adventure" is a useful
    // signal about intent whether or not those words appear in any plot.
    loadRecommendations();
  } catch (err) {
    els.searchResults.innerHTML =
      `<div class="error">Search failed: ${esc(err.message)}</div>`;
  }
}

// Debounced so a search event is logged per pause, not per keystroke.
els.search.addEventListener("input", (ev) => {
  clearTimeout(searchTimer);
  const value = ev.target.value;
  searchTimer = setTimeout(() => runSearch(value), 400);
});

els.search.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    clearTimeout(searchTimer);
    runSearch(ev.target.value);
  }
});

/** Populate the two dropdowns from /api/facets. */
async function loadFilters() {
  try {
    const res = await fetch(`${API}/api/facets`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { languages, genres } = await res.json();

    const fill = (select, items) => {
      select.insertAdjacentHTML("beforeend", items.map((it) =>
        `<option value="${esc(it.value)}">${esc(it.value)} (${it.count})</option>`
      ).join(""));
    };
    fill(els.language, languages);
    fill(els.genre, genres);

    // Restore filters from the URL (?language=Hindi&genre=Action).
    const params = new URLSearchParams(location.search);
    for (const [key, select] of [["language", els.language], ["genre", els.genre]]) {
      const wanted = (params.get(key) || "").toLowerCase();
      if (!wanted) continue;
      const match = [...select.options].find(
        (o) => o.value.toLowerCase() === wanted
      );
      if (match) {
        select.value = match.value;
        filters[key] = match.value;
      }
    }
  } catch {
    // Leave the "All …" defaults in place; the grid still works unfiltered.
  }
}

/** Set or clear one query param without disturbing the others. */
function setParam(key, value) {
  const params = new URLSearchParams(location.search);
  if (value) params.set(key, value);
  else params.delete(key);
  const qs = params.toString();
  history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
}

/** Mirror the active filters into the address bar so views are shareable. */
function syncURL() {
  setParam("language", filters.language);
  setParam("genre", filters.genre);
}

function onFilterChange() {
  filters.language = els.language.value;
  filters.genre = els.genre.value;
  syncURL();
  loadMovies(1); // filtered result sets are shorter, so always restart at page 1
}

els.language.addEventListener("change", onFilterChange);
els.genre.addEventListener("change", onFilterChange);
els.clear.addEventListener("click", () => {
  els.language.value = "";
  els.genre.value = "";
  onFilterChange();
});

els.prev.addEventListener("click", () => loadMovies(page - 1));
els.next.addEventListener("click", () => loadMovies(page + 1));

// Filters must be read from the URL before the first grid load.
loadFilters().then(() => loadMovies(1));
loadRecommendations();

// Support ?movie=tt1234567 so a detail view can be linked to directly.
const initialMovie = new URLSearchParams(location.search).get("movie");
if (initialMovie) openDetail(initialMovie);

// The cookie arrives with the first response, so read it after that resolves.
fetch(`${API}/api/health`, { credentials: "same-origin" })
  .then(() => {
    els.sessionId.textContent = (sessionId() || "not set").slice(0, 12);
  })
  .catch(() => {});
