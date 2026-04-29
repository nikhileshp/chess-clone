// Pulls live data from the Lichess public API and renders it.
// No auth required — endpoints are public.
//
//   * /api/user/<bot>                    rating + online status (every 30s)
//   * /api/user/<bot>/current-game       most recent or ongoing game (every 15s)
//   * /api/games/user/<bot>?max=5        recent games table (on load)
//
// Embed format: lichess.org/embed/<gameId>?theme=brown&bg=light
// This is a spectator iframe — visitors WATCH live, then click the
// "Challenge me" CTA to actually play (Lichess blocks iframing the
// playable interface).

const BOT = "nick_p12_bot";
const API_USER = `https://lichess.org/api/user/${BOT}`;
const API_CURRENT = `https://lichess.org/api/user/${BOT}/current-game?moves=false&clocks=false&evals=false`;
const API_GAMES = `https://lichess.org/api/games/user/${BOT}?max=5&moves=false&pgnInJson=false`;

const EMBED_THEME = "brown";
const EMBED_BG = "light";
const REFRESH_USER_MS = 30_000;
const REFRESH_GAME_MS = 15_000;

const fmt = new Intl.NumberFormat("en-US");
const $ = (id) => document.getElementById(id);

async function fetchJSON(url) {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}
async function fetchNDJSON(url) {
  const r = await fetch(url, { headers: { Accept: "application/x-ndjson" } });
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return (await r.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l));
}
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const TIME_FMT = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });

// ──────────────────────────────────────────────────────────────
// User: ratings + online state
// ──────────────────────────────────────────────────────────────

async function updateUserData() {
  try {
    const u = await fetchJSON(API_USER);
    const blitz = u?.perfs?.blitz?.rating;
    const blitzGames = u?.perfs?.blitz?.games ?? 0;
    const blitzProg = u?.perfs?.blitz?.prog;
    const rapid = u?.perfs?.rapid?.rating;
    const total = u?.count?.rated ?? u?.count?.all ?? 0;

    $("ratingBlitz").textContent = blitz ? fmt.format(blitz) : "—";
    $("ratingRapid").textContent = rapid ? fmt.format(rapid) : "—";
    $("totalGames").textContent = fmt.format(total);

    if (typeof blitzProg === "number" && blitzProg !== 0) {
      const sign = blitzProg > 0 ? "+" : "";
      $("deltaBlitz").textContent = `${sign}${blitzProg} recent`;
    } else if (blitzGames < 5) {
      $("deltaBlitz").textContent = "calibrating…";
    } else {
      $("deltaBlitz").textContent = "";
    }

    const badge = $("liveStatusBadge");
    badge.dataset.state = u?.online ? "online" : "offline";
    badge.querySelector(".lbl").textContent = u?.online ? "accepting challenges" : "asleep — wakes on first challenge";
  } catch (e) {
    console.warn("user fetch failed", e);
    const badge = $("liveStatusBadge");
    badge.dataset.state = "offline";
    badge.querySelector(".lbl").textContent = "status unavailable";
  }
}

// ──────────────────────────────────────────────────────────────
// Current game embed (the centerpiece)
// ──────────────────────────────────────────────────────────────

let currentEmbedGameId = null;

function buildEmbedUrl(gameId, color) {
  const orientation = color || "white";
  return `https://lichess.org/embed/game/${encodeURIComponent(gameId)}?theme=${EMBED_THEME}&bg=${EMBED_BG}&orientation=${orientation}`;
}

function showPlaceholder() {
  $("boardPlaceholder").style.display = "";
  // Remove any existing iframe
  const wrap = $("boardWrap");
  const old = wrap.querySelector("iframe");
  if (old) old.remove();
  currentEmbedGameId = null;
  $("captionEyebrow").textContent = "Status";
  $("captionTitle").textContent = "Awaiting first game";
}

function mountIframe(gameId, opts) {
  const wrap = $("boardWrap");
  const url = buildEmbedUrl(gameId, opts?.orientation);
  let iframe = wrap.querySelector("iframe");
  if (!iframe) {
    iframe = document.createElement("iframe");
    iframe.title = "Live Lichess game";
    iframe.allow = "fullscreen";
    iframe.referrerPolicy = "no-referrer-when-downgrade";
    iframe.loading = "lazy";
    wrap.appendChild(iframe);
  }
  if (iframe.dataset.gid !== gameId) {
    iframe.src = url;
    iframe.dataset.gid = gameId;
  }
  $("boardPlaceholder").style.display = "none";
}

async function updateCurrentGame() {
  // Try /current-game first; if 404 (no games yet) or errors, fall back to recent games list.
  try {
    const r = await fetch(API_CURRENT, { headers: { Accept: "application/json" } });
    if (r.ok) {
      const game = await r.json();
      handleCurrentGame(game, /*ongoing*/ true);
      return;
    }
    if (r.status !== 404) {
      // Unexpected — log but keep what we have.
      console.warn("current-game returned HTTP", r.status);
    }
  } catch (e) {
    console.warn("current-game fetch failed", e);
  }

  // Fallback: latest finished game
  try {
    const games = await fetchNDJSON(`https://lichess.org/api/games/user/${BOT}?max=1&moves=false&pgnInJson=false`);
    if (games.length) handleCurrentGame(games[0], /*ongoing*/ false);
    else showPlaceholder();
  } catch (e) {
    console.warn("recent-game fetch failed", e);
  }
}

function handleCurrentGame(g, ongoing) {
  if (!g?.id) {
    showPlaceholder();
    return;
  }
  // Orient board so bot's pieces are on the bottom (most natural to spectate).
  const wName = g?.players?.white?.user?.name?.toLowerCase();
  const orientation = wName === BOT.toLowerCase() ? "white" : "black";

  if (g.id !== currentEmbedGameId) {
    mountIframe(g.id, { orientation });
    currentEmbedGameId = g.id;
  }

  const oppInfo = opponentInfo(g);
  const tc = timeControlLabel(g);
  $("captionEyebrow").textContent = ongoing ? "● Live now" : "Most recent game";
  const title = `vs ${escapeHTML(oppInfo.name)} (${oppInfo.rating ?? "?"}) · ${tc} ${escapeHTML(g.speed ?? "")}`;
  $("captionTitle").innerHTML = title;
}

// ──────────────────────────────────────────────────────────────
// Recent games list
// ──────────────────────────────────────────────────────────────

function classifyResult(g) {
  const wName = g?.players?.white?.user?.name?.toLowerCase();
  const botIsWhite = wName === BOT.toLowerCase();
  const winner = g?.winner;
  if (!winner) return { tag: "draw", label: "½–½" };
  const botWon = (winner === "white" && botIsWhite) || (winner === "black" && !botIsWhite);
  return botWon ? { tag: "win", label: "1–0 bot" } : { tag: "loss", label: "0–1 opp" };
}

function timeControlLabel(g) {
  const tc = g?.clock;
  if (!tc) return g?.speed ?? "—";
  return `${Math.floor(tc.initial / 60)}+${tc.increment}`;
}

function opponentInfo(g) {
  const w = g?.players?.white?.user?.name;
  const b = g?.players?.black?.user?.name;
  const wr = g?.players?.white?.rating;
  const br = g?.players?.black?.rating;
  const botIsWhite = (w || "").toLowerCase() === BOT.toLowerCase();
  return {
    name: botIsWhite ? (b ?? "anonymous") : (w ?? "anonymous"),
    rating: botIsWhite ? br : wr,
    botColor: botIsWhite ? "white" : "black",
  };
}

async function updateRecentGames() {
  const list = $("recentList");
  try {
    const games = await fetchNDJSON(API_GAMES);
    if (!games.length) {
      list.innerHTML = '<li class="recent-loading">No games yet — be the first to challenge.</li>';
      return;
    }
    list.innerHTML = games.map((g) => {
      const opp = opponentInfo(g);
      const result = classifyResult(g);
      const date = g.createdAt ? TIME_FMT.format(new Date(g.createdAt)) : "";
      const tc = timeControlLabel(g);
      const url = `https://lichess.org/${g.id}`;
      return `<li>
        <span class="recent-date">${date}</span>
        <span class="recent-opp"><a href="${url}" target="_blank" rel="noopener">${escapeHTML(opp.name)}</a><span class="opp-rating">${opp.rating ?? ""}</span></span>
        <span class="recent-tc">${tc} ${escapeHTML(g.speed ?? "")}</span>
        <span class="recent-color">as ${opp.botColor}</span>
        <span class="recent-result recent-result--${result.tag}">${result.label}</span>
      </li>`;
    }).join("");
  } catch (e) {
    console.warn("games fetch failed", e);
    list.innerHTML = '<li class="recent-loading">Couldn’t reach Lichess. Refresh in a moment.</li>';
  }
}

// ──────────────────────────────────────────────────────────────
// Boot
// ──────────────────────────────────────────────────────────────

updateUserData();
updateCurrentGame();
updateRecentGames();
setInterval(updateUserData, REFRESH_USER_MS);
setInterval(updateCurrentGame, REFRESH_GAME_MS);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    updateUserData();
    updateCurrentGame();
  }
});
