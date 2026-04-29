// Pulls live data from the Lichess public API and renders it.
// No auth required — endpoints are public.
//
// Refresh policy:
//   * /api/user/<bot>          (rating, online status)  every 30s
//   * /api/games/user/<bot>    (recent games)           on load only
//
// All DOM updates degrade gracefully if the bot is offline or the
// API rate-limits us.

const BOT = "nick_p12_bot";
const API_USER = `https://lichess.org/api/user/${BOT}`;
const API_GAMES = `https://lichess.org/api/games/user/${BOT}?max=5&moves=false&pgnInJson=false`;
const REFRESH_MS = 30_000;

const fmt = new Intl.NumberFormat("en-US");

const $ = (id) => document.getElementById(id);

async function fetchJSON(url, headers = {}) {
  const r = await fetch(url, { headers: { Accept: "application/json", ...headers } });
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

async function fetchNDJSON(url) {
  const r = await fetch(url, { headers: { Accept: "application/x-ndjson" } });
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  const text = await r.text();
  return text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
}

// ──────────────────────────────────────────────────────────────
// Live status + ratings
// ──────────────────────────────────────────────────────────────

async function updateUserData() {
  try {
    const u = await fetchJSON(API_USER);
    const blitz = u?.perfs?.blitz?.rating;
    const blitzGames = u?.perfs?.blitz?.games ?? 0;
    const blitzProg = u?.perfs?.blitz?.prog;
    const rapid = u?.perfs?.rapid?.rating;
    const rapidGames = u?.perfs?.rapid?.games ?? 0;
    const totalGames = u?.count?.rated ?? u?.count?.all ?? 0;

    $("ratingBlitz").textContent = blitz ? fmt.format(blitz) : "—";
    $("ratingRapid").textContent = rapid ? fmt.format(rapid) : "—";
    $("totalGames").textContent = fmt.format(totalGames);

    if (typeof blitzProg === "number" && blitzProg !== 0) {
      const sign = blitzProg > 0 ? "+" : "";
      $("deltaBlitz").textContent = `${sign}${blitzProg} recent`;
    } else {
      $("deltaBlitz").textContent = blitzGames < 5 ? "calibrating…" : "";
    }

    const online = !!u?.online;
    $("onlineStatus").textContent = online ? "yes" : "asleep";
    const badge = $("liveStatusBadge");
    badge.dataset.state = online ? "online" : "offline";
    badge.lastChild.textContent = online ? "accepting challenges" : "asleep — wakes on first challenge";
  } catch (e) {
    console.warn("user data fetch failed", e);
    const badge = $("liveStatusBadge");
    badge.dataset.state = "offline";
    badge.lastChild.textContent = "status unavailable";
  }
}

// ──────────────────────────────────────────────────────────────
// Recent games — last 5
// ──────────────────────────────────────────────────────────────

const TIME_FMT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
});

function classifyResult(game) {
  // Determine outcome from the bot's perspective.
  const w = game?.players?.white?.user?.name?.toLowerCase();
  const b = game?.players?.black?.user?.name?.toLowerCase();
  const botIsWhite = w === BOT.toLowerCase();
  const winner = game?.winner; // "white" | "black" | undefined (draw)
  if (!winner) return { tag: "draw", label: "½–½" };
  const botWon = (winner === "white" && botIsWhite) || (winner === "black" && !botIsWhite);
  return botWon
    ? { tag: "win", label: "1–0 bot" }
    : { tag: "loss", label: "0–1 opp" };
}

function timeControlLabel(g) {
  const tc = g?.clock;
  if (!tc) {
    const speed = g?.speed;
    return speed ?? "—";
  }
  return `${Math.floor(tc.initial / 60)}+${tc.increment}`;
}

function opponentName(g) {
  const w = g?.players?.white?.user?.name;
  const b = g?.players?.black?.user?.name;
  const wr = g?.players?.white?.rating;
  const br = g?.players?.black?.rating;
  const botIsWhite = (w || "").toLowerCase() === BOT.toLowerCase();
  const oppName = botIsWhite ? b : w;
  const oppRating = botIsWhite ? br : wr;
  return { name: oppName ?? "anonymous", rating: oppRating, botColor: botIsWhite ? "white" : "black" };
}

async function updateRecentGames() {
  const list = $("recentList");
  try {
    const games = await fetchNDJSON(API_GAMES);
    if (!games.length) {
      list.innerHTML = '<li class="recent__loading">No games yet — be the first to challenge.</li>';
      return;
    }
    list.innerHTML = games
      .map((g) => {
        const { name, rating, botColor } = opponentName(g);
        const result = classifyResult(g);
        const date = g.createdAt ? TIME_FMT.format(new Date(g.createdAt)) : "";
        const tc = timeControlLabel(g);
        const url = `https://lichess.org/${g.id}`;
        return `<li>
          <span class="recent__date">${date}</span>
          <span class="recent__opp"><a href="${url}" target="_blank" rel="noopener">${escapeHTML(name)}</a><span class="opp__rating">${rating ?? ""}</span></span>
          <span class="recent__tc">${tc} ${escapeHTML(g.speed ?? "")}</span>
          <span class="recent__color">as ${botColor}</span>
          <span class="recent__result recent__result--${result.tag}">${result.label}</span>
        </li>`;
      })
      .join("");
  } catch (e) {
    console.warn("games fetch failed", e);
    list.innerHTML = '<li class="recent__loading">Couldn’t reach Lichess. Refresh in a moment.</li>';
  }
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ──────────────────────────────────────────────────────────────
// Boot
// ──────────────────────────────────────────────────────────────

updateUserData();
updateRecentGames();
setInterval(updateUserData, REFRESH_MS);

// Also refresh on visibility change (someone tabs back in)
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) updateUserData();
});
