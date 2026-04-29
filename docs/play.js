/* play.js — in-browser game vs nick_p12_bot.
 *
 * Local move legality runs in the browser via chess.js. The bot's reply is
 * fetched from /predict on the HF Space. First request is slow (~10s) while
 * the model loads; subsequent calls are ~1-2s.
 */
(function () {
  'use strict';

  // Public HF Space URL. Override at runtime by setting window.NP12_INFERENCE_URL
  // before play.js loads (e.g. for local backend testing).
  var INFERENCE_URL =
    (typeof window !== 'undefined' && window.NP12_INFERENCE_URL) ||
    'https://nikhileshp12-nick-p12-bot.hf.space/predict';

  var $statusText = document.getElementById('playStatusText');
  var $statusWrap = document.getElementById('playStatus');
  var $spinner = document.getElementById('playSpinner');
  var $moveList = document.getElementById('moveList');
  var $btnNew = document.getElementById('btnNewGame');
  var $btnUndo = document.getElementById('btnUndo');
  var $btnFlip = document.getElementById('btnFlip');
  var $colorSel = document.getElementById('playColor');
  var $eloIn = document.getElementById('playElo');

  var game = new Chess();
  var board = null;
  var humanColor = 'white';
  var botBusy = false;

  function setStatus(text, kind) {
    $statusText.textContent = text;
    $statusWrap.classList.remove('play-status--error', 'play-status--book');
    if (kind) $statusWrap.classList.add('play-status--' + kind);
  }

  function setSpinner(on) {
    $spinner.classList.toggle('d-none', !on);
  }

  function renderMoves() {
    var hist = game.history();
    if (!hist.length) {
      $moveList.innerHTML = '<em class="text-muted small">No moves yet.</em>';
      return;
    }
    var html = '';
    for (var i = 0; i < hist.length; i += 2) {
      var num = i / 2 + 1;
      var w = hist[i];
      var b = hist[i + 1] || '';
      var lastIdx = hist.length - 1;
      var wCls = i === lastIdx ? ' move-pair__san--last' : '';
      var bCls = i + 1 === lastIdx ? ' move-pair__san--last' : '';
      html += '<span class="move-pair">';
      html += '<span class="move-pair__num">' + num + '.</span>';
      html += '<span class="move-pair__san' + wCls + '">' + w + '</span>';
      if (b) html += ' <span class="move-pair__san' + bCls + '">' + b + '</span>';
      html += '</span>';
    }
    $moveList.innerHTML = html;
    $moveList.scrollTop = $moveList.scrollHeight;
  }

  function gameOverText() {
    if (game.in_checkmate()) {
      var winner = game.turn() === 'w' ? 'Black' : 'White';
      return winner + ' wins by checkmate.';
    }
    if (game.in_stalemate()) return 'Draw by stalemate.';
    if (game.in_threefold_repetition()) return 'Draw by threefold repetition.';
    if (game.insufficient_material()) return 'Draw by insufficient material.';
    if (game.in_draw()) return 'Draw (50-move rule).';
    return null;
  }

  function checkGameOver() {
    var msg = gameOverText();
    if (msg) {
      setStatus(msg);
      $btnUndo.disabled = true;
      return true;
    }
    return false;
  }

  // Highlight last move on the board
  function highlightLast(move) {
    $('#playBoard .square-55d63').css('box-shadow', '');
    if (!move) return;
    $('#playBoard .square-' + move.from).css('box-shadow', 'inset 0 0 0 3px rgba(255, 193, 7, 0.7)');
    $('#playBoard .square-' + move.to).css('box-shadow', 'inset 0 0 0 3px rgba(255, 193, 7, 0.9)');
  }

  function onDragStart(_source, piece) {
    if (game.game_over()) return false;
    if (botBusy) return false;
    var humanIsWhite = humanColor === 'white';
    if (humanIsWhite && piece.search(/^b/) !== -1) return false;
    if (!humanIsWhite && piece.search(/^w/) !== -1) return false;
    var turnIsWhite = game.turn() === 'w';
    if (humanIsWhite !== turnIsWhite) return false;
    return true;
  }

  function onDrop(source, target) {
    var move = game.move({ from: source, to: target, promotion: 'q' });
    if (move === null) return 'snapback';
    highlightLast(move);
    renderMoves();
    $btnUndo.disabled = false;
    if (checkGameOver()) return;
    setStatus("Bot is thinking…");
    setSpinner(true);
    botBusy = true;
    requestBotMove();
  }

  function onSnapEnd() {
    board.position(game.fen());
  }

  function requestBotMove() {
    var elo = parseInt($eloIn.value, 10);
    if (isNaN(elo)) elo = 1600;
    fetch(INFERENCE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen: game.fen(), opponent_elo: elo }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (t) {
            throw new Error('HTTP ' + res.status + ': ' + (t || res.statusText));
          });
        }
        return res.json();
      })
      .then(function (data) {
        var mv = game.move({
          from: data.move.slice(0, 2),
          to: data.move.slice(2, 4),
          promotion: data.move.length > 4 ? data.move[4] : 'q',
        });
        if (!mv) {
          throw new Error('Bot returned illegal move ' + data.move + ' for FEN ' + game.fen());
        }
        board.position(game.fen());
        highlightLast(mv);
        renderMoves();
        var t = (data.elapsed_ms / 1000).toFixed(1);
        var label = data.is_book ? 'Your move (book reply, ' + t + 's)' : 'Your move (' + t + 's)';
        setStatus(label, data.is_book ? 'book' : null);
        if (checkGameOver()) {
          // game over already set status
        }
      })
      .catch(function (err) {
        console.error(err);
        setStatus('Could not reach the bot: ' + err.message + '. The HF Space may be sleeping; retry in a few seconds.', 'error');
        // Roll back the human move so they can try again
        game.undo();
        board.position(game.fen());
        renderMoves();
      })
      .finally(function () {
        setSpinner(false);
        botBusy = false;
      });
  }

  function newGame() {
    game = new Chess();
    humanColor = $colorSel.value;
    board.orientation(humanColor);
    board.position('start');
    $('#playBoard .square-55d63').css('box-shadow', '');
    renderMoves();
    $btnUndo.disabled = true;
    if (humanColor === 'white') {
      setStatus('Your move (you are White).');
    } else {
      setStatus('Bot is thinking…');
      setSpinner(true);
      botBusy = true;
      requestBotMove();
    }
  }

  function undoMove() {
    if (botBusy) return;
    // Undo the bot's last reply + the human's move (so it's the human's turn again)
    var u1 = game.undo();
    var u2 = game.undo();
    if (!u1 && !u2) return;
    board.position(game.fen());
    var hist = game.history({ verbose: true });
    highlightLast(hist[hist.length - 1]);
    renderMoves();
    setStatus('Your move.');
    if (game.history().length === 0) $btnUndo.disabled = true;
  }

  function init() {
    if (typeof Chessboard !== 'function' || typeof Chess !== 'function') {
      setStatus('Failed to load chessboard libraries (CDN blocked?). Refresh to retry.', 'error');
      return;
    }
    board = Chessboard('playBoard', {
      draggable: true,
      position: 'start',
      pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
      onDragStart: onDragStart,
      onDrop: onDrop,
      onSnapEnd: onSnapEnd,
    });
    window.addEventListener('resize', function () { board.resize(); });

    $btnNew.addEventListener('click', newGame);
    $btnUndo.addEventListener('click', undoMove);
    $btnFlip.addEventListener('click', function () { board.flip(); });
    $colorSel.addEventListener('change', function () {
      humanColor = $colorSel.value;
      // Don't auto-restart; let the user click "New game"
      setStatus('Click "New game" to start as ' + humanColor + '.');
    });

    setStatus('Your move (you are White). The bot may take ~10s on its first reply while waking up.');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
