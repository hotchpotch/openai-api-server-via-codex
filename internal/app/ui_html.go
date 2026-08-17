package app

// The console is one self-contained page: no CDN, no build step, no external
// fetches. It is served to a single operator on an internal host, so the whole
// thing staying in one string is worth more than any asset pipeline.

const uiHTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Codex Proxy — Console</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f7f9; --panel: #fff; --ink: #16181d; --muted: #666c78;
    --line: #e2e5ea; --accent: #2f6feb; --ok: #1a7f45; --bad: #b3261e;
    --code-bg: #eef1f6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --panel: #1c1f25; --ink: #e8eaee; --muted: #9aa2b1;
      --line: #2b2f37; --accent: #6f9dff; --ok: #4cc38a; --bad: #ff6b60;
      --code-bg: #262a32;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
    display: flex; justify-content: center; padding: 32px 16px;
  }
  main { width: 100%; max-width: 640px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  section {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 18px; margin-bottom: 14px;
  }
  h2 { font-size: 14px; margin: 0 0 12px; text-transform: uppercase;
       letter-spacing: .06em; color: var(--muted); }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  input, textarea {
    width: 100%; padding: 9px 11px; border: 1px solid var(--line);
    border-radius: 7px; background: var(--bg); color: var(--ink);
    font: inherit; margin-bottom: 10px;
  }
  textarea { min-height: 130px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size: 12.5px; resize: vertical; }
  button {
    padding: 9px 15px; border: 1px solid var(--line); border-radius: 7px;
    background: var(--accent); color: #fff; font: inherit; font-weight: 600;
    cursor: pointer; margin-right: 8px;
  }
  button.ghost { background: transparent; color: var(--ink); font-weight: 500; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .row { display: flex; justify-content: space-between; gap: 12px;
         padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 14px; }
  .row:last-child { border-bottom: 0; }
  .row span:first-child { color: var(--muted); }
  .row span:last-child { text-align: right; word-break: break-all; }
  .ok { color: var(--ok); font-weight: 600; }
  .bad { color: var(--bad); font-weight: 600; }
  .code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 27px; letter-spacing: .13em; background: var(--code-bg);
    padding: 14px; border-radius: 8px; text-align: center; margin: 10px 0;
    user-select: all;
  }
  .note { font-size: 13px; color: var(--muted); margin: 8px 0 0; }
  .msg { padding: 10px 12px; border-radius: 7px; margin-bottom: 12px;
         font-size: 13.5px; display: none; }
  .msg.show { display: block; }
  .msg.err { background: color-mix(in srgb, var(--bad) 12%, transparent); color: var(--bad); }
  .msg.good { background: color-mix(in srgb, var(--ok) 14%, transparent); color: var(--ok); }
  a { color: var(--accent); }
  .hide { display: none; }
</style>
</head>
<body>
<main>
  <h1>Codex Proxy</h1>
  <div class="sub">Operator console — sign this proxy in to a ChatGPT account.</div>

  <div id="msg" class="msg"></div>

  <section id="gate">
    <h2>Console access</h2>
    <label for="key">API key</label>
    <input id="key" type="password" autocomplete="current-password" placeholder="OPENAI_VIA_CODEX_API_KEY">
    <button id="unlock">Unlock</button>
  </section>

  <div id="app" class="hide">
    <section>
      <h2>Status</h2>
      <div id="status"></div>
      <div style="margin-top:14px">
        <button class="ghost" id="refresh">Refresh</button>
        <button class="ghost" id="lock">Lock console</button>
      </div>
    </section>

    <section>
      <h2>Sign in with a device code</h2>
      <div id="device-idle">
        <p class="note">Starts a sign-in and shows a one-time code. Open the link on any
          device with a browser, enter the code, and this proxy receives the tokens.</p>
        <button id="start">Start sign-in</button>
      </div>
      <div id="device-active" class="hide">
        <div class="code" id="user-code">…</div>
        <p class="note">Open <a id="verify-url" href="#" target="_blank" rel="noreferrer noopener">the verification page</a>
          and enter this code. It expires in about 15 minutes.</p>
        <p class="note" id="device-msg"></p>
        <button class="ghost" id="cancel">Cancel</button>
      </div>
    </section>

    <section>
      <h2>Paste an auth.json</h2>
      <p class="note" style="margin-top:0">Fallback for when device sign-in is unavailable.
        Run <code>codex login</code> elsewhere, then paste the resulting
        <code>~/.codex/auth.json</code> here.</p>
      <textarea id="authjson" spellcheck="false" placeholder='{"auth_mode":"chatgpt","tokens":{...}}'></textarea>
      <button id="save">Save auth.json</button>
      <button class="ghost" id="signout">Sign out</button>
    </section>
  </div>
</main>

<script>
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var poll = null;

  function show(text, kind) {
    var box = $('msg');
    box.textContent = text;
    box.className = 'msg show ' + (kind || 'err');
    if (kind === 'good') { setTimeout(function () { box.className = 'msg'; }, 4000); }
  }

  function api(path, body) {
    return fetch('/ui/api/' + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'same-origin'
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) { throw new Error(data.error || ('Request failed (' + r.status + ')')); }
        return data;
      });
    });
  }

  function row(label, value, cls) {
    return '<div class="row"><span>' + label + '</span><span' +
      (cls ? ' class="' + cls + '"' : '') + '>' + value + '</span></div>';
  }

  function esc(v) {
    return String(v === undefined || v === null ? '—' : v).replace(/[&<>"]/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c];
    });
  }

  function renderStatus(s) {
    var html = '';
    html += s.signed_in
      ? row('Signed in', 'yes', 'ok')
      : row('Signed in', 'no', 'bad');
    if (s.email) { html += row('Account', esc(s.email)); }
    if (s.account_id) { html += row('Account id', esc(s.account_id)); }
    if (s.expires_at) {
      var mins = Math.round((s.expires_in_seconds || 0) / 60);
      html += row('Token expires', esc(s.expires_at) + ' (' + mins + ' min)');
    }
    html += row('auth.json', s.auth_file_present ? esc(s.auth_path) : 'missing',
      s.auth_file_present ? '' : 'bad');
    html += row('Default model', esc(s.default_model));
    html += row('Codex CLI', s.codex_cli ? 'available' : 'not installed',
      s.codex_cli ? '' : 'bad');
    if (!s.signed_in && s.error) { html += row('Detail', esc(s.error), 'bad'); }
    $('status').innerHTML = html;

    var d = s.signin || {};
    // Only a live sign-in shows the code panel. Once the CLI stops — completed,
    // cancelled or failed — the code is dead, so keying off user_code alone
    // would leave an unusable code on screen with no way back to the start.
    if (d.running) {
      $('device-idle').className = 'hide';
      $('device-active').className = '';
      $('user-code').textContent = d.user_code || '…';
      $('verify-url').href = d.url || 'https://auth.openai.com/codex/device';
      $('device-msg').textContent = d.message || '';
    } else {
      $('device-idle').className = '';
      $('device-active').className = 'hide';
      // Surface why a finished attempt ended, since the panel it happened in
      // has just been replaced by the idle one.
      if (d.failed && d.message) { show(d.message); }
    }
    // Stop polling once the login settles, so an idle console is quiet.
    if (!d.running && poll) { clearInterval(poll); poll = null; }
    if (!s.codex_cli) { $('start').disabled = true; }
  }

  function refresh() {
    return api('status').then(renderStatus).catch(function (e) {
      if (/not signed in/i.test(e.message)) { lock(); } else { show(e.message); }
    });
  }

  function unlock() {
    $('gate').className = 'hide';
    $('app').className = '';
    refresh();
  }

  function lock() {
    $('gate').className = '';
    $('app').className = 'hide';
    if (poll) { clearInterval(poll); poll = null; }
  }

  $('unlock').onclick = function () {
    api('login', { key: $('key').value }).then(function () {
      $('key').value = '';
      $('msg').className = 'msg';
      unlock();
    }).catch(function (e) { show(e.message); });
  };

  $('lock').onclick = function () {
    api('logout', {}).then(lock).catch(function (e) { show(e.message); });
  };

  $('refresh').onclick = refresh;

  $('start').onclick = function () {
    $('start').disabled = true;
    api('signin/start', {}).then(function () {
      if (poll) { clearInterval(poll); }
      poll = setInterval(refresh, 3000);
      return refresh();
    }).catch(function (e) { show(e.message); })
      .then(function () { $('start').disabled = false; });
  };

  $('cancel').onclick = function () {
    api('signin/cancel', {}).then(refresh).catch(function (e) { show(e.message); });
  };

  $('save').onclick = function () {
    api('auth/paste', { content: $('authjson').value }).then(function () {
      $('authjson').value = '';
      show('Saved. Credentials reloaded.', 'good');
      return refresh();
    }).catch(function (e) { show(e.message); });
  };

  $('signout').onclick = function () {
    if (!confirm('Remove auth.json? The proxy stops serving until it is signed in again.')) { return; }
    api('auth/delete', {}).then(function () {
      show('Signed out.', 'good');
      return refresh();
    }).catch(function (e) { show(e.message); });
  };

  $('key').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { $('unlock').click(); }
  });

  // A live session cookie means the gate can be skipped on reload.
  api('status').then(function (s) { unlock(); renderStatus(s); }).catch(function () { lock(); });
})();
</script>
</body>
</html>
`
