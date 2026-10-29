const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Run the actual inline UI functions with a small DOM and delayed HTTP responses.
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
function source(name) {
  const start = html.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(start >= 0, name);
  const rest = html.slice(start);
  const end = rest.slice(1).search(/\n(?:async )?function /);
  return rest.slice(0, end + 1);
}
class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.attributes = {};
    this.className = ''; this.textContent = '';
    this.classList = {
      toggle: (name, on) => {
        const classes = new Set(this.className.split(' ').filter(Boolean));
        on ? classes.add(name) : classes.delete(name);
        this.className = [...classes].join(' ');
      },
    };
  }
  append(...nodes) { nodes.forEach(node => this.appendChild(node)); }
  appendChild(node) { node.parent = this; this.children.push(node); }
  replaceChildren() { this.children = []; }
  remove() { this.parent.children = this.parent.children.filter(node => node !== this); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { this.listeners ||= {}; this.listeners[name] = handler; }
  querySelectorAll(selector) {
    return this.children.flatMap(node => [
      ...(node.className.split(' ').includes(selector.slice(1)) ? [node] : []),
      ...node.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0]; }
}
function setup() {
  const choices = [
    { id: 'open', number: 1, label: '扉を開ける' },
    { id: 'leave', number: 2, label: '別の出口を探す' },
  ];
  const initial = {
    game_id: 'game', status: 'active', revision: 1,
    interaction: { id: 'interaction-1', choices },
    transcript: [{ type: 'interaction', id: 'card-1', choices }],
  };
  let finish;
  const pending = new Promise(resolve => { finish = resolve; });
  const context = vm.createContext({
    TextDecoder, selectedName: 'ミオ', selectedImageUrl: null,
    gameAudioPending: new Set(), gameAudioRequested: new Set(['char-stream']), gameAudioCache: new Map(), gameAudioFailed: new Set(), ttsEnabled: false,
    document: { createElement: () => new Element(), createDocumentFragment: () => new Element() },
    gameView: initial, gameViewRoot: new Element(), $messages: new Element(), gameUiPhase: 'active',
    gameRequest: null, gameDisplayGeneration: 1, conversationEpoch: 'epoch', pageId: 'page', API: '',
    $gameModeBadge: new Element(), $input: new Element(), $sendBtn: new Element(),
    gameRequestId: () => 'request', scrollBottom() {}, showErrorToast(message) { context.error = message; },
    fetch: () => pending, setTimeout,
  });
  for (const name of ['gameVisible', 'clearGameStreamPreview', 'updateGameProgress', 'fetchGameResponse', 'appendGameExperimentalNotice', 'gameFailureMessage', 'showGameFailure', 'setGamePhase', 'makeGameEventNode', 'renderGameView',
    'mergeGameTranscript', 'showGameWorking', 'submitGameInput', 'submitGameChoice', 'recoverGameRequest']) {
    vm.runInContext(source(name), context);
  }
  context.renderGameView(initial);
  return { context, finish, initial };
}

test('choice is visible and locked before HTTP completes, then committed once', async () => {
  const { context: c, finish, initial } = setup();
  const sending = c.submitGameChoice('interaction-1', 'open');
  assert.equal(c.gameUiPhase, 'submitting');
  assert.equal(c.gameViewRoot.querySelector('.game-pending-selection').textContent, '送信済み：扉を開ける');
  const buttons = c.gameViewRoot.querySelectorAll('.game-choice');
  assert.ok(buttons.every(button => button.disabled));
  assert.equal(buttons[0].attributes['aria-pressed'], 'true');
  assert.equal(buttons[1].attributes['aria-pressed'], 'false');
  await c.submitGameChoice('interaction-1', 'leave');
  assert.equal(c.gameRequest.selectedChoiceId, 'open');
  finish({ ok: true, json: async () => ({
    ...initial, revision: 2, transcript: undefined,
    interaction: { id: 'interaction-2', choices: initial.interaction.choices },
    events: [{ id: 'player-1', type: 'player', text: '扉を開ける' },
      { id: 'card-2', type: 'interaction', choices: initial.interaction.choices }],
  }) });
  await sending;
  assert.equal(c.gameRequest, null);
  assert.equal(c.gameViewRoot.querySelector('.game-pending-selection'), undefined);
  assert.equal(c.gameViewRoot.querySelectorAll('.game-player-event').length, 1);
  const after = c.gameViewRoot.querySelectorAll('.game-choice');
  assert.ok(after.slice(0, 2).every(button => button.disabled));
  assert.ok(after.slice(2).every(button => !button.disabled));
});

test('fullwidth numeric selection immediately displays the choice label', async () => {
  const { context: c, finish } = setup();
  const sending = c.submitGameInput({ kind: 'text', text: '２', interaction_id: 'interaction-1' }, '２');
  assert.equal(c.gameViewRoot.querySelector('.game-pending-selection').textContent, '送信済み：別の出口を探す');
  finish({ ok: true, json: async () => ({ ...c.gameView, events: [] }) });
  await sending;
});

test('failed generation clears pending feedback and allows choosing again', async () => {
  const { context: c, finish } = setup();
  const sending = c.submitGameChoice('interaction-1', 'open');
  c.fetch = async () => ({ ok: true, json: async () => ({ status: 'failed', error: 'INVALID_MODEL_OUTPUT' }) });
  finish({ ok: false, status: 502, json: async () => ({ detail: { code: 'INVALID_MODEL_OUTPUT' } }) });
  await sending;
  assert.equal(c.gameUiPhase, 'active');
  assert.equal(c.gameRequest, null);
  assert.equal(c.gameViewRoot.querySelector('.game-pending-selection'), undefined);
  assert.ok(c.gameViewRoot.querySelectorAll('.game-choice').every(button => !button.disabled));
  assert.match(c.error, /INVALID_MODEL_OUTPUT/);
  assert.match(c.gameViewRoot.querySelector('.game-error-notice').textContent, /もう一度お試し/);
  assert.match(c.error, /コンテキスト長/);
});

test('experimental notice stays visible; terminal failures give appropriate guidance', () => {
  const { context: c } = setup();
  assert.match(c.gameViewRoot.querySelector('.game-notice').textContent, /実験的/);
  const closed = c.gameFailureMessage('GAME_CLOSED');
  assert.match(closed, /新しく開始/);
  assert.doesNotMatch(closed, /進行は更新されていません|コンテキスト長/);
  const start = c.gameFailureMessage('INVALID_MODEL_OUTPUT', 'start');
  assert.match(start, /もう一度ゲームモードを開始/);
});

test('unknown turn outcome keeps the same request and never recommends a new submission', async () => {
  const { context: c, finish } = setup();
  c.setTimeout = callback => callback();
  const sending = c.submitGameChoice('interaction-1', 'open');
  const requestId = c.gameRequest.body.request_id;
  c.fetch = async () => ({ ok: true, json: async () => ({ status: 'running' }) });
  finish({ ok: false, status: 502, json: async () => ({ detail: { code: 'LLM_FAILED' } }) });
  await sending;
  assert.equal(c.gameRequest.body.request_id, requestId);
  assert.ok(c.gameViewRoot.querySelectorAll('.game-choice').every(button => button.disabled));
  const notice = c.gameViewRoot.querySelector('.game-error-notice').textContent;
  assert.match(notice, /同じ要求の結果を再確認/);
  assert.doesNotMatch(notice, /もう一度お試し/);
});

test('unknown start outcome can be rechecked with the original start request', async () => {
  const { context: c, initial } = setup();
  vm.runInContext(source('requestGameStart'), c);
  c.gameUiPhase = 'starting';
  c.gameStartRequest = { request_id: 'start-once' };
  c.setTimeout = callback => callback();
  c.fetch = async () => ({ ok: true, status: 202, json: async () => ({ detail: { code: 'REQUEST_RUNNING' } }) });
  await c.requestGameStart(c.gameStartRequest, 1);
  const retry = c.gameViewRoot.querySelector('.game-choice');
  assert.equal(retry.textContent, '開始結果を再確認');
  let body;
  c.ensureGameLease = () => {};
  c.fetch = async (_url, options) => { body = JSON.parse(options.body); return { ok: true, json: async () => initial }; };
  await retry.listeners.click();
  assert.equal(body.request_id, 'start-once');
  assert.equal(c.gameUiPhase, 'active');
});

function eventStream() {
  let controller;
  const body = new ReadableStream({ start(value) { controller = value; } });
  return { response: { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body },
    send(event) {
      const bytes = new TextEncoder().encode('data: ' + JSON.stringify(event) + '\n\n');
      // Exercise Japanese UTF-8 split across network chunks and SSE frame boundaries.
      for (let i = 0; i < bytes.length; i += 2) controller.enqueue(bytes.slice(i, i + 2));
    }, close() { controller.close(); } };
}

test('Japanese previews appear before commit, reset on retry and become a single final reply', async () => {
  const { context: c, initial } = setup();
  const stream = eventStream(); c.fetch = async () => stream.response;
  const sending = c.submitGameChoice('interaction-1', 'open');
  stream.send({ type: 'narration_preview', text: '窓から風が入ってきた。' });
  await new Promise(setImmediate);
  assert.equal(c.gameViewRoot.querySelector('.game-narration-preview-text').textContent, '窓から風が入ってきた。');
  assert.equal(c.gameViewRoot.querySelector('.game-stream-preview'), undefined);
  stream.send({ type: 'phase', phase: 'character' });
  stream.send({ type: 'character_preview', text: 'こんにちは。' });
  await new Promise(setImmediate);
  assert.equal(c.gameViewRoot.querySelector('.game-stream-text').textContent, 'こんにちは。');
  assert.equal(c.gameView.revision, 1);
  assert.ok(c.gameRequest);
  stream.send({ type: 'phase', phase: 'character', retry: true });
  await new Promise(setImmediate);
  assert.equal(c.gameViewRoot.querySelector('.game-stream-preview'), undefined);
  assert.ok(c.gameViewRoot.querySelector('.game-narration-preview'));
  stream.send({ type: 'character_preview', text: '窓を開けたんだね。' });
  stream.send({ type: 'result', result: {
    ...initial, revision: 2, transcript: undefined,
    interaction: { id: 'interaction-2', choices: initial.interaction.choices },
    events: [{ id: 'char-stream', type: 'character', text: '窓を開けたんだね。' },
      { id: 'card-2', type: 'interaction', choices: initial.interaction.choices }],
  } });
  await sending;
  assert.equal(c.gameViewRoot.querySelector('.game-stream-preview'), undefined);
  assert.equal(c.gameViewRoot.querySelectorAll('.game-character-bubble').length, 1);
  assert.equal(c.gameViewRoot.querySelector('.game-narration-preview'), undefined);
  assert.equal(c.gameView.revision, 2);
  assert.equal(c.gameRequest, null);
});

test('interrupted stream removes provisional text and recovers the same request', async () => {
  const { context: c, initial } = setup();
  const stream = eventStream(); c.fetch = async () => stream.response;
  const sending = c.submitGameChoice('interaction-1', 'open');
  stream.send({ type: 'character_preview', text: '途中の返答' });
  await new Promise(setImmediate);
  const requestId = c.gameRequest.body.request_id;
  let recoveryUrl;
  c.fetch = async url => { recoveryUrl = url; return { ok: true, json: async () => ({ status: 'committed', result: initial }) }; };
  stream.close(); await sending;
  assert.match(recoveryUrl, new RegExp('/requests/' + requestId));
  assert.equal(c.gameViewRoot.querySelector('.game-stream-preview'), undefined);
  assert.equal(c.gameRequest, null);
});

test('failed streamed generation discards its preview and retains retryable choices', async () => {
  const { context: c } = setup();
  const stream = eventStream(); c.fetch = async () => stream.response;
  const sending = c.submitGameChoice('interaction-1', 'open');
  stream.send({ type: 'character_preview', text: '確定してはいけない返答' });
  stream.send({ type: 'narration_preview', text: '確定してはいけない地の文' });
  await new Promise(setImmediate);
  c.fetch = async () => ({ ok: true, json: async () => ({ status: 'failed', error: 'INVALID_MODEL_OUTPUT' }) });
  stream.send({ type: 'error', status: 502, code: 'INVALID_MODEL_OUTPUT' });
  await sending;
  assert.equal(c.gameViewRoot.querySelector('.game-stream-preview'), undefined);
  assert.equal(c.gameView.revision, 1);
  assert.equal(c.gameViewRoot.querySelector('.game-narration-preview'), undefined);
  assert.equal(c.gameRequest, null);
  assert.ok(c.gameViewRoot.querySelectorAll('.game-choice').every(button => !button.disabled));
  assert.match(c.error, /もう一度お試し/);
});
