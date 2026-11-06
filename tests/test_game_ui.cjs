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
    gameRequest: null, gameSaveRequest: null, gameLoadRequest: null, gameStorageBusy: false,
    gameDisplayGeneration: 1, conversationEpoch: 'epoch', pageId: 'page', API: '',
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

test('game can start while normal chat and audio are pending', async () => {
  const { context: c } = setup();
  Object.assign(c, {
    gameUiPhase: 'off', gameView: null, isGenerating: true,
    activePolls: new Set(['normal-audio']), isScenarioPlaying: false, isFileReading: false,
    autoReplyTimer: null, autoReplyEnabled: true, normalRevision: 4,
    stopAllAudio() { c.audioStopped = true; }, enterGameView() {},
    refreshConversationStatus: async () => ({ busy: true }),
    requestGameStart: async body => { c.startedBody = body; },
  });
  vm.runInContext(source('toggleGameMode'), c);
  await c.toggleGameMode();
  assert.equal(c.gameUiPhase, 'starting');
  assert.equal(c.startedBody.normal_revision, 4);
  assert.equal(c.autoReplyEnabled, false);
  assert.equal(c.audioStopped, true);
  assert.equal(c.error, undefined);
});

test('late normal completion preserves game controls and suppresses normal audio', () => {
  const { context: c } = setup();
  Object.assign(c, { isGenerating: true, audioQueue: [], audioQueueMeta: [] });
  for (const name of ['finishNormalGeneration', '_pushAudio', 'playNextAudio']) {
    vm.runInContext(source(name), c);
  }
  for (const phase of ['starting', 'submitting', 'active']) {
    c.setGamePhase(phase);
    c.finishNormalGeneration();
    assert.equal(c.$sendBtn.disabled, phase !== 'active');
    assert.equal(c.$input.disabled, phase !== 'active');
    assert.equal(c.isGenerating, false);
  }
  c._pushAudio('late-normal-audio', null);
  c.playNextAudio();
  assert.equal(c.audioQueue.length, 0);
});

test('switching to game stops normal audio and cancels delayed playback', () => {
  const { context: c } = setup();
  let audio;
  let fallback;
  Object.assign(c, {
    gameUiPhase: 'off', audioQueue: ['normal-audio'], audioQueueMeta: [], isAudioPlaying: false,
    getAudioCtx: () => null, _clearActivePlayBtn() {}, stopReplay() {},
    setTimeout(callback) { fallback = callback; return 7; },
    clearTimeout(id) { if (id === 7) c.timerCleared = true; },
    Audio: class {
      constructor() { audio = this; }
      setAttribute() {} load() {} pause() { this.paused = true; }
      play() { assert.fail('Stopped audio must not start'); }
    },
  });
  vm.runInContext('let stopCurrentNormalAudio = null;', c);
  for (const name of ['stopAllAudio', 'playNextAudio']) vm.runInContext(source(name), c);
  c.playNextAudio();
  c.stopAllAudio();
  c.setGamePhase('starting');
  fallback();
  assert.equal(audio.paused, true);
  assert.equal(audio.oncanplaythrough, null);
  assert.equal(c.timerCleared, true);
  assert.equal(c.audioQueue.length, 0);
  assert.equal(c.isAudioPlaying, false);
});

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

test('save locks choices and retries a lost response with the same request', async () => {
  const { context: c } = setup();
  for (const name of ['gameStorageError', 'saveGameProgress']) vm.runInContext(source(name), c);
  c.document.getElementById = () => ({ close() {} });
  const bodies = [];
  c.fetch = async (_url, options) => {
    assert.equal(c.gameUiPhase, 'saving');
    assert.ok(c.gameViewRoot.querySelectorAll('.game-choice').every(button => button.disabled));
    bodies.push(JSON.parse(options.body));
    throw new Error('lost');
  };
  await c.saveGameProgress('窓辺の続き');
  assert.equal(c.gameUiPhase, 'active');
  assert.ok(c.gameSaveRequest);
  c.fetch = async (_url, options) => {
    bodies.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ name: '窓辺の続き' }) };
  };
  await c.saveGameProgress();
  assert.deepEqual(bodies[0], bodies[1]);
  assert.equal(c.gameSaveRequest, null);
  assert.equal(c.gameStorageBusy, false);
  assert.match(c.error, /セーブしました/);
});

function setupLoad() {
  const result = setup(), c = result.context;
  for (const name of ['gameStorageError', 'requestGameLoad', 'loadGameSave', 'openGameSaves']) vm.runInContext(source(name), c);
  c.gameUiPhase = 'loading';
  c.gameLoadRequest = { saveId: 'saved', generation: 1, phase: 'active', body: { request_id: 'load-once' } };
  c.ensureGameLease = () => {}; c.closeSidebar = () => {};
  return result;
}

test('loading replaces history and character, unlocks choices, and never autoplays old speech', async () => {
  const { context: c, initial } = setupLoad();
  c.selectedName = '別のキャラクター';
  c.requestGameAudio = () => { throw new Error('loaded speech must not autoplay'); };
  const loaded = { ...initial, game_id: 'loaded-game', character_name: 'ミオ',
    transcript: [{ id: 'restored-speech', type: 'character', text: 'おかえり。' }, ...initial.transcript] };
  c.fetch = async () => ({ ok: true, json: async () => loaded });
  await c.requestGameLoad();
  assert.equal(c.gameLoadRequest, null);
  assert.equal(c.gameView.game_id, 'loaded-game');
  assert.equal(c.gameUiPhase, 'active');
  assert.equal(c.gameViewRoot.querySelector('.game-character-name').textContent, 'ミオ');
  assert.ok(c.gameAudioRequested.has('restored-speech'));
  assert.ok(c.gameViewRoot.querySelectorAll('.game-choice').every(button => !button.disabled));
});

test('uncertain load locks input and rechecks the same request, corrupt save retains current view', async () => {
  const { context: c, initial } = setupLoad();
  c.fetch = async () => { throw new Error('connection lost'); };
  await c.requestGameLoad();
  assert.equal(c.gameUiPhase, 'loading');
  assert.equal(c.gameLoadRequest.body.request_id, 'load-once');
  const retry = c.gameViewRoot.querySelector('.game-load-retry');
  assert.ok(retry);
  let sent;
  c.fetch = async (_url, options) => {
    sent = JSON.parse(options.body);
    return { ok: false, status: 422, json: async () => ({ detail: { code: 'SAVE_CORRUPT' } }) };
  };
  await retry.listeners.click();
  assert.equal(sent.request_id, 'load-once');
  assert.equal(c.gameLoadRequest, null);
  assert.equal(c.gameView, initial);
  assert.equal(c.gameUiPhase, 'active');
  assert.match(c.error, /破損/);
});

test('late load response cannot replace a newer display', async () => {
  const { context: c, finish, initial } = setupLoad();
  const loading = c.requestGameLoad();
  c.gameDisplayGeneration++;
  finish({ ok: true, json: async () => ({ ...initial, game_id: 'old-load' }) });
  await loading;
  assert.equal(c.gameView, initial);
});

test('load cancellation keeps current game and save names are rendered as text', async () => {
  const { context: c, initial } = setupLoad();
  c.gameUiPhase = 'active'; c.gameLoadRequest = null;
  c.isGenerating = false; c.activePolls = new Set(); c.isScenarioPlaying = false; c.isFileReading = false;
  const confirmDialog = { dataset: {}, showModal() {} };
  c.document.getElementById = () => confirmDialog;
  c.fetch = () => { throw new Error('cancel must not send a request'); };
  await c.loadGameSave('saved');
  assert.equal(confirmDialog.dataset.saveId, 'saved');
  assert.equal(c.gameView, initial);
  const list = new Element(), dialog = { showModal() {}, close() {} };
  c.document.getElementById = id => id === 'game-saves-list' ? list : dialog;
  c.fetch = async () => ({ ok: true, json: async () => ({ saves: [
    { id: 'saved', name: '<img onerror=alert(1)>', character_name: 'ミオ', world_turn: 4, saved_at: '2026-11-07T00:00:00Z' },
  ] }) });
  await c.openGameSaves();
  assert.equal(list.children[0].children[0].textContent, '<img onerror=alert(1)>');
  assert.match(list.children[0].children[1].textContent, /ターン 4/);
});
