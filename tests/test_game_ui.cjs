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
  addEventListener() {}
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
    document: { createElement: () => new Element(), createDocumentFragment: () => new Element() },
    gameView: initial, gameViewRoot: new Element(), gameUiPhase: 'active',
    gameRequest: null, gameDisplayGeneration: 1, conversationEpoch: 'epoch', pageId: 'page', API: '',
    $gameModeBadge: new Element(), $input: new Element(), $sendBtn: new Element(),
    gameRequestId: () => 'request', scrollBottom() {}, showErrorToast(message) { context.error = message; },
    fetch: () => pending, setTimeout,
  });
  for (const name of ['gameVisible', 'setGamePhase', 'makeGameEventNode', 'renderGameView',
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
});
