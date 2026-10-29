const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
function source(name) {
  const start = html.search(new RegExp(`(?:async )?function ${name}\\(`));
  const rest = html.slice(start);
  return rest.slice(0, rest.slice(1).search(/\n(?:async )?function /) + 1);
}
function field() {
  return { _value: '', get value() { return this._value; }, set value(v) { this._value = String(v); },
    disabled: false, textContent: '', replaceChildren() {}, appendChild() {} };
}
function setup() {
  const writes = [];
  const stored = { v2: { num_steps: 20 }, v3: { num_steps: 20, checkpoint_v3: 'legacy', sway_coeff: 0, t_schedule_mode: 'sway' },
    v4: { num_steps: 20, checkpoint_v4: '', duration_scale: 1 } };
  const c = vm.createContext({ API: '', ttsConfigDrafts: null, ttsConfigDirty: new Set(), ttsEditingProfile: 'v4',
    document: { createElement: field },
    fetch: async (_url, options) => {
      if (options) writes.push(JSON.parse(options.body));
      return { ok: true, headers: { get: () => 'v4' }, json: async () => JSON.parse(JSON.stringify(stored)) };
    },
  });
  for (const name of ['Generation', 'Profile', 'ModelPreset', 'Checkpoint', 'Steps', 'Schedule', 'Sway', 'SamplingHint', 'GenerationError', 'Version']) c['$acTts' + name] = field();
  c.$acTtsVersion.value = 'auto';
  const functions = ['ttsModelPresets', 'updateTtsSamplingControls', 'readTtsGenerationForm', 'stashTtsGenerationForm',
    'renderTtsGenerationForm', 'loadTtsGenerationSettings'];
  for (const name of functions) vm.runInContext(source(name), c);
  // The following event handlers are outside the last function; extract through its closing brace.
  const save = html.slice(html.indexOf('async function saveTtsGenerationSettings()'), html.indexOf("$acTtsProfile.addEventListener('change'"));
  vm.runInContext(save, c);
  return { c, writes };
}
test('profile switching keeps drafts; saving sends only edited fields and profiles', async () => {
  const { c, writes } = setup();
  await c.loadTtsGenerationSettings();
  assert.equal(c.ttsEditingProfile, 'v4');
  c.$acTtsCheckpoint.value = 'Aratako/Irodori-TTS-v4.1-Small-MF';
  c.$acTtsSteps.value = 4;
  c.stashTtsGenerationForm(); c.renderTtsGenerationForm('v3');
  assert.equal(c.$acTtsSteps.value, '20');
  c.stashTtsGenerationForm(); c.renderTtsGenerationForm('v4');
  assert.equal(c.$acTtsSteps.value, '4');
  assert.equal(writes.length, 0);
  await c.saveTtsGenerationSettings();
  assert.equal(writes.length, 1);
  assert.equal(writes[0].v4.num_steps, 4);
  assert.equal(writes[0].v3, undefined);
  assert.equal(writes[0].v4.duration_scale, undefined);
});
test('closing without save and reopening discards local edits', async () => {
  const { c, writes } = setup();
  await c.loadTtsGenerationSettings();
  c.$acTtsSteps.value = 5; c.stashTtsGenerationForm();
  await c.loadTtsGenerationSettings();
  assert.equal(c.$acTtsSteps.value, '20');
  assert.equal(c.ttsConfigDirty.size, 0);
  assert.equal(writes.length, 0);
});
test('Sway controls follow schedule, API profile and MeanFlow availability', async () => {
  const { c } = setup();
  await c.loadTtsGenerationSettings();
  c.$acTtsSchedule.value = 'sway'; c.updateTtsSamplingControls();
  assert.equal(c.$acTtsSway.disabled, false);
  c.$acTtsCheckpoint.value = 'Aratako/Irodori-TTS-v4.1-Small-MF'; c.updateTtsSamplingControls();
  assert.equal(c.$acTtsSway.disabled, true);
  assert.match(c.$acTtsSamplingHint.textContent, /MeanFlow/);
  c.renderTtsGenerationForm('v2');
  assert.equal(c.$acTtsSchedule.disabled, true);
});
test('invalid steps never trigger a settings write', async () => {
  const { c, writes } = setup();
  await c.loadTtsGenerationSettings();
  c.$acTtsSteps.value = '1.5';
  await assert.rejects(c.saveTtsGenerationSettings(), /整数/);
  assert.equal(writes.length, 0);
});
