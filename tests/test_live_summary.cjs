const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const script = readFileSync(resolve(__dirname, '../docs/live.js'), 'utf8');

function browser() {
  const elements = new Map();
  const downloads = [];
  const makeElement = () => ({
    dataset: {},
    addEventListener(type, handler) { this[type] = handler; },
    getContext() { return { fillRect() {}, clearRect() {} }; },
    pause() {},
    remove() {},
  });
  let requestedFile;
  const document = {
    body: { append() {} },
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, makeElement());
      return elements.get(id);
    },
    createElement(tag) {
      const element = makeElement();
      if (tag === 'a') element.click = () => downloads.push({ name: element.download, file: requestedFile });
      return element;
    },
  };
  const context = vm.createContext({
    document, Blob, clearTimeout,
    setTimeout() {},
    URL: { createObjectURL(file) { requestedFile = file; return 'blob:test'; } },
    performance: { now: () => 1000 },
    window: { addEventListener() {} },
  });
  vm.runInContext(script, context);
  return { run: (code) => vm.runInContext(code, context), elements, downloads };
}

function snapshot({ source = 'sample', status = 'breathing measured', action = 'report',
  reason = 'rate confirmed', bpm = 15.2, outcome = null } = {}) {
  const state = {
    t: 151.3, view: 2, views: 3, window: 30, min_seconds: 25, tracking: 'ok',
    decision: { final: action !== 'wait', action, status, reason, bpm, completed_at_s: 151.3 },
    reading: { bpm: 15.2, confidence: .83, reason: 'ok' },
    prompt: { state: outcome ? 'done' : 'idle', outcome },
    samples: [[151.3, 99.999]],
  };
  return `session = { mode: '${source}', assessmentStartedAt: 123, started: true }; render(${JSON.stringify(state)});`;
}

test('exports the first completed snapshot and keeps it available after Stop', async () => {
  const { run, elements, downloads } = browser();
  run(snapshot({ action: 'wait' }));
  const button = elements.get('download-summary');
  assert.equal(button.disabled, true);
  button.click();
  assert.equal(downloads.length, 0);

  run(snapshot());
  assert.equal(button.disabled, false);
  run(snapshot({ bpm: 19, reason: 'changed' }));
  run('stop()');
  assert.equal(button.disabled, false);
  button.click();

  assert.equal(downloads[0].name, 'ostinato-assessment.txt');
  assert.equal(downloads[0].file.type, 'text/plain;charset=utf-8');
  const content = await downloads[0].file.text();
  assert.match(content, /Generated sample \(15 breaths\/min reference\)/);
  assert.match(content, /Completed after: 28\.3 seconds/);
  assert.match(content, /Assessment outcome: Breathing measured/);
  assert.match(content, /Breathing rate: 15\.2 breaths\/min/);
  assert.match(content, /Signal score of confirmed rate: 0\.83/);
  assert.doesNotMatch(content, /99\.999|changed/);
});

test('a hand raise and an inconclusive response stay separate from breathing measurements', async () => {
  const { run, elements, downloads } = browser();
  run(snapshot({ source: 'camera', status: 'responded', reason: 'response seen', bpm: null, outcome: 'response' }));
  const button = elements.get('download-summary');
  button.click();
  const handRaise = await downloads[0].file.text();
  assert.match(handRaise, /Source: Webcam/);
  assert.match(handRaise, /Assessment outcome: Responded/);
  assert.match(handRaise, /Breathing rate: No confirmed rate/);
  assert.match(handRaise, /Response check: Hand raise seen/);
  assert.doesNotMatch(handRaise, /15\.2|0\.83/);

  run(`session.command = { t: 152 }; onWorkerMessage(session, {
    requestType: 'command', type: 'state', t: 152, action: 'reset',
    state: {
      t: 152, samples: [], reading: { reason: 'need more data' },
      decision: { final: false, action: 'wait' }, prompt: { state: 'idle' },
    },
  });`);
  assert.equal(button.disabled, true);

  run(snapshot({ source: 'camera', status: 'unable to assess', action: 'get help',
    reason: 'time limit reached', bpm: null, outcome: 'none seen' }));
  button.click();
  const inconclusive = await downloads[1].file.text();
  assert.match(inconclusive, /Assessment outcome: Unable to assess/);
  assert.match(inconclusive, /Response check: None seen \(inconclusive\)/);
  assert.match(inconclusive, /Breathing rate: No confirmed rate/);
});
