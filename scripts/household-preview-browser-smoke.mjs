// Real Chromium against a fresh disposable household preview, never the live app.
// Supply TAMAGI_PREVIEW_PASSWORD through the environment; it is never reported.
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const preview = new URL(process.env.TAMAGI_PREVIEW_URL || 'http://127.0.0.1:18744');
assert(['127.0.0.1', 'localhost', '[::1]'].includes(preview.hostname), 'Only a loopback preview is allowed');
assert(['http:', 'https:'].includes(preview.protocol), 'Use the local preview HTTP service');
assert(!preview.username && !preview.password && !preview.search && !preview.hash, 'Preview URL must not contain credentials or query data');
assert.equal(preview.pathname, '/', 'Point at the preview origin');
const base = preview.origin;
const password = process.env.TAMAGI_PREVIEW_PASSWORD;
assert(password, 'TAMAGI_PREVIEW_PASSWORD must be supplied privately');
const debuggerPort = Number(process.env.TAMAGI_PREVIEW_BROWSER_PORT || 18745);
assert(Number.isInteger(debuggerPort) && debuggerPort >= 1024 && debuggerPort <= 65535);
assert.notEqual(String(debuggerPort), preview.port, 'CDP and the preview need separate ports');
const debuggerBase = `http://127.0.0.1:${debuggerPort}`;
const chrome = process.env.TAMAGI_CHROME || '/root/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome';
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const js = value => JSON.stringify(value);
const redact = value => String(value).split(password).join('[redacted]');

// Refuse to drive an arbitrary service or attach to somebody else's browser.
const healthResponse = await fetch(base + '/api/preview/health');
assert(healthResponse.ok, 'Preview health endpoint must be available');
const health = await healthResponse.json();
assert.equal(health.fixture_only, true);
assert.equal(health.real_model, false);
assert.equal(health.ready, true);
assert.equal((await fetch(base + '/api/preview/state', {redirect:'manual'})).status, 401, 'Preview state must require its private session');
let occupied = false;
try { occupied = (await fetch(debuggerBase + '/json/version')).ok; } catch {}
assert.equal(occupied, false, 'CDP port is occupied; refusing to attach to an existing browser');

const reportRoot = await mkdtemp(join(tmpdir(), 'tamagi-household-browser-report-'));
const profile = await mkdtemp(join(tmpdir(), 'tamagi-household-browser-profile-'));
const browserEnv = {...process.env};
delete browserEnv.TAMAGI_PREVIEW_PASSWORD;
const browser = spawn(chrome, [
  '--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
  '--no-first-run', '--no-default-browser-check', '--disable-default-apps',
  '--disable-background-networking', '--disable-component-update', '--disable-sync',
  '--disable-domain-reliability', '--disable-client-side-phishing-detection',
  '--disable-extensions', '--disable-breakpad', '--metrics-recording-only',
  '--safebrowsing-disable-auto-update', '--password-store=basic', '--use-mock-keychain',
  '--disable-features=MediaRouter,OptimizationHints,AutofillServerCommunication,CertificateTransparencyComponentUpdater,UseDnsHttpsSvcbAlpn',
  '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1, EXCLUDE [::1]',
  '--remote-debugging-address=127.0.0.1', `--remote-debugging-port=${debuggerPort}`,
  `--user-data-dir=${profile}`, 'about:blank',
], {stdio:'ignore', env:browserEnv});

const errors = [], blockedRequests = [], requests = [], checks = [];
let browserError;
browser.on('error', error => { browserError = error; });

class Page {
  constructor(socket) {
    this.socket = socket;
    this.next = 1;
    this.pending = new Map();
    socket.onmessage = event => {
      const value = JSON.parse(event.data);
      if (value.id) {
        const waiting = this.pending.get(value.id);
        if (!waiting) return;
        this.pending.delete(value.id);
        clearTimeout(waiting.timer);
        value.error ? waiting.reject(new Error(JSON.stringify(value.error))) : waiting.resolve(value.result);
      } else if (value.method === 'Runtime.exceptionThrown') {
        errors.push(redact(value.params.exceptionDetails.exception?.description || value.params.exceptionDetails.text));
      } else if (value.method === 'Fetch.requestPaused') {
        const request = value.params;
        const url = new URL(request.request.url);
        const allowed = url.origin === base;
        if (allowed) {
          requests.push({method:request.request.method, path:url.pathname});
          this.send('Fetch.continueRequest', {requestId:request.requestId}).catch(error => errors.push(redact(error.message)));
        } else {
          blockedRequests.push({origin:url.origin, path:url.pathname});
          this.send('Fetch.failRequest', {requestId:request.requestId, errorReason:'BlockedByClient'}).catch(error => errors.push(redact(error.message)));
        }
      }
    };
  }
  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.next++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('CDP timeout: ' + method));
      }, 20000);
      this.pending.set(id, {resolve, reject, timer});
      this.socket.send(JSON.stringify({id, method, params}));
    });
  }
  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', {expression, awaitPromise:true, returnByValue:true});
    if (result.exceptionDetails) throw new Error(redact(result.exceptionDetails.exception?.description || result.exceptionDetails.text));
    return result.result?.value;
  }
  async until(expression, description, timeout = 15000) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      try { if (await this.evaluate(expression)) return; } catch {}
      await sleep(80);
    }
    throw new Error('Browser condition failed: ' + description);
  }
  async screenshot(name, fullPage = false) {
    const options = {format:'png', captureBeyondViewport:fullPage};
    if (fullPage) {
      const metrics = await this.send('Page.getLayoutMetrics');
      options.clip = {...metrics.cssContentSize, scale:1};
    }
    const result = await this.send('Page.captureScreenshot', options);
    await writeFile(join(reportRoot, name + '.png'), Buffer.from(result.data, 'base64'));
  }
  async click(selector) {
    await this.evaluate(`(() => { const button = document.querySelector(${js(selector)}); if (!button || button.disabled) throw new Error('Required enabled control is unavailable'); button.click(); })()`);
  }
  async fill(id, value) {
    await this.evaluate(`(() => { const input = document.getElementById(${js(id)}); input.value = ${js(value)}; input.dispatchEvent(new Event('input', {bubbles:true})); })()`);
  }
}

const stateExpression = `fetch('/api/preview/state', {credentials:'same-origin'}).then(async response => { if (!response.ok) throw new Error('Preview state request failed'); return response.json(); })`;
const stateOf = page => page.evaluate(stateExpression);
const dataOf = record => record.data || record;
let page, summary;

try {
  await mkdir(reportRoot, {recursive:true});
  let ready = false;
  for (let count = 0; count < 100; count++) {
    if (browserError) throw browserError;
    try { ready = (await fetch(debuggerBase + '/json/version')).ok; } catch {}
    if (ready) break;
    await sleep(80);
  }
  assert(ready, 'Disposable Chromium must start');
  const target = await fetch(debuggerBase + '/json/new?about%3Ablank', {method:'PUT'}).then(response => response.json());
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  page = new Page(socket);
  await page.send('Runtime.enable');
  await page.send('Page.enable');
  await page.send('Network.enable');
  await page.send('Fetch.enable', {patterns:[{urlPattern:'*', requestStage:'Request'}]});
  await page.send('Emulation.setDeviceMetricsOverride', {width:1280, height:1050, deviceScaleFactor:1, mobile:false});
  await page.send('Page.navigate', {url:base + '/'});
  await page.until(`location.pathname === '/login' && !!document.getElementById('password')`, 'private preview login');
  await page.screenshot('login');
  await page.fill('password', password);
  await page.click('#login button');
  await page.until(`location.pathname === '/' && document.querySelectorAll('#residents .resident').length === 3`, 'three scripted fixture homes loaded');
  await page.send('Page.bringToFront');
  await page.until(`document.hasFocus() && !document.hidden`, 'foreground browser for reading');
  const cookies = (await page.send('Network.getCookies', {urls:[base]})).cookies;
  const privateCookie = cookies.find(cookie => cookie.name.startsWith('tamagi_household_preview_'));
  assert(privateCookie?.httpOnly && privateCookie.sameSite === 'Strict', 'Preview session uses a private HttpOnly Strict cookie');
  assert.equal(await page.evaluate(`document.cookie.includes(${js(privateCookie.name)})`), false, 'JavaScript cannot read the private session cookie');
  const initial = await stateOf(page);
  assert.equal(initial.residents.length, 3);
  assert.equal(initial.conversations.length, 0, 'Use a newly launched throwaway preview');
  assert.equal(initial.attempts.length, 0);
  assert.equal(initial.unread_count, 0);
  assert.equal(await page.evaluate(`document.body.textContent.includes('No LLM or push delivery runs here')`), true);
  const permission = await page.evaluate(`typeof Notification === 'undefined' ? 'unsupported' : Notification.permission`);
  checks.push('private login and HttpOnly preview cookie', 'fresh fixture-only page with three residents');

  const people = initial.residents;
  const chooseParticipants = async ids => {
    await page.evaluate(`(() => { const selected = ${js(ids)}; for (const input of document.querySelectorAll('#thread-participants input')) input.checked = selected.includes(input.value); document.getElementById('thread-participants').dispatchEvent(new Event('change', {bubbles:true})); })()`);
  };
  const openByTitle = async title => {
    await page.evaluate(`(() => { const button = [...document.querySelectorAll('#threads .thread-item')].find(item => item.querySelector('.thread-title > span')?.textContent === ${js(title)}); if (!button) throw new Error('Conversation control missing'); button.scrollIntoView({block:'center'}); button.click(); })()`);
    await page.until(`document.getElementById('selected-title').textContent === ${js(title)} && !document.getElementById('thread-view').classList.contains('hidden')`, 'open selected conversation');
  };
  const createHumanThread = async (title, content, participants) => {
    await page.click('#new-thread');
    await chooseParticipants(participants);
    await page.fill('thread-title', title);
    await page.fill('thread-opening', content);
    await page.click('#new-thread-form button[type=submit]');
    await page.until(`document.getElementById('selected-title').textContent === ${js(title)} && document.getElementById('messages').textContent.includes(${js(content)})`, 'human thread submitted through the form');
    await page.until(`!document.querySelector('#new-thread-form button[type=submit]').disabled`, 'human submission completed');
    return (await stateOf(page)).conversations.find(thread => thread.title === title);
  };

  const privateThread = await createHumanThread('Browser private conversation', 'A human starts this private fixture conversation.', [people[0].id]);
  assert.equal(privateThread.participants.length, 2);
  assert.equal(privateThread.created_by, 'operator');
  assert.equal((await stateOf(page)).attempts.length, 0, 'Sending only offers a reply; it does not run a model');
  await page.evaluate(`(() => { const row = [...document.querySelectorAll('#offers .offer')].find(item => item.textContent.includes('Browser private conversation')); if (!row) throw new Error('Scripted reply control missing'); row.querySelector('button').click(); })()`);
  await page.until(`document.getElementById('messages').textContent.includes('Scripted fixture reply from')`, 'manual scripted reply added');
  await page.until(`document.getElementById('status').textContent.includes('scripted reply')`, 'scripted reply completed');
  const replied = await stateOf(page);
  assert.equal(replied.attempts.length, 1);
  assert.equal(dataOf(replied.attempts[0]).actor_id, people[0].id);
  assert.equal(dataOf(replied.attempts[0]).status, 'completed');
  await page.evaluate(`document.getElementById('messages').scrollIntoView({block:'center'}); document.getElementById('messages').dispatchEvent(new Event('scroll'));`);
  await page.until(stateExpression + '.then(state => state.unread_count === 0)', 'displayed scripted reply acknowledged');
  checks.push('human private thread and explicit scripted reply without automatic generation');

  const savedDraft = 'Keep this private-thread draft while opening another conversation.';
  await page.fill('message', savedDraft);
  const groupThread = await createHumanThread('Browser group conversation', 'A human chooses two distinct fixture residents.', [people[0].id, people[1].id]);
  assert.equal(groupThread.participants.length, 3);
  assert.equal(groupThread.group, true);
  assert.equal((await stateOf(page)).attempts.length, 1);
  await openByTitle(privateThread.title);
  assert.equal(await page.evaluate(`document.getElementById('message').value`), savedDraft);
  checks.push('human group thread with two fixtures', 'draft preservation across New Thread and thread selection');
  await page.screenshot('chat-desktop');

  await page.click('#new-thread');
  await page.fill('contact-title', 'Browser resident opening');
  await page.fill('contact-message', 'A resident chooses to start a new topic; no push is delivered.');
  await page.evaluate(`(() => { const resident = document.getElementById('contact-resident'); resident.value = ${js(people[0].id)}; resident.dispatchEvent(new Event('change')); for (const input of document.querySelectorAll('#contact-participants input, #contact-mentions input')) input.checked = false; })()`);
  const beforeContact = await stateOf(page);
  // Navigate while admission is pending to keep the new thread deliberately unseen.
  await page.evaluate(`document.querySelector('#resident-form button[type=submit]').click(); document.getElementById('new-thread').click();`);
  await page.until(stateExpression + `.then(state => state.conversations.some(thread => thread.title === 'Browser resident opening') && state.unread_count === 1)`, 'unseen resident initiation retained as unread');
  await page.until(`document.getElementById('unread-filter').textContent === 'Unread threads (1)'`, 'unread thread control updated');
  const contacted = await stateOf(page);
  const residentThread = contacted.conversations.find(thread => thread.title === 'Browser resident opening');
  assert.equal(residentThread.created_by, people[0].id);
  assert.equal(residentThread.unread, 1);
  assert.equal(contacted.attempts.length, beforeContact.attempts.length, 'Initiation commits a message without an automatic generation');
  assert.equal(contacted.notifications.length, beforeContact.notifications.length + 1);
  assert(contacted.notifications.every(record => dataOf(record).status === 'pending'), 'Only saved notification intents exist');
  assert.equal(await page.evaluate(`document.getElementById('new-thread-form').classList.contains('hidden')`), false);
  await page.screenshot('unread-desktop');
  await page.click('#unread-filter');
  assert.equal(await page.evaluate(`document.querySelectorAll('#threads .thread-item').length`), 1);
  await openByTitle(residentThread.title);
  await page.evaluate(`document.getElementById('messages').scrollIntoView({block:'center'}); document.getElementById('messages').dispatchEvent(new Event('scroll'));`);
  await page.until(stateExpression + '.then(state => state.unread_count === 0)', 'foreground displayed snapshot clears unread');
  await page.until(`document.getElementById('unread-filter').textContent === 'Unread threads (0)'`, 'read state reflected by the UI');
  await page.click('#unread-filter');
  assert.equal(await page.evaluate(`typeof Notification === 'undefined' ? 'unsupported' : Notification.permission`), permission);
  assert.equal(await page.evaluate(`'serviceWorker' in navigator ? navigator.serviceWorker.getRegistrations().then(items => items.length) : 0`), 0);
  checks.push('resident-origin thread and durable unread indicator', 'foreground snapshot reading clears unread', 'saved notification intent without push enrollment or delivery');

  const beforeSchedule = await stateOf(page);
  await page.evaluate(`document.getElementById('scheduling-demo').closest('details').open = true;`);
  await page.click('#scheduling-demo');
  await page.until(stateExpression + `.then(state => state.attempts.length === ${beforeSchedule.attempts.length + 3} && state.attempts.every(record => record.data.status === 'completed'))`, 'three synthetic scheduled generations completed');
  await page.until(`document.getElementById('status').textContent.includes('Scheduling demo completed')`, 'scheduling demonstration finished');
  const scheduled = await stateOf(page);
  const timeline = scheduled.provider_timeline.slice(beforeSchedule.provider_timeline.length);
  assert.equal(timeline.length, 3);
  assert.deepEqual(timeline.map(item => item.actor_id).sort(), people.map(person => person.id).sort());
  assert.deepEqual(timeline.map(item => item.end - item.start).sort((a,b) => a-b), [3, 6, 12]);
  assert.equal(scheduled.conversations.length, beforeSchedule.conversations.length);
  checks.push('three-resident synthetic provider-time scheduling demonstration');

  await page.fill('world-name', 'Browser shared river stone');
  await page.evaluate(`document.getElementById('world-owner').value = ${js(people[1].id)}; document.getElementById('world-visibility').value = 'shared';`);
  await page.click('#world-form button[type=submit]');
  await page.until(`document.getElementById('world').textContent.includes('Browser shared river stone')`, 'shared world object rendered');
  const withWorld = await stateOf(page);
  const stone = withWorld.world.find(record => dataOf(record).name === 'Browser shared river stone');
  assert.equal(dataOf(stone).owner, people[1].id);
  assert.equal(dataOf(stone).visibility, 'shared');
  checks.push('shared object creation with the selected fixture owner');

  await page.send('Emulation.setDeviceMetricsOverride', {width:390, height:844, deviceScaleFactor:1, mobile:true});
  await page.evaluate(`document.getElementById('new-thread').scrollIntoView({block:'center'});`);
  await page.until(`document.documentElement.scrollWidth <= innerWidth`, 'mobile page fits the viewport');
  const controls = await page.evaluate(`['new-thread', 'unread-filter'].map(id => { const button = document.getElementById(id), bounds = button.getBoundingClientRect(); return {id, tag:button.tagName, text:button.textContent.trim(), left:bounds.left, right:bounds.right, width:innerWidth, height:bounds.height}; })`);
  for (const control of controls) {
    assert.equal(control.tag, 'BUTTON');
    assert(control.text && control.left >= 0 && control.right <= control.width && control.height >= 36, 'Mobile Chat controls stay accessible');
  }
  const mobileThread = await createHumanThread('Browser mobile thread', 'A mobile user can choose a fixture and send.', [people[2].id]);
  await page.fill('message', 'A second human message from the mobile composer.');
  await page.click('#message-form button[type=submit]');
  await page.until(`document.getElementById('messages').textContent.includes('A second human message from the mobile composer.')`, 'mobile Send message works');
  await page.evaluate(`document.getElementById('messages').scrollIntoView({block:'center'});`);
  await page.screenshot('chat-mobile');
  await openByTitle(privateThread.title);
  assert.equal(await page.evaluate(`document.getElementById('message').value`), savedDraft);
  assert.equal(await page.evaluate(`document.documentElement.scrollWidth <= innerWidth`), true);
  checks.push('390px mobile Chat controls and both submission forms', 'original draft survives desktop/mobile navigation');

  const finalState = await stateOf(page);
  assert.equal(finalState.conversations.find(thread => thread.id === mobileThread.id).last_sequence, 2);
  assert.deepEqual(errors, [], 'No uncaught page exceptions');
  assert.deepEqual(blockedRequests, [], 'The application attempted no traffic beyond its disposable preview origin');
  assert(requests.every(request => !/push|subscribe|service-worker|sw\.js/.test(request.path)), 'No push enrollment/delivery paths were called');
  summary = {
    passed:true, browser:'Chromium', fixture_only:true, checks,
    counts:{residents:finalState.residents.length, conversations:finalState.conversations.length,
      attempts:finalState.attempts.length, world_objects:finalState.world.length,
      notification_intents:finalState.notifications.length, unread_threads:finalState.unread_count},
    session:{name:privateCookie.name, http_only:privateCookie.httpOnly, same_site:privateCookie.sameSite},
    synthetic_demo:timeline.map(item => ({actor_id:item.actor_id, seconds:item.end-item.start})),
    uncaught_errors:errors, blocked_requests:blockedRequests,
    limits:['Scripted fixture responses and synthetic provider durations only.',
      'No real-model calls, Ember state, household migration, production adoption, or device push delivery.',
      'Desktop and 390px Chromium paths; real devices, OS background policies, restart, and offline delivery remain separate checks.'],
    artifacts:reportRoot,
  };
  await writeFile(join(reportRoot, 'report.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify(summary));
} catch (error) {
  const failure = {passed:false, message:redact(error.stack || error), checks, uncaught_errors:errors, blocked_requests:blockedRequests, artifacts:reportRoot};
  if (page) await page.screenshot('failure').catch(() => {});
  await writeFile(join(reportRoot, 'failure.json'), JSON.stringify(failure, null, 2) + '\n');
  console.error(JSON.stringify(failure));
  process.exitCode = 1;
} finally {
  page?.socket.close();
  browser.kill('SIGTERM');
  await sleep(250);
  if (browser.exitCode === null) browser.kill('SIGKILL');
  await rm(profile, {recursive:true, force:true});
}
