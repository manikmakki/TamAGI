// Real Chromium checks against scripts/navigation-fixture.py; no live resident.
import {spawn} from 'node:child_process';
import {mkdtemp, rm, writeFile, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';

const base=process.env.TAMAGI_FIXTURE_URL||'http://127.0.0.1:18742';
const debuggerPort=Number(process.env.TAMAGI_BROWSER_PORT||18743);
const chrome=process.env.TAMAGI_CHROME||'/root/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome';
const reportRoot=process.env.TAMAGI_BROWSER_REPORT||'.platform/navigation-browser';
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const profile=await mkdtemp(join(tmpdir(),'tamagi-browser-'));
const browser=spawn(chrome,['--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-first-run','--host-resolver-rules=MAP fonts.googleapis.com 127.0.0.1, MAP fonts.gstatic.com 127.0.0.1',`--remote-debugging-port=${debuggerPort}`,`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});
const errors=[];
class Page {
  constructor(socket,target){this.socket=socket;this.target=target;this.next=1;this.pending=new Map();socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.id){const item=this.pending.get(value.id);if(item){this.pending.delete(value.id);clearTimeout(item.timer);value.error?item.reject(new Error(JSON.stringify(value.error))):item.resolve(value.result);}}else if(value.method==='Fetch.requestPaused'){const event=value.params;if(this.dropAck&&event.request.method==='POST'&&event.responseStatusCode===202){this.dropAck=false;this.dropped=(this.dropped||0)+1;this.send('Fetch.failRequest',{requestId:event.requestId,errorReason:'ConnectionClosed'}).catch(e=>errors.push(e.message));}else this.send('Fetch.continueRequest',{requestId:event.requestId}).catch(e=>errors.push(e.message));}else if(value.method==='Runtime.exceptionThrown')errors.push(value.params.exceptionDetails.exception?.description||value.params.exceptionDetails.text);};}
  send(method,params={}){return new Promise((resolve,reject)=>{const id=this.next++,timer=setTimeout(()=>{this.pending.delete(id);reject(new Error('CDP timeout: '+method));},20000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params}));});}
  async evaluate(expression){const result=await this.send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw new Error(result.exceptionDetails.exception?.description||result.exceptionDetails.text);return result.result?.value;}
  async until(expression,message,timeout=15000){const end=Date.now()+timeout;while(Date.now()<end){try{if(await this.evaluate(expression))return;}catch{}await sleep(100);}throw new Error('Browser condition: '+message);}
  async screenshot(name){const result=await this.send('Page.captureScreenshot',{format:'png'});await writeFile(join(reportRoot,name+'.png'),Buffer.from(result.data,'base64'));}
}
async function open(url=base){const target=await fetch(`http://127.0.0.1:${debuggerPort}/json/new?${encodeURIComponent(url)}`,{method:'PUT'}).then(r=>r.json());const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});const page=new Page(socket,target.id);await page.send('Runtime.enable');await page.send('Page.enable');await page.send('Page.navigate',{url});await page.until('location.href.startsWith('+JSON.stringify(base)+')','target navigation');return page;}
const js=value=>JSON.stringify(value);
const api=(path,options={})=>`fetch('/api'+${js(path)},${js(options)}).then(async r=>{const data=await r.json();if(!r.ok)throw new Error(JSON.stringify(data));return data;})`;
let page;
try {
  await mkdir(reportRoot,{recursive:true});
  for(let i=0;i<80;i++){try{await fetch(`http://127.0.0.1:${debuggerPort}/json/version`);break;}catch{await sleep(100);}}
  page=await open(base+'/login');
  await page.until('document.readyState === "complete"','login loaded');
  const auth=await page.evaluate(api('/auth/status'));
  await page.evaluate(api(auth.has_password?'/auth/login':'/auth/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:'disposable navigation fixture password'})}));
  await page.send('Page.navigate',{url:base+'/'});
  await page.send('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
  await page.until('!!window.TamagiShell && document.body.dataset.view === "chat"','shared shell initialized');
  await page.until('!document.getElementById("onboarding").classList.contains("hidden") === false','fixture onboarding hidden');
  await page.evaluate('openDrawer()');
  assert.deepEqual(await page.evaluate('[...document.querySelectorAll(".app-menu-item > span:nth-child(2)")].map(a=>a.textContent.trim())'),['Chat','Map','Journal','Inventory','Companions','Archive','Settings']);
  await page.send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape'});
  await page.until('document.getElementById("drawer").inert','Escape closes drawer');
  const initial=await page.evaluate(api('/navigation-fixture'));
  await page.evaluate(`document.getElementById('input').value='An unsent draft'; const transfer=new DataTransfer();transfer.items.add(new File(['draft'],'draft.txt',{type:'text/plain'}));document.getElementById('file-input').files=transfer.files;handleFileSelect(document.getElementById('file-input'));`);
  await page.evaluate(`document.getElementById('chat-messages').style.scrollBehavior='auto';for(let n=0;n<30;n++)addMsg('assistant','A fixture history entry '+n);document.getElementById('chat-messages').scrollTop=50;document.getElementById('chat-messages').style.scrollBehavior=''`);
  await page.until('Math.abs(document.getElementById("chat-messages").scrollTop-50)<2','fixture history positioned');
  for(const view of ['map','journal','inventory','companions','archive','settings']){
    await page.evaluate(`openDrawer();document.querySelector('.app-menu-item[data-view="${view}"]').click()`);
    await page.until(`document.body.dataset.view===${js(view)}`,'drawer '+view);
  }
  await page.evaluate(`TamagiShell.navigate('/settings/connections')`);
  await page.until(`document.getElementById('settings-connections').contentDocument?.getElementById('sec-name')`,'connection settings adapter');
  await page.evaluate(`TamagiShell.navigate('/settings/appearance')`);
  await page.until(`document.getElementById('settings-appearance').contentDocument?.querySelectorAll('#avatar-buttons button').length>=3`,'appearance choices retained');
  await page.evaluate(`TamagiShell.navigate('/')`);
  assert.equal(await page.evaluate('document.getElementById("input").value'),'An unsent draft');
  assert.equal(await page.evaluate('pendingFile.name'),'draft.txt');
  await page.until('Math.abs(document.getElementById("chat-messages").scrollTop-50)<2','chat scroll position restored');
  const browsed=await page.evaluate(api('/navigation-fixture'));
  assert.equal(browsed.input_epoch,initial.input_epoch);assert.equal(browsed.model_calls,initial.model_calls);
  await page.evaluate(`TamagiShell.navigate('/inventory')`);
  await page.until(`document.querySelectorAll('.creation-card').length===2`,'unregistered workshop files');
  await page.evaluate(`[...document.querySelectorAll('.creation-card')].find(c=>c.querySelector('h2').textContent==='preview.html').querySelector('button').click()`);
  await page.until(`document.getElementById('preview-dialog').open && document.querySelector('#preview-content iframe')`,'static HTML preview');
  await sleep(300);assert.equal(await page.evaluate('window.__previewEscaped === true'),false);
  await page.evaluate(`document.getElementById('preview-dialog').close()`);
  await page.screenshot('inventory-desktop');
  await page.evaluate(`TamagiShell.navigate('/journal')`);await page.screenshot('journal-desktop');
  await page.send('Network.enable');
  await page.send('Network.setBypassServiceWorker',{bypass:true});
  page.dropAck=true;
  await page.send('Fetch.enable',{patterns:[{urlPattern:base+'/api/chat/requests',requestStage:'Response'}]});
  await page.evaluate(`TamagiShell.navigate('/');clearAttachment();document.getElementById('input').value='A simple conversation';sendMessage()`);
  await page.until(`!!convId && !loading && [...document.querySelectorAll('.msg.assistant .bubble')].some(n=>n.textContent.includes('Navigation fixture finished.'))`,'first delivered reply');
  assert.equal(page.dropped,1,'acceptance acknowledgment was deliberately lost');
  await page.send('Fetch.disable');
  await page.send('Network.setBypassServiceWorker',{bypass:false});
  const simpleRequests=await page.evaluate(api('/chat/requests'));assert.equal(simpleRequests.requests.length,1,'lost acknowledgment did not submit a second job');
  const earlier=await page.evaluate('convId');
  await page.evaluate(`newConversation();document.getElementById('input').value='Slow work while I explore';sendMessage()`);
  await page.until(`!!convId && loading`,'accepted slow request');
  const slow=await page.evaluate('[...TamagiShell.works.values()].filter(w=>w.conversation_id===convId).at(-1)');
  await page.until(api('/navigation-fixture')+'.then(v=>v.tool_calls===1)','in-flight tool started');
  await page.evaluate(`TamagiShell.navigate('/journal')`);
  assert.equal(await page.evaluate('document.getElementById("shell-stop").hidden'),false);
  await page.evaluate(`openConversation(${js(earlier)})`);
  await page.until(`convId===${js(earlier)} && document.body.dataset.view==='chat'`,'selected earlier conversation');
  await page.until(api('/chat/requests/'+slow.work_id)+`.then(v=>v.work.status==='completed')`,'background result retained');
  await sleep(500);assert.equal(await page.evaluate('convId'),earlier);
  assert.equal(await page.evaluate(`document.getElementById('chat-messages').textContent.includes('Slow work while I explore')`),false);
  await page.evaluate(`openConversation(${js(slow.conversation_id)})`);
  await page.until(`document.getElementById('chat-messages').textContent.includes('Slow work while I explore')`,'correct conversation retrieves result');
  await page.evaluate(`newConversation();document.getElementById('input').value='Quiet long work';sendMessage()`);
  await page.until('quietActive && loading','quiet continuation active');
  await page.until(api('/navigation-fixture')+'.then(v=>v.tool_calls===2)','quiet tool started');
  const observationBefore=await page.evaluate(api('/navigation-fixture'));
  await page.evaluate(api('/memory/recall',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:'a leaf',limit:5})}));
  const observationAfter=await page.evaluate(api('/navigation-fixture'));assert.equal(observationAfter.input_epoch,observationBefore.input_epoch);
  assert.equal((await page.evaluate(api('/chat/requests'))).requests.find(w=>w.status==='running')?.kind,'chat','memory search did not interrupt quiet work');
  const quiet=await page.evaluate('[...TamagiShell.works.values()].filter(w=>w.conversation_id===convId).at(-1)');
  await page.evaluate(`TamagiShell.navigate('/inventory');TamagiShell.navigate('/');document.getElementById('input').value='A correction';sendMessage()`);
  await page.until(api('/chat/requests/'+quiet.work_id)+`.then(v=>v.work.status==='interrupted')`,'actual input interrupts quiet work');
  await page.until('!loading','new actual input finishes');
  await page.evaluate(`newConversation();document.getElementById('input').value='Slow disconnect test';sendMessage()`);
  await page.until('!!convId && loading','disconnect request accepted');
  const detached=await page.evaluate('[...TamagiShell.works.values()].filter(w=>w.conversation_id===convId).at(-1)');
  await page.until(api('/navigation-fixture')+'.then(v=>v.tool_calls===3)','disconnect tool started');
  const count=await page.evaluate(api('/navigation-fixture'));
  await page.send('Page.close');page.socket.close();
  page=await open(base+'/');
  await page.until('!!window.TamagiShell && !!convId','new browser view restores conversation');
  await page.until(api('/chat/requests/'+detached.work_id)+`.then(v=>v.work.status==='completed')`,'closed page did not cancel work');
  await page.evaluate(`openConversation(${js(detached.conversation_id)})`);
  await page.until(`document.querySelectorAll('.msg.assistant .bubble').length===1`,'one retained final response');
  const after=await page.evaluate(api('/navigation-fixture'));assert.equal(after.tool_calls,count.tool_calls);
  await page.send('Page.reload');
  await page.until('!!window.TamagiShell && !!convId && !loading','reload restores completed work');
  await page.until(`document.querySelectorAll('.msg.assistant .bubble').length===1`,'reload does not duplicate response');
  const reloaded=await page.evaluate(api('/navigation-fixture'));assert.equal(reloaded.model_calls,after.model_calls);assert.equal(reloaded.tool_calls,after.tool_calls);
  await page.evaluate(`TamagiShell.navigate('/map');TamagiShell.navigate('/inventory');history.back()`);
  await page.until(`document.body.dataset.view==='map'`,'browser Back');
  await page.evaluate('history.forward()');await page.until(`document.body.dataset.view==='inventory'`,'browser Forward');
  const queuedFirst=await page.evaluate(api('/chat/requests',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:'Slow queue primary',request_id:'queue-first'})}));
  const queuedSecond=await page.evaluate(api('/chat/requests',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:'Queue followup',request_id:'queue-second',conversation_id:queuedFirst.conversation_id})}));
  const admittedHistory=await page.evaluate(api('/conversations/'+queuedFirst.conversation_id));assert.deepEqual(admittedHistory.messages.filter(m=>m.role==='user').map(m=>m.content),['Slow queue primary','Queue followup']);
  await page.until(api('/chat/requests/'+queuedSecond.work_id)+`.then(v=>v.work.status==='completed')`,'queued work finishes');
  const queuedHistory=await page.evaluate(api('/conversations/'+queuedFirst.conversation_id));assert.deepEqual(queuedHistory.messages.map(m=>m.role),['user','assistant','user','assistant']);
  const queuedEvidence=await page.evaluate(api('/navigation-fixture'));assert(queuedEvidence.prompts['queue-first'].every(t=>t==='Slow queue primary'));
  await page.send('Page.navigate',{url:base+'/home#growth'});
  await page.until('!!window.TamagiShell && document.getElementById("growth-dialog").open','legacy growth deep link');
  assert.equal(await page.evaluate('document.getElementById("growth-summary").textContent.includes("undefined")'),false);
  await page.evaluate('document.getElementById("growth-dialog").close()');
  await page.send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await page.evaluate(`TamagiShell.navigate('/journal')`);await page.screenshot('journal-mobile');
  assert.equal(await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),true);
  await page.evaluate(`TamagiShell.navigate('/');openDrawer()`);await sleep(350);assert(Math.abs(await page.evaluate('document.getElementById("drawer").getBoundingClientRect().left'))<2);await page.screenshot('drawer-mobile');
  assert.deepEqual(errors,[],'No uncaught browser exceptions');
  await page.evaluate(`closeDrawer();TamagiShell.navigate('/companions');document.getElementById('friend-name').value='Juniper';document.getElementById('friend-seed').value='A patient fixture companion';document.getElementById('companion-form').requestSubmit()`);
  await page.until(`document.querySelectorAll('#companions button').length>0`,'companion roster retained');
  await page.evaluate(`document.querySelector('#companions button').click()`);
  await page.until(`!document.getElementById('consultation').hidden`,'companion conversation selected');
  const growthBefore=await page.evaluate(api('/runtime/growth'));
  await page.evaluate(`document.getElementById('friend-message').value='Hello companion';document.getElementById('consult-form').requestSubmit()`);
  await page.until(`[...TamagiShell.works.values()].some(w=>w.kind==='companion'&&['queued','running'].includes(w.status))`,'companion admitted');
  const companionWork=await page.evaluate(`[...TamagiShell.works.values()].filter(w=>w.kind==='companion').at(-1)`);
  await page.evaluate(`TamagiShell.navigate('/map')`);
  await page.until(api('/chat/requests/'+companionWork.work_id)+`.then(v=>v.work.status==='completed')`,'companion continues in another view');
  await page.until(`!document.getElementById('shell-return').hidden && document.getElementById('shell-return').textContent==='Open companion'`,'companion completion is reachable from shell');
  await page.evaluate(`document.getElementById('shell-return').click()`);
  assert.equal(await page.evaluate(`document.getElementById('friend-reply').textContent.includes('Navigation fixture finished.')`),true);
  assert.equal((await page.evaluate(api('/runtime/growth'))).total,growthBefore.total,'operator conversation does not level the resident');
  await page.evaluate(`newConversation();document.getElementById('input').value='Approval fixture';sendMessage()`);
  await page.until(`document.getElementById('shell-work-label').textContent==='Approval needed'`,'guarded approval retained');
  const approvalWork=await page.evaluate(`[...TamagiShell.works.values()].find(w=>w.approval)`);
  await page.evaluate(`TamagiShell.navigate('/journal');document.getElementById('shell-work-label').click()`);
  await page.until(`document.getElementById('work-approval-dialog').open`,'approval accessible from another view');
  await page.evaluate(`[...document.querySelectorAll('#work-approval-content button')].find(b=>b.textContent==='Allow once').click()`);
  await page.until(api('/chat/requests/'+approvalWork.work_id)+`.then(v=>v.work.status==='completed')`,'approved work completes');
  const approvalFinal=await page.evaluate(api('/chat/requests/'+approvalWork.work_id));assert(approvalFinal.work.approval===null);
  const finalCounts=await page.evaluate(api('/navigation-fixture'));
  assert.deepEqual(errors,[],'No uncaught browser exceptions after companion flow');
  const report={passed:true,browser:'Chromium',fixture:true,checks:['lost acknowledgment deduplication','queued input persistence and response ordering','drawer and keyboard','all views and compatibility panels','draft, attachment and scroll preservation','browsing and memory search do not interrupt or infer','folder inventory and isolated HTML preview','late result stays in original thread','quiet navigation and actual-input interruption','closed-page continuation and result recovery','reload deduplication','Back/Forward','mobile layout','guarded approval from another view','companion invocation survives view change without resident Growth'],counts:finalCounts,uncaught_errors:errors};
  await writeFile(join(reportRoot,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
} catch(error) {
  if(page){await page.screenshot('failure').catch(()=>{});await writeFile(join(reportRoot,'failure.json'),JSON.stringify({message:error.stack,uncaught_errors:errors},null,2));}
  throw error;
} finally {
  page?.socket.close();browser.kill('SIGTERM');await sleep(300);await rm(profile,{recursive:true,force:true});
}
