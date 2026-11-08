import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import assert from 'node:assert/strict';
const base='http://127.0.0.1:18846',cdp='http://127.0.0.1:18847',root=await mkdtemp('/tmp/tamagi-native-actual-browser-'),profile=await mkdtemp('/tmp/tamagi-native-actual-profile-');
const password=await readFile(process.env.TAMAGI_BROWSER_CREDENTIAL||'/tmp/tamagi-native-browser-credential','utf8');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms)),js=JSON.stringify,errors=[],checks=[];
const browser=spawn('/root/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome',['--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-first-run','--disable-background-networking','--disable-component-update','--disable-sync','--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1','--remote-debugging-port=18847','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
class Page{
 constructor(socket){this.socket=socket;this.next=0;this.pending=new Map();socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.id){const p=this.pending.get(value.id);if(p){this.pending.delete(value.id);clearTimeout(p.timer);value.error?p.reject(new Error(JSON.stringify(value.error))):p.resolve(value.result);}}else if(value.method==='Runtime.exceptionThrown')errors.push(value.params.exceptionDetails.exception?.description||value.params.exceptionDetails.text);};}
 send(method,params={}){return new Promise((resolve,reject)=>{const id=++this.next,timer=setTimeout(()=>reject(new Error('CDP timeout '+method)),20000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params}));});}
 async evaluate(expression){const value=await this.send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(value.exceptionDetails)throw new Error(value.exceptionDetails.exception?.description||value.exceptionDetails.text);return value.result?.value;}
 async until(expression,text){for(let n=0;n<250;n++){try{if(await this.evaluate(expression))return;}catch{}await sleep(70);}throw new Error(text);}
 async shot(name){const value=await this.send('Page.captureScreenshot',{format:'png'});await writeFile(root+'/'+name+'.png',Buffer.from(value.data,'base64'));}
}
let page;
try{
 for(let n=0;n<100;n++){try{if((await fetch(cdp+'/json/version')).ok)break;}catch{}await sleep(70);}
 const target=await fetch(cdp+'/json/new?about%3Ablank',{method:'PUT'}).then(r=>r.json()),socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});page=new Page(socket);
 await page.send('Runtime.enable');await page.send('Page.enable');await page.send('Network.enable');await page.send('Emulation.setDeviceMetricsOverride',{width:1280,height:900,mobile:false,deviceScaleFactor:1});await page.send('Page.navigate',{url:base+'/login'});await page.send('Page.bringToFront');
 await page.until(`!!document.getElementById('password')`,'login loaded');await page.evaluate(`document.getElementById('password').value=${js(password)};document.getElementById('submit-btn').click();`);
 await page.until(`!!window.TamagiHousehold&&document.querySelectorAll('.hh-wheel-person').length===2&&document.getElementById('a-name').textContent==='Cedar'`,'actual native adapter starts');
 await page.until(`document.getElementById('onboarding').classList.contains('hidden')`,'fresh named households open to visible chat, without an onboarding overlay');checks.push('chat controls are visible and unobstructed by onboarding');
 const cookies=await page.send('Network.getCookies',{urls:[base]});assert.equal(cookies.cookies.find(c=>c.name.startsWith('tamagi_household_')).httpOnly,true);checks.push('normal login with private household session cookie');
 const actors=await page.evaluate(`TAMAGI_HOUSEHOLD_BOOTSTRAP.residents.map(r=>({id:r.id,name:r.name}))`),first=actors[0].id,second=actors[1].id;
 await page.evaluate(`document.getElementById('input').value='Cedar draft';document.getElementById('input').dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('household-character-wheel').click();`);await page.send('Input.dispatchKeyEvent',{type:'keyDown',key:'End',code:'End'});assert.equal(await page.evaluate('document.activeElement.dataset.residentId'),second);
 await page.evaluate(`document.querySelector('.hh-wheel-person[data-resident-id=${js(second)}]').click()`);await page.until(`document.getElementById('a-name').textContent==='Willow'`,'actual personal state switches');assert.equal(await page.evaluate(`document.getElementById('input').value`),'');
 await page.evaluate(`document.getElementById('input').value='Willow draft';document.getElementById('input').dispatchEvent(new Event('input',{bubbles:true}));TamagiHousehold.selectResident(${js(first)})`);assert.equal(await page.evaluate(`document.getElementById('input').value`),'Cedar draft');checks.push('keyboard character wheel retains main rig and distinct resident drafts');
 await page.evaluate(`document.getElementById('household-participants').click();for(const field of document.querySelectorAll('#household-participant-options input'))field.checked=true;document.querySelector('#household-participants-dialog>button:last-child').click();document.getElementById('input').value='@Wi';document.getElementById('input').setSelectionRange(3,3);document.getElementById('input').dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('input').focus();`);
 await page.until(`!document.getElementById('household-mention-list').hidden`,'stable participant suggestions');await page.send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter'});assert.equal(await page.evaluate(`document.getElementById('input').value`),'@Willow ');assert.equal(await page.evaluate(`convId`),null);checks.push('mention completion Enter completes without sending');
 const message=Array.from({length:28},(_,i)=>`Context line ${i+1} for both residents.`).join('\n');await page.evaluate(`document.getElementById('input').value=${js(message)};document.getElementById('send-btn').click();`);
 await page.until(`!!convId&&loading&&document.querySelector('.msg.user')`,'real work admitted');await page.evaluate(`document.getElementById('input').value='Draft while both residents work';document.getElementById('input').dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('chat-messages').style.scrollBehavior='auto';document.getElementById('chat-messages').scrollTop=0;`);
 await page.until(`!!document.querySelector('.hh-live-progress .interim-entry strong')`,'actual interim Markdown appears live');
 assert.equal(await page.evaluate(`document.querySelector('.hh-live-progress').dataset.authorId`),first);
 assert.equal(await page.evaluate(`document.querySelector('.hh-live-progress .hh-progress-speaker').textContent`),'Cedar');
 assert.equal(await page.evaluate(`!!window.reasoningUnsafe||!!document.querySelector('.interim-entry img')`),false);
 await page.until(`document.querySelectorAll('.msg.assistant .hh-speaker').length===1`,'first canonical child result');assert.equal(await page.evaluate(`loading`),true,'first child does not finish shared exchange');assert.equal(await page.evaluate(`document.getElementById('input').value`),'Draft while both residents work');assert.equal(await page.evaluate(`document.getElementById('chat-messages').scrollTop`),0);assert.equal(await page.evaluate(`TamagiHousehold.selectedResidentId()`),first);checks.push('thread_changed renders first real child without ending shared work or moving draft, scroll, or resident');
 assert.equal(await page.evaluate(`document.querySelectorAll('.history-progress .interim-group.collapsed').length`),1);
 assert.equal(await page.evaluate(`document.querySelector('.history-progress').nextElementSibling.dataset.authorId`),first);
 assert.equal(await page.evaluate(`document.querySelector('.history-progress .interim-entry strong').textContent`),'Checking the workspace');
 assert.equal(await page.evaluate(`document.querySelectorAll('.hh-live-progress[data-author-id=${js(first)}]').length`),0,'completed resident reasoning is not replayed at the bottom');
 await page.until(`!!document.querySelector('.hh-live-progress[data-author-id=${js(second)}] .interim-entry')`,'second resident has its own live reasoning');
 assert.equal(await page.evaluate(`document.querySelector('.hh-live-progress .hh-progress-speaker').textContent`),'Willow');
 assert.equal(await page.evaluate(`document.querySelector('.hh-live-progress').querySelectorAll('.interim-entry').length`),1,'second resident does not inherit Cedar steps');
 await page.evaluate(`TamagiShell.refreshHistory()`);
 assert.equal(await page.evaluate(`document.querySelectorAll('.history-progress').length`),1);
 assert.equal(await page.evaluate(`document.querySelectorAll('.hh-live-progress .interim-entry').length`),1,'refresh during Willow work does not replay completed Cedar reasoning');
 await page.until(`document.querySelectorAll('.msg.assistant .hh-speaker').length===2&&!loading`,'final shared exchange ends');assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#chat-messages .hh-speaker')].map(e=>e.textContent)`),['Tester','Cedar','Willow']);assert.equal(await page.evaluate(`document.getElementById('input').value`),'Draft while both residents work');
 await page.until(`document.querySelectorAll('.history-progress').length===2&&document.querySelectorAll('.hh-live-progress').length===0`,'both resident turns have saved reasoning only');
 assert.equal(await page.evaluate(`document.querySelectorAll('details.history-progress').length`),0);
 assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('.history-progress')].map(e=>({author:e.dataset.authorId,next:e.nextElementSibling.dataset.authorId,steps:e.querySelectorAll('.interim-entry').length}))`),[{author:first,next:first,steps:1},{author:second,next:second,steps:1}]);
 await page.evaluate(`document.querySelector('.history-progress .interim-toggle').click()`);
 assert.equal(await page.evaluate(`document.querySelector('.history-progress .interim-group').classList.contains('collapsed')`),false);
 await page.evaluate(`TamagiShell.refreshHistory()`);
 assert.equal(await page.evaluate(`document.querySelectorAll('.history-progress .interim-entry strong').length`),2,'history refresh preserves Markdown without replay duplication');
 assert.equal(await page.evaluate(`!!window.reasoningUnsafe`),false);
 checks.push('live and saved reasoning share production bubble styling, sanitized Markdown and correct resident attribution','completed resident steps stay above their own message while the next resident works','history refresh retires live reasoning and never duplicates completed steps');
 assert.equal(await page.evaluate(`document.getElementById('chat-messages').scrollTop`),0,'final completion also preserves reading position');
 assert.equal(await page.evaluate(`document.querySelectorAll('.msg.assistant .hh-mention').length`),2);
 await page.evaluate(`for(const group of document.querySelectorAll('.history-progress .interim-group'))group.classList.remove('collapsed');document.getElementById('chat-messages').scrollTop=document.getElementById('chat-messages').scrollHeight`);
 await page.shot('native-reasoning');
 await page.evaluate(`for(const group of document.querySelectorAll('.history-progress .interim-group'))group.classList.add('collapsed')`);
 await page.evaluate(`document.getElementById('chat-messages').scrollTop=document.getElementById('chat-messages').scrollHeight`);
 await page.until(`document.getElementById('household-unread').textContent==='Unread threads (0)'`,'foreground reads clear durable unread');checks.push('canonical attribution and human mention highlights from two actual worker Agents','final root completion preserves a scrolled-up reader','foreground visible snapshots clear durable unread');
 const group=await page.evaluate('convId');
 const gap=`(()=>{const a=document.getElementById('chat-messages');return a.scrollHeight-a.scrollTop-a.clientHeight})()`;
 for(const mobile of [false,true]){
  await page.send('Emulation.setDeviceMetricsOverride',{width:mobile?390:1280,height:mobile?844:900,mobile,deviceScaleFactor:1});
  const before=await page.evaluate(`document.querySelectorAll('.msg.assistant .hh-speaker').length`);
  await page.evaluate(`document.getElementById('input').value='Continue together ${mobile?'mobile':'desktop'}';document.getElementById('send-btn').click()`);
  await page.until(`loading`,'new pinned exchange starts');
  await page.until(`document.querySelectorAll('.msg.assistant .hh-speaker').length===${before+1}`,'first pinned group reply');
  assert.ok(await page.evaluate(gap)<2,'first reply follows bottom');
  await page.until(`!loading&&document.querySelectorAll('.msg.assistant .hh-speaker').length===${before+2}`,'last pinned group reply');
  assert.ok(await page.evaluate(gap)<2,'final reply follows bottom');
 }
 checks.push('desktop and mobile follow each group reply and final completion while pinned to bottom');
 await page.send('Emulation.setDeviceMetricsOverride',{width:1280,height:900,mobile:false,deviceScaleFactor:1});
 await page.evaluate(`(async()=>{document.getElementById('input').value='Group draft survives navigation';await TamagiHousehold.selectResident(${js(second)});window.newConversation();document.getElementById('input').value='Willow private navigation check';await window.sendMessage()})()`);
 const privateWork=await page.evaluate(`({conversation_id:convId})`);
 await page.evaluate(`(async()=>{await TamagiHousehold.selectResident(${js(first)});TamagiShell.navigate('/map')})()`);
 await page.until(`!document.getElementById('shell-return').hidden`,'main interface Open chat is available');
 await page.evaluate(`document.getElementById('shell-return').click()`);
 await page.until(`convId===${js(privateWork.conversation_id)}&&TamagiHousehold.selectedResidentId()===${js(second)}&&document.getElementById('a-name').textContent==='Willow'`,'Open chat aligns thread, resident and avatar');
 await page.until(`!loading`,'private work completes');
 await page.evaluate(`window.openConversation(${js(group)})`);
 assert.equal(await page.evaluate('TamagiHousehold.selectedResidentId()'),second,'group retains an already participating resident');
 assert.equal(await page.evaluate(`document.getElementById('input').value`),'Group draft survives navigation');
 checks.push('main interface Open chat selects the private thread resident; group retains a participating selection and draft');
 await page.evaluate(`document.getElementById('input').value='Browser image to both residents';pendingFile=new File([Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN2kAAAAASUVORK5CYII='),c=>c.charCodeAt(0))],'pixel.png',{type:'image/png'});document.getElementById('send-btn').click()`);
 await page.until(`loading`,'browser image is admitted');
 await page.until(`!loading&&!!document.querySelector('.hh-attachment')`,'image exchange completes');
 await page.until(`document.querySelector('.hh-attachment').complete&&document.querySelector('.hh-attachment').naturalWidth===1`,'canonical attached image renders');
 assert.ok(await page.evaluate(gap)<2,'image and replies remain pinned');
 const latest=await page.evaluate(`fetch('/api/chat/requests').then(r=>r.json()).then(v=>v.requests[0])`);
 assert.equal(latest.resident_id,second,'next submission uses the selected resident');
 assert.equal(latest.status,'completed');assert.equal(latest.responders.length,2);
 checks.push('browser file input reaches both actual Agents, persists and renders from the canonical message');
 await page.evaluate(`(async()=>{const original=window.fetch;let release,hold=true;window.fetch=async function(input,options){if(hold&&String(input)==='/api/conversations/'+${js(group)}){hold=false;await new Promise(resolve=>release=resolve);}return original(input,options)};try{const pending=window.openConversation(${js(group)});window.newConversation();release();await pending;if(convId!==null)throw new Error('Stale navigation replaced New Thread');}finally{window.fetch=original;}await window.openConversation(${js(group)});})()`);
 checks.push('New Thread cancels an older conversation navigation still awaiting its response');
 await page.evaluate(`TamagiShell.navigate('/settings');document.querySelector('#household-color-form input[data-actor-id=${js(second)}]').value='#abcdef';document.querySelector('#household-color-form button').click();`);await page.until(`fetch('/api/household').then(r=>r.json()).then(s=>s.preferences.colors[${js(second)}]==='#abcdef')`,'native durable color save');await page.until(`document.querySelector('.msg[data-author-id=${js(second)}]').style.getPropertyValue('--hh-speaker-color')==='#abcdef'`,'saved color is rendered after asynchronous UI refresh');await page.evaluate(`TamagiShell.navigate('/')`);assert.equal(await page.evaluate(`document.querySelector('.msg[data-author-id=${js(second)}]').style.getPropertyValue('--hh-speaker-color')`),'#abcdef');checks.push('durable user color preference updates native attributed bubbles');
 await page.shot('native-desktop');await page.send('Emulation.setDeviceMetricsOverride',{width:390,height:844,mobile:true,deviceScaleFactor:1});await page.evaluate(`document.getElementById('household-character-wheel').click()`);assert.equal(await page.evaluate(`document.documentElement.scrollWidth<=innerWidth`),true);await page.shot('native-mobile-wheel');checks.push('390px resident wheel stays within viewport');assert.deepEqual(errors,[]);
 const report={passed:true,actual_worker_processes:true,external_provider:false,checks,uncaught_errors:errors,limits:['Injected deterministic provider; real upstream model quality and background autonomy are separate checks.'],artifacts:root};await writeFile(root+'/report.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}catch(error){if(page)await page.shot('failure').catch(()=>{});console.error(JSON.stringify({passed:false,message:error.stack,errors,artifacts:root}));process.exitCode=1;}finally{page?.socket.close();browser.kill('SIGTERM');await sleep(250);await rm(profile,{recursive:true,force:true});}
