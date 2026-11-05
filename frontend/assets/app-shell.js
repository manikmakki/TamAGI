/* Shared navigation and observation. Execution is owned by the backend. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const works = new Map(), cursors = new Map(), delivered = new Set(), unread = new Set(), drafts = new Map(), progress = new Map();
  const submissions = new Map();
  let seen;
  try{seen=new Set(JSON.parse(localStorage.getItem('tamagi-seen-work')||'[]'));}catch{seen=new Set();}
  const legacyEvent = handleWsEvent;
  const legacySend = sendMessage;
  let durable=true;
  let view = 'chat', selection = 0, hydrating = true, drawerFocus = null, archiveTab = 'chats', polling = false;
  let chatPosition={scroll:0,bottom:true};
  const titles = {chat:'Chat',map:'Map',journal:'Journal',inventory:'Inventory',companions:'Companions',archive:'Archive',settings:'Settings'};
  const active = work => work && ['queued','running'].includes(work.status);
  const currentWork = () => [...works.values()].filter(w => w.kind === 'chat' && w.conversation_id === convId && active(w)).sort((a,b) => b.created_at-a.created_at)[0];

  async function request(path, options = {}) {
    const headers = options.body instanceof FormData ? {} : {'Content-Type':'application/json'};
    const response = await fetch('/api' + path, {...options, headers:{...headers,...options.headers}});
    if (response.status === 401) { location.assign('/login'); throw new Error('Please sign in again'); }
    const result = await response.json();
    if (!response.ok) {const error=new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be completed');error.status=response.status;throw error;}
    return result;
  }
  function notice(message='') { $('shell-notice').textContent=message; $('shell-notice').hidden=!message; }
  function restoreScroll(element,top){const previous=element.style.scrollBehavior;element.style.scrollBehavior='auto';element.scrollTop=top;element.style.scrollBehavior=previous;}
  const uuid = () => crypto.randomUUID?.() || Date.now().toString(36)+Math.random().toString(36).slice(2);
  function markSeen(id){seen.add(id);unread.delete(id);localStorage.setItem('tamagi-seen-work',JSON.stringify([...seen].slice(-500)));}
  function subscribe(work) {
    if (wsReady) ws.send(JSON.stringify({type:'subscribe', work_id:work.work_id, after:cursors.get(work.work_id)||0}));
  }
  function remember(work, initial=false) {
    const previous = works.get(work.work_id);
    works.set(work.work_id, {...previous,...work});
    const pending=submissions.get(work.request_id);
    if(pending){
      submissions.delete(work.request_id);
      if(pending.selection===selection && pending.conversation===convId && $('input').value.trim()===pending.message && pendingFile===pending.file){
        $('input').value='';autoResize($('input'));clearAttachment();drafts.delete(convId||'new');convId=work.conversation_id;localStorage.setItem('tamagi-conversation',convId);activeRequestId=work.request_id;renderHistory(convId,true).catch(e=>notice(e.message));
      }
    }
    if (initial && !active(work)) {delivered.add(work.work_id);if(!seen.has(work.work_id))unread.add(work.work_id);}
    if (!active(work) && !delivered.has(work.work_id)) {
      delivered.add(work.work_id);
      if (!initial) {
        if (work.kind === 'chat' && work.conversation_id === convId) {renderHistory(convId, true).catch(e=>notice(e.message));if(view==='chat')markSeen(work.work_id);else unread.add(work.work_id);}
        else if (work.kind === 'companion' && view === 'companions' && window.TamagiHomeViews?.finished(work)) markSeen(work.work_id);
        else unread.add(work.work_id);
        if (work.result?.state) updateState(work.result.state);
        loadConversations();
      }
    }
    updateActivity();
  }
  function receive(event) {
    if(!durable){legacyEvent(event);return;}
    if(event.type==='work_resync'){cursors.delete(event.work_id);progress.delete(event.work_id);return;}
    if (event.type === 'accepted') {
      remember(event);
      if (activeRequestId === event.request_id && !convId) {convId=event.conversation_id;localStorage.setItem('tamagi-conversation',convId);}
      return;
    }
    if (event.type === 'work_snapshot') {
      remember(event.work);
      return;
    }
    if (!event.work_id) return;
    if (event.event_id && event.event_id <= (cursors.get(event.work_id)||0)) return;
    if (event.event_id) cursors.set(event.work_id,event.event_id);
    const work = works.get(event.work_id);
    if (!work) return;
    if (event.type === 'quiet_turn') {work.quiet_turns=event.ordinal;updateActivity();}
    if (event.type === 'message' || event.type === 'work_ended') {
      remember({...work,status:event.turn_status||event.status||'completed',result:event.type==='message'?event:work.result});
      return;
    }
    if (event.type === 'tool_approval_required') {
      work.approval={...event,expires_at:Date.now()/1000+(event.timeout_seconds||30)};
      updateActivity(); return;
    }
    const buffer = progress.get(work.work_id)||[];
    buffer.push(event);if(buffer.length>256)buffer.shift();progress.set(work.work_id,buffer);
    if (work.kind === 'chat' && work.conversation_id === convId) legacyEvent(event);
  }
  async function feed(work) {
    const snapshot = await request('/chat/requests/'+encodeURIComponent(work.work_id)+'?after='+(cursors.get(work.work_id)||0));
    if(snapshot.reset){cursors.delete(work.work_id);progress.delete(work.work_id);}
    for (const event of snapshot.events) receive(event);
    remember(snapshot.work);
  }
  window.connectWS = function () {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {wsReady=true;setConnected(true);if(!hydrating)for(const work of works.values())if(active(work))subscribe(work);};
    ws.onmessage = message => {try{receive(JSON.parse(message.data));}catch(error){console.error('Activity update',error);}};
    ws.onclose = event => {wsReady=false;setConnected(false);if(event.code===1008){location.assign('/login');return;}setTimeout(connectWS,3000);};
    ws.onerror = () => {wsReady=false;};
  };
  window.handleWsEvent = receive;
  async function poll(initial=false) {
    if(polling)return;polling=true;
    try {
      const data = await request('/chat/requests');
      for (const work of data.requests.slice().reverse()) {
        const known = works.has(work.work_id);
        remember(work,initial);
        if(active(work) && !hydrating) {
          if(!known)subscribe(work);
          if(!wsReady)await feed(work);
        }
      }
      setConnected(true);
    } finally {polling=false;}
  }
  function saveDraft() {
    drafts.set(convId||'new',{text:$('input').value,file:pendingFile,scroll:$('chat-messages').scrollTop});
  }
  function restoreDraft() {
    const saved=drafts.get(convId||'new');
    $('input').value=saved?.text||'';autoResize($('input'));clearAttachment();
    if(saved?.file){pendingFile=saved.file;$('chip-name').textContent=pendingFile.name;$('chip-icon').textContent=pendingFile.type.startsWith('image/')?'🖼':'📄';$('attach-chip').classList.add('on');$('attach-btn').classList.add('has-file');}
  }
  async function renderHistory(id, bottom=false) {
    if(!id)return;
    const version=selection, data=await request('/conversations/'+encodeURIComponent(id));
    if(version!==selection||id!==convId)return;
    const area=$('chat-messages'),scroll=area.scrollTop;area.replaceChildren();currentInterimGroup=null;clearLiveTags();liveToolOutputs={};
    const ids=new Set();
    for(const message of data.messages||[]) {
      ids.add(message.id);
      const meta=message.metadata||{},thinking=meta.thinking_blocks||[],interim=meta.interim_messages||[];
      if(message.role==='assistant'&&(thinking.length||interim.length)) {
        const details=document.createElement('details');details.className='msg assistant history-progress';
        const summary=document.createElement('summary');summary.textContent=(thinking.length?'Reasoning':'Activity')+' · '+(thinking.length+interim.length)+' steps';details.append(summary);
        for(const text of [...thinking,...interim]){const item=document.createElement('div');item.textContent=text;details.append(item);}area.append(details);
      }
      if(message.role==='assistant'&&!message.content?.trim())addTurnStatus(meta.turn_status||'completed');
      else addMsg(message.role,message.content,meta.skills_used||[],message.timestamp,meta.context_compressed||0,meta.sm_mutations||[]);
    }
    const matching=[...works.values()].filter(w=>w.kind==='chat'&&w.conversation_id===id).sort((a,b)=>a.created_at-b.created_at);
    if(view==='chat')for(const work of matching)if(!active(work))markSeen(work.work_id);
    for(const work of matching) {
      if(!ids.has(work.input_message.id))addMsg('user',work.input_message.content,[],work.input_message.timestamp);
    }
    const running=currentWork();activeRequestId=running?.request_id||null;
    if(running) {
      for(const event of progress.get(running.work_id)||[])if(!['typing','tool_approval_required'].includes(event.type))legacyEvent(event);
    } else {
      const latest=matching.at(-1);
      if(latest && !active(latest) && latest.status!=='completed' && !(data.messages||[]).some(m=>m.role==='assistant'&&m.metadata?.execution_id===latest.execution_id))addTurnStatus(latest.status);
    }
    restoreScroll(area,bottom?area.scrollHeight:(drafts.get(id)?.scroll??scroll));
    updateActivity();
  }
  window.openConversation = async function (id) {
    saveDraft();selection++;convId=id;localStorage.setItem('tamagi-conversation',id);restoreDraft();
    for(const work of works.values())if(work.conversation_id===id&&!active(work))markSeen(work.work_id);
    await renderHistory(id);navigate('/');loadConversations();
  };
  window.newConversation = function () {
    saveDraft();selection++;convId=null;localStorage.removeItem('tamagi-conversation');restoreDraft();
    $('chat-messages').innerHTML='<div class="welcome-msg" id="welcome"><strong>A new conversation.</strong><br>What shall we explore?</div>';
    activeRequestId=null;currentInterimGroup=null;clearLiveTags();navigate('/');updateActivity();$('input').focus();loadConversations();
  };
  window.sendMessage = async function () {
    if(!durable)return legacySend();
    const input=$('input'),message=input.value.trim(),file=pendingFile;
    if((!message&&!file)||(loading&&!quietActive))return;
    const version=selection, originalConversation=convId;
    const retry=[...submissions.values()].find(s=>s.conversation===convId&&s.message===message&&s.file===file);
    const body=retry?.body||{message:message||`Attached: ${file.name}`,conversation_id:convId,request_id:uuid()};
    loading=true;$('send-btn').disabled=true;notice('');
    try {
      if(!retry&&file?.type.startsWith('image/')){body.image_data=await toBase64(file);body.image_media_type=file.type;}
      else if(!retry&&file){const form=new FormData();form.append('file',file);const upload=await request('/upload',{method:'POST',body:form});body.message+=` [file uploaded to workspace: ${upload.path}]`;}
      submissions.set(body.request_id,{body,conversation:originalConversation,message,file,selection:version});
      let work;
      // An uncertain acknowledgment is retried with the same identity and payload.
      for(let attempt=0;attempt<2;attempt++) {
        try{work=await request('/chat/requests',{method:'POST',body:JSON.stringify(body)});break;}
        catch(error){if(attempt||!(error instanceof TypeError))throw error;}
      }
      remember(work);subscribe(work);
      if(work.conversation_id===convId)unread.delete(work.work_id);
      loadConversations();
    } catch(error) {if(!(error instanceof TypeError))submissions.delete(body.request_id);notice(error.message);}
    finally {updateActivity();}
  };
  function updateActivity() {
    if(!durable){$('shell-work-label').textContent=loading?'Working in Chat':'Ready';return;}
    const pending=[...works.values()].filter(active),target=pending.find(w=>w.status==='running')||pending[0],selected=currentWork();
    loading=!!selected;quietActive=!!(selected?.quiet_turns);activeRequestId=selected?.request_id||null;
    $('send-btn').disabled=loading&&!quietActive;setTyping(loading);
    $('shell-work').dataset.busy=String(!!target);
    const approval=pending.find(w=>w.approval&&w.approval.expires_at>Date.now()/1000);
    const label=approval?'Approval needed':target?(target.status==='queued'?'Queued':target.quiet_turns?'Continuing quietly':'Working')+(target.kind==='companion'?' with a companion':' in Chat')+(pending.length>1?` · ${pending.length-1} queued`:''):unread.size?`${unread.size} new ${unread.size===1?'update':'updates'}`:'Ready';
    if($('shell-work-label').textContent!==label)$('shell-work-label').textContent=label;
    $('shell-work-label').onclick=approval?()=>showApproval(approval):null;
    $('shell-work-label').setAttribute('role',approval?'button':'status');
    $('shell-work-label').tabIndex=approval?0:-1;
    $('shell-work-label').onkeydown=event=>{if(approval&&['Enter',' '].includes(event.key)){event.preventDefault();showApproval(approval);}};
    $('shell-work-label').style.cursor=approval?'pointer':'';
    $('shell-stop').hidden=!target;$('shell-stop').onclick=target?async()=>{try{remember(await request('/chat/requests/'+target.work_id+'/stop',{method:'POST'}));}catch(error){notice(error.message);}}:null;
    const destination=target||[...works.values()].filter(w=>unread.has(w.work_id)).sort((a,b)=>b.created_at-a.created_at)[0];
    $('shell-return').hidden=!destination||(view==='chat'&&destination.conversation_id===convId);
    $('shell-return').textContent=destination?.kind==='companion'?'Open companion':'Open chat';
    $('shell-return').onclick=()=>{if(destination?.kind==='companion'){navigate('/companions');if(!active(destination)){window.TamagiHomeViews?.finished(destination,true);markSeen(destination.work_id);updateActivity();}}else openConversation(destination.conversation_id);};
    const badge=document.querySelector('[data-unread="chat"]');badge.hidden=!unread.size;badge.textContent=unread.size;
  }
  function showApproval(work) {
    const approval=work.approval;if(!approval||approval.expires_at<=Date.now()/1000)return;
    const area=$('work-approval-content');area.replaceChildren();
    const command=document.createElement('pre');command.textContent=approval.command;
    const text=document.createElement('p');text.textContent=(approval.reason||'Approval required')+' · expires '+new Date(approval.expires_at*1000).toLocaleTimeString();area.append(command,text);
    for(const [label,approved,always] of [['Allow once',true,false],['Allow always',true,true],['Deny',false,false]]) {
      const button=document.createElement('button');button.textContent=label;
      button.onclick=async()=>{try{await request('/chat/requests/'+work.work_id+'/approval',{method:'POST',body:JSON.stringify({approval_id:approval.approval_id,approved,allow_always:always})});work.approval=null;$('work-approval-dialog').close();updateActivity();}catch(error){notice(error.message);}};
      area.append(button);
    }
    if(!$('work-approval-dialog').open)$('work-approval-dialog').showModal();
  }
  window.resolveApproval = function (id, approved, always, banner) {
    const work=[...works.values()].find(w=>w.approval?.approval_id===id);
    if(work)request('/chat/requests/'+work.work_id+'/approval',{method:'POST',body:JSON.stringify({approval_id:id,approved,allow_always:always})}).catch(e=>notice(e.message));
    banner?.remove();
  };
  window.openDrawer = function () {
    drawerFocus=document.activeElement;$('drawer').inert=false;$('drawer').classList.add('open');$('drawer-overlay').classList.add('open');
    document.querySelectorAll('[aria-controls="drawer"]').forEach(b=>b.setAttribute('aria-expanded','true'));
    $('drawer').querySelector('[aria-current="page"]')?.focus();
  };
  window.closeDrawer = function () {
    $('drawer').classList.remove('open');$('drawer-overlay').classList.remove('open');$('drawer').inert=true;
    document.querySelectorAll('[aria-controls="drawer"]').forEach(b=>b.setAttribute('aria-expanded','false'));
    if(drawerFocus?.isConnected)drawerFocus.focus();
  };
  window.switchTab = function (name) {
    archiveTab=name==='memories'?'memories':'chats';activeTab=view==='journal'?'mind':archiveTab;
    for(const item of ['chats','memories']){$('panel-'+item).classList.toggle('hidden',archiveTab!==item);$('tab-'+item).classList.toggle('active',archiveTab===item);$('tab-'+item).setAttribute('aria-selected',String(archiveTab===item));}
    if(archiveTab==='chats')loadConversations();else{loadMemTotalCount();$('mem-search').focus();}
  };
  function route(path) {
    const url=new URL(path,location.origin);let name=(url.pathname.split('/')[1]||'chat').replace(/\.html$/,'');
    if(name==='home')name='chat';if(name==='self-model')name='map';if(name==='rig-editor')name='settings';if(name==='growth')name='chat';
    return {name:titles[name]?name:'chat',setting:url.pathname.startsWith('/rig-editor')?'appearance':url.pathname.split('/')[2]||'operator',growth:url.pathname==='/growth'||url.hash==='#growth'};
  }
  function navigate(path, replace=false, historyEvent=false) {
    const next=route(path), previous=view;
    if(previous==='chat'&&next.name!=='chat'){const area=$('chat-messages');chatPosition={scroll:area.scrollTop,bottom:area.scrollHeight-area.scrollTop-area.clientHeight<30};}
    view=next.name;document.body.dataset.view=view;
    if(!historyEvent)history[replace?'replaceState':'pushState']({},'',path);
    $('app').hidden=view!=='chat';$('shell-pages').hidden=view==='chat';$('shell-view-title').textContent=titles[view];document.title='TamAGI'+(view==='chat'?'':' — '+titles[view]);
    document.querySelectorAll('.shell-page').forEach(section=>section.hidden=section.id!=='view-'+view);
    document.querySelectorAll('.app-menu-item').forEach(link=>link.dataset.view===view?link.setAttribute('aria-current','page'):link.removeAttribute('aria-current'));
    closeDrawer();
    if(previous!=='chat'&&view==='chat')requestAnimationFrame(()=>{restoreScroll($('chat-messages'),chatPosition.bottom?$('chat-messages').scrollHeight:chatPosition.scroll);});
    if(view==='chat')for(const work of works.values())if(work.kind==='chat'&&work.conversation_id===convId&&!active(work))markSeen(work.work_id);
    if(!next.growth&&$('growth-dialog').open)$('growth-dialog').close();
    if(previous!==view&&$('preview-dialog').open)$('preview-dialog').close();
    if(view==='journal'){activeTab='mind';$('panel-mind').classList.remove('hidden');loadMindPanel();}
    if(view==='archive')switchTab(archiveTab);
    if(view==='settings') {
      const setting=['connections','appearance'].includes(next.setting)?next.setting:'operator';
      for(const id of ['operator','connections','appearance'])$('settings-'+id).hidden=setting!==id;
      document.querySelectorAll('[data-setting]').forEach(link=>link.dataset.setting===setting?link.setAttribute('aria-current','page'):link.removeAttribute('aria-current'));
    }
    document.querySelectorAll('#view-'+view+' iframe[data-src]').forEach(frame=>{if(!frame.src&&!frame.hidden)frame.src=frame.dataset.src;});
    if(['journal','map','companions','settings'].includes(view))window.TamagiHomeViews?.load().catch(e=>notice(e.message));
    if(view==='inventory')window.TamagiHomeViews?.inventory().catch(e=>notice(e.message));
    if(view==='chat'&&previous==='settings'){loadSpriteConfig().catch(()=>{});loadState();}
    if(next.growth)window.TamagiHomeViews?.growth();updateActivity();
  }
  window.triggerDream = async function () {
    const button=$('dream-trigger-btn');button.disabled=true;button.textContent='Ticking…';
    try{const result=await request('/world/tick',{method:'POST'});notice(result.detail||'World Thread: '+result.status);await loadMindPanel();await loadState();}
    catch(error){notice(error.message);}finally{button.disabled=false;button.textContent='▶ Tick Now';}
  };
  window.TamagiShell={api:request,notice,navigate,works,remember,subscribe,uuid,markSeen,view:()=>view};
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[href]');if(!link||event.ctrlKey||event.metaKey||event.shiftKey||link.hasAttribute('download')||link.target)return;
    const url=new URL(link.href,location.href);
    if(url.origin===location.origin && !url.pathname.startsWith('/api/') && ['','home','map','journal','inventory','companions','archive','settings','rig-editor','self-model','growth'].includes(url.pathname.split('/')[1])) {event.preventDefault();navigate(url.pathname+url.search+url.hash);}
  });
  window.addEventListener('popstate',()=>navigate(location.pathname+location.hash,false,true));
  window.addEventListener('message',event=>{if(event.origin!==location.origin||![...document.querySelectorAll('iframe.view-frame')].some(frame=>frame.contentWindow===event.source))return;if(event.data?.type==='tamagi:navigate')navigate(event.data.path);if(event.data?.type==='tamagi:escape')$('shell-menu').focus();});
  document.addEventListener('keydown',event=>{
    if(!$('drawer').classList.contains('open'))return;
    if(event.key==='Escape'){event.preventDefault();closeDrawer();}
    if(event.key==='Tab'){const elements=[...$('drawer').querySelectorAll('button,a[href]')],first=elements[0],last=elements.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}}
  });
  document.addEventListener('DOMContentLoaded',async()=>{
    for(const dialog of document.querySelectorAll('.shell-dialog'))dialog.querySelector('.dialog-close')?.addEventListener('click',()=>dialog.close());
    try {
      await poll(true);
      const saved=localStorage.getItem('tamagi-conversation')||[...works.values()].filter(w=>w.kind==='chat').sort((a,b)=>b.created_at-a.created_at)[0]?.conversation_id;
      if(saved){convId=saved;try{await renderHistory(saved);}catch{convId=null;localStorage.removeItem('tamagi-conversation');}}
      hydrating=false;
      for(const work of works.values())if(active(work)){subscribe(work);await feed(work);}
    }catch(error){hydrating=false;if(error.status===503&&error.message.includes('home platform'))durable=false;else notice(error.message);}
    navigate(location.pathname+location.hash,true);
    setInterval(()=>{if(durable)poll().catch(()=>setConnected(false));},3000);
  });
})();
