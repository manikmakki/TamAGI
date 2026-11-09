/* Native household UI. Without the gateway bootstrap, standalone stays unchanged. */
(() => {
  'use strict';
  const bootstrap = window.TAMAGI_HOUSEHOLD_BOOTSTRAP;
  if (!bootstrap || window.TamagiHousehold) return;
  const $ = id => document.getElementById(id);
  const storageKey = name => `tamagi-household:${bootstrap.household_id}:${name}`;
  const residents = bootstrap.residents || [];
  const operator = bootstrap.operator || {id:'operator',name:'You'};
  const people = () => [operator,...residents];
  const person = id => people().find(entry => entry.id === id);
  const label = id => person(id)?.name || id;
  const bound = location.pathname.match(/^\/r\/([^/]+)\//)?.[1];
  let selected = bound ? decodeURIComponent(bound) : residents.find(entry => entry.id === localStorage.getItem(storageKey('resident')))?.id || bootstrap.default_resident_id || residents[0]?.id;
  let epoch = 0, opening = 0, current = null, snapshot = null, colors = bootstrap.preferences?.colors || {}, reading = false, readFrame = null;
  let choices = [], choiceIndex = 0, mentionStart = 0;
  const newParticipants = new Map(), lastThreads = new Map(), readWatermarks = new Map();
  const originalFetch = window.fetch.bind(window);
  const validColor = value => typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value);
  const palette = ['#e6b978','#b9a4e9','#91bfc9','#dfa3c4','#a9c78e'];
  function defaultColor(id) {
    if (id === operator.id) return '#73c997';
    const name = (person(id)?.name || '').toLowerCase();
    if (name === 'ember') return '#ef7771';
    if (name === 'echo') return '#8ecdf4';
    return palette[Math.max(0,residents.findIndex(entry => entry.id === id)) % palette.length];
  }
  const color = id => validColor(colors[id]) ? colors[id] : validColor(person(id)?.color) ? person(id).color : defaultColor(id);
  function scopeUrl(path, residentId = selected) {
    const url = new URL(path,location.origin);
    if (url.origin === location.origin && residentId && !url.pathname.startsWith('/r/') && (/^\/api\//.test(url.pathname) || /^\/panels\//.test(url.pathname))) url.pathname = '/r/' + encodeURIComponent(residentId) + url.pathname;
    return url.href;
  }
  // Capture the resident when a fetch starts, not when its response arrives.
  window.fetch = function(input,options = {}) {
    const url = new URL(input instanceof Request ? input.url : input,location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith('/api/') || /^\/api\/(auth|household)(\/|$)/.test(url.pathname)) return originalFetch(input,options);
    const headers = new Headers(options.headers || (input instanceof Request ? input.headers : undefined));
    if (!headers.has('X-Tamagi-Resident')) headers.set('X-Tamagi-Resident',selected);
    return originalFetch(input,{...options,headers});
  };
  async function request(path,options = {}) {
    const response = await window.fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers}});
    if (response.status === 401) { location.assign('/login'); throw new Error('Please sign in again'); }
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The household request could not be completed');
    return result;
  }
  const notice = message => window.TamagiShell?.notice(message);
  function node(tag,text,className) { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; }
  function participantIds() { return convId && current?.id === convId ? current.participants : [operator.id,...(newParticipants.get(selected) || [selected])]; }

  // Markdown is sanitized before insertion; mention highlighting only edits text.
  function safeMarkdown(content) {
    const template = document.createElement('template');
    template.innerHTML = typeof marked === 'undefined' ? esc(content || '') : marked.parse(content || '',{breaks:true,gfm:true});
    const allowed = new Set(['P','BR','EM','STRONG','CODE','PRE','BLOCKQUOTE','UL','OL','LI','A','H1','H2','H3','H4','H5','H6','HR','TABLE','THEAD','TBODY','TR','TH','TD','DEL']);
    for (const element of [...template.content.querySelectorAll('*')]) {
      if (!allowed.has(element.tagName)) { if (['SCRIPT','STYLE','IFRAME','OBJECT','EMBED','IMG','FORM'].includes(element.tagName)) element.remove(); else element.replaceWith(...element.childNodes); continue; }
      for (const attribute of [...element.attributes]) if (!(element.tagName === 'A' && ['href','title'].includes(attribute.name))) element.removeAttribute(attribute.name);
      if (element.tagName === 'A') {
        try { const url = new URL(element.getAttribute('href') || '',location.origin); if (!['https:','http:'].includes(url.protocol)) element.removeAttribute('href'); else if (url.origin !== location.origin) { element.target = '_blank'; element.rel = 'noopener noreferrer'; } } catch { element.removeAttribute('href'); }
      }
    }
    return template.innerHTML;
  }
  const escapePattern = value => value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  function mentionMatches(text,ids = participantIds()) {
    const matches = [];
    const candidates = people().filter(entry => ids.includes(entry.id)).sort((a,b) => b.name.length-a.name.length);
    for (const entry of candidates) for (const token of [entry.name,entry.id]) {
      const expression = new RegExp('(^|[^\\p{L}\\p{N}_@])@(' + escapePattern(token) + ')(?=$|[^\\p{L}\\p{N}_-])','giu');
      for (const match of text.matchAll(expression)) { const start = match.index + match[1].length, end = start+match[2].length+1; if (!matches.some(item => start < item.end && end > item.start)) matches.push({start,end,id:entry.id}); }
    }
    return matches.sort((a,b) => a.start-b.start);
  }
  function resolveMentions(text) {
    const matches = mentionMatches(text); let remainder = text;
    for (const match of matches.slice().reverse()) remainder = remainder.slice(0,match.start) + ' '.repeat(match.end-match.start) + remainder.slice(match.end);
    if (/(^|[^\p{L}\p{N}_@])@[^\s@]+/u.test(remainder)) throw new Error('Choose someone from the @ suggestions, or add them with Participants.');
    const ids = [...new Set(matches.map(match => match.id))]; return ids.length ? ids : undefined;
  }
  function highlightMentions(bubble,ids) {
    if (!bubble) return;
    const walker = document.createTreeWalker(bubble,NodeFilter.SHOW_TEXT), texts = [];
    while (walker.nextNode()) if (!walker.currentNode.parentElement.closest('pre,code,a,.hh-mention')) texts.push(walker.currentNode);
    for (const text of texts) {
      const matches = mentionMatches(text.textContent,ids); if (!matches.length) continue;
      const fragment = document.createDocumentFragment(); let offset = 0;
      for (const match of matches) { fragment.append(document.createTextNode(text.textContent.slice(offset,match.start))); const span = node('span',text.textContent.slice(match.start,match.end),'hh-mention'); span.dataset.actorId = match.id; span.style.setProperty('--hh-mention-color',color(match.id)); fragment.append(span); offset = match.end; }
      fragment.append(document.createTextNode(text.textContent.slice(offset))); text.replaceWith(fragment);
    }
  }
  function decorateMessage(element,role,metadata = {}) {
    const author = metadata.author_id || (role === 'user' ? operator.id : selected);
    element.dataset.authorId = author;
    if (metadata.sequence) element.dataset.sequence = String(metadata.sequence);
    if (metadata.message_id) element.dataset.messageId = metadata.message_id;
    element.style.setProperty('--hh-speaker-color',color(author));
    element.querySelector('.bubble')?.after(node('div',metadata.author_name || label(author),'hh-speaker'));
    if (metadata.image) {
      const image=node('img',undefined,'hh-attachment');image.alt='Attached image';
      image.src='/api/conversations/'+encodeURIComponent(convId)+'/attachments/'+encodeURIComponent(metadata.message_id);
      image.onerror=()=>{image.replaceWith(node('span','Attached image unavailable','hh-attachment-error'));};
      element.querySelector('.bubble')?.append(image);
    }
    highlightMentions(element.querySelector('.bubble'),metadata.mentions || []);
  }
  function paintParticipants() {
    const strip = $('household-participant-strip'); if (!strip) return; strip.replaceChildren();
    const addressed = mentionMatches($('input').value).map(match => match.id);
    for (const id of [...new Set([operator.id,...participantIds()])]) { const chip = node('span',label(id),'hh-person'); chip.style.setProperty('--hh-person-color',color(id)); chip.classList.toggle('addressed',addressed.includes(id)); strip.append(chip); }
    if(current?.id===convId)for(const invitation of current.invitations||[]){
      if(['pending','declined'].includes(invitation.status)&&!participantIds().includes(invitation.participant_id)){
        const chip=node('span',label(invitation.participant_id)+' · '+(invitation.status==='pending'?'invited':'declined'),'hh-person');
        chip.style.setProperty('--hh-person-color',color(invitation.participant_id));strip.append(chip);
      }
    }
  }
  function decorateProgress(element,metadata={}){
    const author=metadata.author_id||metadata.resident_id;if(!author)return;
    element.dataset.authorId=author;element.style.setProperty('--hh-speaker-color',color(author));
    element.querySelector('.interim-toggle')?.append(node('span',metadata.author_name||label(author),'hh-progress-speaker'));
  }
  function hideMentions() { choices = []; if (!$('household-mention-list')) return; $('household-mention-list').hidden = true; $('input').setAttribute('aria-expanded','false'); $('input').removeAttribute('aria-activedescendant'); }
  function paintMentionChoices() {
    const list = $('household-mention-list'); list.replaceChildren();
    choices.forEach((entry,index) => { const button = node('button','@' + entry.name); button.type = 'button'; button.id = 'hh-mention-' + index; button.setAttribute('role','option'); button.setAttribute('aria-selected',String(index === choiceIndex)); button.style.setProperty('--hh-person-color',color(entry.id)); button.onmousedown = event => event.preventDefault(); button.onclick = () => completeMention(index); list.append(button); });
    list.hidden = !choices.length; $('input').setAttribute('aria-expanded',String(!!choices.length));
    if (choices.length) $('input').setAttribute('aria-activedescendant','hh-mention-' + choiceIndex);
  }
  function updateMentions() {
    paintParticipants(); const input = $('input'), match = input.value.slice(0,input.selectionStart).match(/(^|\s)@([^\s@]*)$/);
    if (!match) { hideMentions(); return; }
    mentionStart = input.selectionStart-match[2].length-1;
    choices = people().filter(entry => participantIds().includes(entry.id) && entry.name.toLowerCase().includes(match[2].toLowerCase())).slice(0,8); choiceIndex = 0; paintMentionChoices();
  }
  function completeMention(index) {
    const entry = choices[index]; if (!entry) return;
    const input = $('input'); input.setRangeText('@' + entry.name + ' ',mentionStart,input.selectionStart,'end'); hideMentions(); input.focus(); input.dispatchEvent(new Event('input',{bubbles:true}));
  }
  function paintWheel() {
    const wheel = $('household-wheel-options'); if (!wheel) return;
    const focused = wheel.contains(document.activeElement) ? document.activeElement.dataset.residentId : null;
    wheel.replaceChildren();
    wheel.append(node('span',label(selected),'hh-wheel-current'));
    residents.forEach((entry,index) => {
      const button = node('button',undefined,'hh-wheel-person'); button.type = 'button'; button.dataset.residentId = entry.id; button.setAttribute('aria-pressed',String(entry.id === selected)); button.style.setProperty('--hh-person-color',color(entry.id));
      const angle = -Math.PI/2 + index*2*Math.PI/residents.length; button.style.left = `${50+Math.cos(angle)*34}%`; button.style.top = `${50+Math.sin(angle)*34}%`;
      const face = document.createElement('canvas'); face.className='hh-wheel-face'; face.width=300; face.height=320; face.setAttribute('aria-hidden','true');
      const avatar = entry.avatar || 'vex', bones = TamagiAvatarRuntime.createBones(avatar,createBone,() => applyPivotDefaults(PRESETS.default.bones()));
      renderFrame(face,bones,TamagiAvatarRuntime.manifest(avatar,bones,buildManifest),ANIMATIONS.idle(bones,0)||{},0,avatar,{});
      button.append(face,node('span',entry.name,'hh-wheel-name'));
      button.onclick = () => { $('household-wheel-dialog').close(); selectResident(entry.id).catch(error => notice(error.message)); };
      button.onkeydown = event => { if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key)) return; event.preventDefault(); const next = event.key === 'Home' ? 0 : event.key === 'End' ? residents.length-1 : (index+(['ArrowLeft','ArrowUp'].includes(event.key)?-1:1)+residents.length)%residents.length; wheel.querySelectorAll('button')[next].focus(); };
      wheel.append(button);
    });
    if (focused) [...wheel.querySelectorAll('button')].find(button => button.dataset.residentId === focused)?.focus();
    $('household-character-wheel').textContent = '◉ ' + label(selected); $('household-character-wheel').style.setProperty('--hh-person-color',color(selected));
  }
  async function selectResident(id,{openThread = true} = {}) {
    if (!residents.some(entry => entry.id === id)) throw new Error('That resident is unavailable');
    if (id === selected) return;
    window.TamagiShell?.saveDraft(); selected = id; epoch++; current = null; snapshot = null;
    localStorage.setItem(storageKey('resident'),id); paintWheel(); paintParticipants(); hideMentions(); const token = epoch;
    if (!convId) window.TamagiShell?.restoreDraft();
    _avatarId = person(id)?.avatar || 'vex'; _spriteConfig = null; _spriteImages = {};
    _rigBones = TamagiAvatarRuntime.createBones(_avatarId,createBone,() => applyPivotDefaults(PRESETS.default.bones())); _rigStartTime = null;
    updateState({name:label(id),resident_id:id});
    for (const frame of document.querySelectorAll('iframe.view-frame[data-src]')) if (frame.hasAttribute('src')) frame.src = scopeUrl(frame.dataset.src,id);
    if (openThread) { const previous = lastThreads.get(id); if (previous) await window.openConversation(previous); else window.newConversation(); }
    if (token !== epoch) return;
    await Promise.all([loadSpriteConfig(),loadState()]);
    if (token !== epoch) return;
    await window.loadConversations();
    if (window.TamagiShell && ['map','journal','inventory','companions','settings'].includes(TamagiShell.view())) { if (TamagiShell.view() === 'inventory') await window.TamagiHomeViews?.inventory(); else await window.TamagiHomeViews?.load(); }
  }
  function scheduleRead() { if (readFrame !== null) return; readFrame = requestAnimationFrame(() => { readFrame = requestAnimationFrame(() => { readFrame = null; acknowledgeRead().catch(error => notice(error.message)); }); }); }
  async function acknowledgeRead() {
    if (reading || !snapshot || document.hidden || !document.hasFocus() || window.TamagiShell?.view() !== 'chat' || snapshot.id !== convId) return;
    const rendered = snapshot, area = $('chat-messages'), bounds = area.getBoundingClientRect();
    const visible = [...area.querySelectorAll('.msg[data-sequence]')].filter(element => { const rectangle = element.getBoundingClientRect(); return rectangle.bottom > Math.max(bounds.top,0) && rectangle.top < Math.min(bounds.bottom,innerHeight); });
    const watermark = Math.min(rendered.watermark,Math.max(0,...visible.map(element => Number(element.dataset.sequence))));
    if (!watermark || watermark <= (readWatermarks.get(rendered.id) || 0)) return;
    reading = true;
    try { await request('/api/household/threads/' + encodeURIComponent(rendered.id) + '/read',{method:'POST',body:JSON.stringify({snapshot_id:rendered.snapshot_id,watermark})}); readWatermarks.set(rendered.id,Math.max(watermark,readWatermarks.get(rendered.id)||0)); await refresh(); } finally { reading = false; }
  }
  function historyRendered(id,history) {
    current = {id,participants:history.participants || [],invitations:history.invitations||[]}; if (current.participants.includes(selected)) lastThreads.set(selected,id);
    snapshot = history.snapshot_id ? {id,snapshot_id:history.snapshot_id,watermark:history.watermark} : null; paintParticipants(); scheduleRead();
  }
  async function refresh() {
    const state = await request('/api/household'); colors = state.preferences?.colors || state.colors || colors;
    for (const update of state.residents || []) Object.assign(person(update.id) || {},update);
    if ($('household-unread')) $('household-unread').textContent = `Unread threads (${state.unread_count || 0})`;
    if ($('household-for-you')) { $('household-for-you').textContent = `For you (${state.addressed_thread_count || 0})`; $('household-for-you').disabled = !state.addressed_thread_count; }
    paintWheel(); paintParticipants();
    for (const element of document.querySelectorAll('.msg[data-author-id]')) element.style.setProperty('--hh-speaker-color',color(element.dataset.authorId));
    for (const element of document.querySelectorAll('.hh-mention[data-actor-id]')) element.style.setProperty('--hh-mention-color',color(element.dataset.actorId));
    await refreshWorld();
    if(window.TamagiShell?.view()==='journal')await window.loadMindPanel();
    return state;
  }
  async function refreshWorld() {
    const panel=$('household-world-status');if(!panel)return;
    const state=await request('/api/household/world');panel.replaceChildren(node('h3','Household World Thread'));
    panel.append(node('p',state.enabled?'Scheduled opportunities are running.':'Scheduled opportunities are paused.'));
    if(state.enabled&&state.next_at)panel.append(node('p','Next opportunity: '+new Date(state.next_at*1000).toLocaleString(undefined,{timeZone:state.timezone})+' ('+state.timezone+')'));
    if(state.last_resident_id)panel.append(node('p','Last opportunity: '+label(state.last_resident_id)));
    for(const work of state.active||[]){const row=node('p',label(work.resident_id)+' is taking an autonomous turn. '),stop=node('button','Stop');stop.type='button';stop.onclick=async()=>{try{await request('/api/household/world/'+encodeURIComponent(work.work_id)+'/stop',{method:'POST'});await refreshWorld();}catch(error){notice(error.message);}};row.append(stop);panel.append(row);}
  }
  function dialog(id,title) {
    const element = node('dialog',undefined,'shell-dialog hh-dialog'); element.id = id;
    const close = node('button','×','dialog-close'); close.type = 'button'; close.setAttribute('aria-label','Close'); close.onclick = () => element.close();
    element.append(close,node('h2',title)); document.body.append(element); return element;
  }
  async function showUnread(addressedOnly=false) {
    const result = await request('/api/household/threads?unread=1'), list = $('household-unread-list'); list.replaceChildren();
    for (const thread of result.threads || []) { if(addressedOnly&&!thread.addressed_count)continue;const button = node('button',thread.title || 'Conversation','hh-thread-choice'); button.type='button'; button.append(node('small',(thread.participants||[]).map(label).join(' · ')));if(thread.addressed_count)button.append(node('strong',thread.mention_count?'@'+operator.name+' mentioned':'For you','hh-attention')); button.onclick=async()=>{try{$('household-unread-dialog').close();if(!thread.participants?.includes(selected)){const entry=residents.find(item=>thread.participants?.includes(item.id));if(entry)await selectResident(entry.id,{openThread:false});}await window.openConversation(thread.id);}catch(error){notice(error.message);}};list.append(button); }
    if (!list.childElementCount) list.append(node('p','You are all caught up.')); $('household-unread-dialog').showModal();
  }
  async function showLibrary() {
    const result=await request('/api/household/artifacts'),list=$('household-library-list');list.replaceChildren();
    for(const artifact of result.artifacts||[]){const section=node('section');section.append(node('h3',artifact.title),node('small',`v${artifact.version} · Published by ${label(artifact.author_id)}`),node('p',artifact.description||''));for(const path of Object.keys(artifact.files||{})){const link=node('a',path);link.href='/api/household/artifacts/'+encodeURIComponent(artifact.artifact_id)+'/versions/'+artifact.version+'/files/'+path.split('/').map(encodeURIComponent).join('/');link.download=path.split('/').pop();section.append(link,document.createElement('br'));}list.append(section);}
    if(!list.childElementCount)list.append(node('p','No household-public artifacts have been published yet.'));$('household-library-dialog').showModal();
  }
  function showParticipants() {
    const list = $('household-participant-options'); list.replaceChildren();
    for (const entry of residents) { const row=node('label',undefined,'hh-check'), input=document.createElement('input');input.type='checkbox';input.value=entry.id;input.checked=participantIds().includes(entry.id);input.disabled=!!convId&&input.checked;row.append(input,node('span',entry.name));list.append(row); }
    $('household-membership-notice').hidden=!convId; $('household-participants-dialog').showModal();
  }
  function mount() {
    document.body.dataset.household='true';
    const worldStatus=node('section',undefined,'card');worldStatus.id='household-world-status';$('view-journal')?.querySelector('.page-heading')?.after(worldStatus);
    const wheelButton=node('button','','hh-wheel-toggle');wheelButton.type='button';wheelButton.id='household-character-wheel';wheelButton.setAttribute('aria-haspopup','dialog');wheelButton.setAttribute('aria-label','Choose a resident');$('a-name').before(wheelButton);
    const wheel=dialog('household-wheel-dialog','Choose a resident');wheel.classList.add('hh-wheel-dialog');const options=node('div',undefined,'hh-wheel-options');options.id='household-wheel-options';wheel.append(options);wheelButton.onclick=()=>{paintWheel();wheel.showModal();options.querySelector('[aria-pressed=true]')?.focus();};
    const toolbar=node('div',undefined,'hh-chat-toolbar');
    for(const[id,text,action]of[['household-new-thread','＋ New Thread',()=>window.newConversation()],['household-unread','Unread threads (0)',()=>showUnread().catch(error=>notice(error.message))],['household-for-you','For you (0)',()=>showUnread(true).catch(error=>notice(error.message))],['household-participants','Participants',showParticipants],['household-library','Shared library',()=>showLibrary().catch(error=>notice(error.message))]]){const button=node('button',text);button.id=id;button.type='button';button.onclick=action;if(id==='household-for-you'){button.disabled=true;button.classList.add('hh-attention');button.setAttribute('aria-live','polite');}toolbar.append(button);}
    const strip=node('div',undefined,'hh-participant-strip');strip.id='household-participant-strip';$('chat-messages').before(toolbar,strip);
    const mentions=node('div',undefined,'hh-mention-list');mentions.id='household-mention-list';mentions.hidden=true;mentions.setAttribute('role','listbox');mentions.setAttribute('aria-label','Mention a participant');$('input').before(mentions);$('input').setAttribute('aria-autocomplete','list');$('input').setAttribute('aria-controls',mentions.id);$('input').setAttribute('aria-expanded','false');
    $('input').addEventListener('input',updateMentions);$('input').addEventListener('click',updateMentions);$('input').addEventListener('keydown',event=>{if(!choices.length||!['ArrowDown','ArrowUp','Enter','Tab','Escape'].includes(event.key))return;event.preventDefault();event.stopImmediatePropagation();if(event.key==='Escape')hideMentions();else if(event.key==='Enter'||event.key==='Tab')completeMention(choiceIndex);else{choiceIndex=(choiceIndex+(event.key==='ArrowDown'?1:-1)+choices.length)%choices.length;paintMentionChoices();}},true);
    const unread=dialog('household-unread-dialog','Unread conversations'), unreadList=node('div');unreadList.id='household-unread-list';unread.append(unreadList);
    const library=dialog('household-library-dialog','Household-public artifacts'),libraryList=node('div');libraryList.id='household-library-list';library.append(node('p','Published versions are available to every resident. Importing a version makes a private copy; executable skills require separate activation.'),libraryList);
    const membership=dialog('household-participants-dialog','Conversation participants'), memberList=node('div');memberList.id='household-participant-options';membership.append(memberList);const text=node('p','Invited residents choose whether to join. Joining shares all earlier messages in this thread. Current participants choose whether to leave.','muted');text.id='household-membership-notice';membership.append(text);
    const save=node('button','Use these participants');save.type='button';save.onclick=async()=>{const ids=[...memberList.querySelectorAll('input:checked')].map(input=>input.value);if(!ids.length){notice('Choose at least one resident.');return;}save.disabled=true;try{if(convId){const id=convId,result=await request('/api/household/threads/'+encodeURIComponent(id)+'/participants',{method:'POST',body:JSON.stringify({participants:ids})});if(id===convId)await TamagiShell.refreshHistory();notice(result.notice||'Invitations sent. Residents choose whether to join.');}else{newParticipants.set(selected,ids);paintParticipants();}membership.close();hideMentions();}catch(error){notice(error.message);}finally{save.disabled=false;}};membership.append(save);
    const appearance=node('article',undefined,'card hh-colors');appearance.append(node('h2','Conversation colors'),node('p','Choose a color for each speaker. Names stay visible below every message.','muted'));const form=node('form');form.id='household-color-form';
    for(const entry of people()){const row=node('label',undefined,'hh-color-row'),input=document.createElement('input');input.type='color';input.value=color(entry.id);input.dataset.actorId=entry.id;row.append(node('span',entry.name),input);form.append(row);}
    const submit=node('button','Save conversation colors');submit.type='submit';form.append(submit);form.onsubmit=async event=>{event.preventDefault();const values={};for(const input of form.querySelectorAll('input[type=color]'))values[input.dataset.actorId]=input.value;submit.disabled=true;try{await request('/api/household/preferences',{method:'PUT',body:JSON.stringify({colors:values})});colors=values;await refresh();notice('Conversation colors saved.');}catch(error){notice(error.message);}finally{submit.disabled=false;}};appearance.append(form);$('settings-operator').prepend(appearance);
    $('chat-messages').addEventListener('scroll',scheduleRead,{passive:true});window.addEventListener('focus',scheduleRead);document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh().then(scheduleRead).catch(error=>notice(error.message));});paintWheel();paintParticipants();
  }
  async function start() {
    if (!$('chat-messages')) return; mount();
    const oldNew=window.newConversation, oldOpen=window.openConversation;
    window.newConversation=function(){opening++;snapshot=null;current=null;oldNew();paintParticipants();hideMentions();};
    window.openConversation=async function(id){
      const attempt=++opening,token=epoch,thread=await request('/api/conversations/'+encodeURIComponent(id));
      if(attempt!==opening||token!==epoch)return;
      if(!thread.participants?.includes(selected)){
        const entry=residents.find(item=>thread.participants?.includes(item.id));
        if(!entry)throw new Error('No available resident participates in this thread.');
        const switching=selectResident(entry.id,{openThread:false}),switchedEpoch=epoch;
        await switching;if(attempt!==opening||epoch!==switchedEpoch)return;
      }
      await oldOpen(id);paintParticipants();scheduleRead();
    };
    window.loadConversations=async function(){try{const residentId=selected,result=await request('/api/conversations',{headers:{'X-Tamagi-Resident':residentId}});if(residentId!==selected)return;const list=$('conv-list');list.replaceChildren();for(const thread of result.conversations||[]){const button=node('button',undefined,'conv-item hh-thread-choice'+(thread.id===convId?' active':''));button.type='button';button.append(node('div',thread.title||'Conversation','ci-title'),node('small',(thread.participants||[]).map(label).join(' · '),'ci-meta'));button.onclick=()=>window.openConversation(thread.id).catch(error=>notice(error.message));list.append(button);}}catch(error){notice(error.message);}};
    await refresh();for(const input of $('household-color-form').querySelectorAll('input[type=color]'))input.value=color(input.dataset.actorId);
    setInterval(()=>{if(!document.hidden)refresh().then(scheduleRead).catch(()=>{});},5000);
  }
  window.TamagiHousehold={enabled:true,selectedResidentId:()=>selected,residentName:label,selectionEpoch:()=>epoch,selectResident,scopeUrl,storageKey,
    draftKey:id=>id||'new:'+selected,participants:()=>[...participantIds()].filter(id=>id!==operator.id),resolveMentions,safeMarkdown,decorateMessage,decorateProgress,historyRendered,refresh,scheduleRead,
    clearThread:()=>{snapshot=null;current=null;if($('household-participant-strip'))paintParticipants();},acceptResponse:(id,token)=>id===selected&&token===epoch};
  const stylesheet=document.createElement('link');stylesheet.rel='stylesheet';stylesheet.href='/assets/household-ui.css?v=1';document.head.append(stylesheet);
  document.addEventListener('DOMContentLoaded',()=>start().catch(error=>notice(error.message)));
})();
