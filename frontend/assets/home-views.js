/* Existing home capabilities, placed in the shared navigation. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const api=(...args)=>TamagiShell.api(...args),notice=(...args)=>TamagiShell.notice(...args);
  let state=null,friend=null,thread=null,restore=null;
  const item=(title,text='')=>{const node=document.createElement('div');node.className='item';const strong=document.createElement('strong');strong.textContent=title;const p=document.createElement('p');p.className='muted';p.textContent=text;node.append(strong,p);return node;};
  const empty=(target,text='')=>{target.replaceChildren();if(text){const p=document.createElement('p');p.className='empty';p.textContent=text;target.append(p);}};
  const button=(label,handler)=>{const b=document.createElement('button');b.textContent=label;b.onclick=()=>perform(handler);return b;};
  const link=(label,href)=>{const a=document.createElement('a');a.textContent=label;a.href=href;a.className='button';return a;};
  async function perform(work){try{await work();}catch(error){notice(error.message);}}
  const form=(id,handler)=>$(id).addEventListener('submit',event=>{event.preventDefault();const submit=$(id).querySelector('button');submit.disabled=true;perform(handler).finally(()=>submit.disabled=false);});

  function renderGrowth(growth){
    if(!growth)return;
    $('growth-title').textContent='Level '+growth.level+' · '+growth.stage;
    $('growth-summary').textContent=growth.total+' Growth · '+growth.points_to_next+' to Level '+(growth.level+1);
    $('recognition-progress').max=1;$('recognition-progress').value=growth.progress||0;
    $('growth-age').textContent='Age: '+Math.floor(growth.age_seconds/86400)+' days';
    empty($('growth-recognitions'));
    for(const record of (growth.recognitions||[]).slice(0,20)) {
      const d=record.data||record,node=item((d.pattern||'Recognition')+' · +'+d.points,d.reason||'Retained behavioral evidence');
      const details=document.createElement('details'),summary=document.createElement('summary'),evidence=document.createElement('pre');summary.textContent='Evidence';evidence.textContent=JSON.stringify({ruleVersion:d.rule_version,occasion:d.occasion_id,observedAt:d.observed_at,recognizedAt:d.recognized_at,sources:d.source_refs},null,2);details.append(summary,evidence);node.append(details);$('growth-recognitions').append(node);
    }
    empty($('growth-stages'));
    for(const record of growth.stage_history||[]) {const d=record.data||record;$('growth-stages').append(item('Level '+d.level,(d.name||'Stage '+d.level)+(d.timestamp?' · '+new Date(d.timestamp*1000).toLocaleString():'')));}
  }
  function render(){
    if(!state)return;
    renderGrowth(state.growth);empty($('places'));
    for(const place of state.home)$('places').append(item(place.data.name,place.data.description||''));
    empty($('notes'),state.notebook.length?'':'The notebook is open for a first shared thought.');
    for(const note of state.notebook.slice(-12).reverse()) {
      const actor=state.companions.find(p=>p.id===note.actor);
      const details=document.createElement('details'),summary=document.createElement('summary'),body=document.createElement('p');
      summary.textContent=(actor?.data.name||(note.actor==='operator'?'You':'TamAGI'))+' · '+note.data.text.slice(0,80)+(note.data.text.length>80?'…':'');body.textContent=note.data.text;details.className='item';details.append(summary,body);$('notes').append(details);
    }
    empty($('companions'),state.companions.length?'':'A lasting companion can begin with a name and a few qualities.');
    for(const peer of state.companions) {
      if(peer.data.status!=='active')continue;
      const card=document.createElement('article');card.className='card';card.append(item(peer.data.name,peer.data.self_description||peer.data.seed));
      card.append(button('Invite a conversation',async()=>{
        friend=peer.id;thread=null;$('consultation').hidden=false;$('consult-title').textContent='With '+peer.data.name;empty($('friend-reply'));
        const history=await api('/companions/'+encodeURIComponent(peer.id));
        const previous=(history.invocations||[]).filter(r=>r.data.caller_actor_id==='operator').at(-1);
        if(previous){thread=previous.data.thread_id;$('friend-reply').append(item('Earlier conversation',previous.data.text||previous.data.status));for(const work of TamagiShell.works.values())if(work.kind==='companion'&&work.conversation_id===thread&&!['queued','running'].includes(work.status))TamagiShell.markSeen(work.work_id);}
        $('friend-message').focus();
      }));$('companions').append(card);
    }
    empty($('pursuits'),state.pursuits.length?'':'There is room for an interest.');
    for(const record of state.pursuits) {
      const d=record.data,node=item(d.title,d.next_step||d.status),actions=document.createElement('div');actions.className='row';
      const options=d.status==='active'?[['Pause','paused'],['Complete','completed'],['Let go','abandoned']]:d.status==='paused'||d.status==='blocked'?[['Resume','active'],['Let go','abandoned']]:[];
      for(const [label,status] of options)actions.append(button(label,async()=>{await api('/pursuits',{method:'POST',body:JSON.stringify({action:'revise',id:record.id,attributes:{status},expected:record.revision})});await load();}));
      node.append(actions);$('pursuits').append(node);
    }
    $('runtime-details').textContent=JSON.stringify({resident:state.resident_id,installation:state.installation_id,authority:state.authority,automaticDeployment:state.automatic_deployment},null,2);
    $('enable-deployment').disabled=!!state.automatic_deployment;$('disable-deployment').disabled=!state.automatic_deployment;
  }
  async function load(){state=await api('/runtime');render();}
  async function growth(){await load();$('growth').open=true;if(!$('growth-dialog').open)$('growth-dialog').showModal();}
  async function preview(entry){
    const path='/api/inventory/file?path='+encodeURIComponent(entry.path)+'&preview=true';
    $('preview-title').textContent=entry.name;empty($('preview-content'));$('preview-note').textContent='Current workshop file';
    const dialog=$('preview-dialog');if(!dialog.open)dialog.showModal();
    if(['text/html','image/svg+xml'].includes(entry.media_type)){
      const iframe=document.createElement('iframe');iframe.title=entry.name+' · static preview';iframe.setAttribute('sandbox','');iframe.src=path;$('preview-content').append(iframe);$('preview-note').textContent='Static preview · scripts and network access are disabled.';
    }else if(['image/png','image/jpeg','image/gif','image/webp'].includes(entry.media_type)){
      const img=document.createElement('img');img.alt=entry.name;img.src=path;img.onerror=()=>$('preview-note').textContent='Preview unavailable; use download.';$('preview-content').append(img);
    }else{
      const response=await fetch(path);if(!response.ok){$('preview-note').textContent='Preview unavailable; use download.';return;}
      const pre=document.createElement('pre');pre.textContent=await response.text();$('preview-content').append(pre);
    }
  }
  async function inventory(){
    const data=await api('/inventory');empty($('inventory-items'),data.items.length?'':'The workshop is ready for a first creation.');
    for(const entry of data.items){
      const card=document.createElement('article');card.className='card creation-card';card.setAttribute('aria-label',entry.name);
      const icon=document.createElement('div');icon.className='creation-icon';icon.textContent=entry.media_type.startsWith('image/')?'◈':entry.media_type==='text/html'?'▧':'▤';icon.setAttribute('aria-hidden','true');
      const title=document.createElement('h2');title.textContent=entry.name;const description=document.createElement('p');description.className='muted';description.textContent=entry.description||(entry.path!==entry.name?entry.path:'');description.hidden=!description.textContent;
      const info=document.createElement('small');info.textContent='Current file · '+new Intl.NumberFormat().format(entry.size)+' bytes';
      const actions=document.createElement('div');actions.className='row';actions.append(button('Preview',()=>preview(entry)),link('Download','/api/inventory/file?path='+encodeURIComponent(entry.path)));
      card.append(icon,title,description,info,actions);
      if(entry.versions.length){const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent=entry.versions.length+' saved '+(entry.versions.length===1?'version':'versions');details.append(summary);for(const version of entry.versions.slice().reverse())details.append(item(new Date(version.created*1000).toLocaleString(),version.data.size+' bytes'),link('Download saved version','/api/runtime/artifacts/'+encodeURIComponent(version.id)));card.append(details);}
      $('inventory-items').append(card);
    }
    empty($('earlier-items'),data.earlier_saved_artifacts.length?'':'No earlier saved artifacts.');
    for(const record of data.earlier_saved_artifacts){const node=item(record.data.original_path||'Saved creation',record.data.size+' bytes · saved snapshot');node.append(link('Download','/api/runtime/artifacts/'+encodeURIComponent(record.id)));$('earlier-items').append(node);}
  }
  function finished(work,force=false){
    if(work.kind!=='companion')return;
    if(force||work.actor_id===friend || !friend){friend=work.actor_id;thread=work.result?.thread_id||work.conversation_id;$('consultation').hidden=false;$('consult-title').textContent='With '+(work.result?.speaker||'a companion');$('friend-reply').append(item(work.result?.speaker||'Companion',work.result?.text||work.detail||work.status));return true;}
    return false;
  }
  window.TamagiHomeViews={load,growth,inventory,finished};
  document.addEventListener('DOMContentLoaded',()=>{
    form('note-form',async()=>{await api('/runtime/notebook',{method:'POST',body:JSON.stringify({text:$('note-text').value})});$('note-text').value='';await load();});
    form('companion-form',async()=>{await api('/companions',{method:'POST',body:JSON.stringify({name:$('friend-name').value,seed:$('friend-seed').value})});$('companion-form').reset();await load();});
    form('consult-form',async()=>{
      if(!friend)throw new Error('Choose a companion first');
      const work=await api('/companions/'+encodeURIComponent(friend)+'/consult',{method:'POST',body:JSON.stringify({content:$('friend-message').value,mode:$('friend-mode').value,thread_id:thread,request_id:TamagiShell.uuid(),background:true})});
      TamagiShell.remember(work);TamagiShell.subscribe(work);thread=work.conversation_id;$('friend-message').value='';$('friend-reply').append(item('Invitation accepted','You can explore while the conversation continues.'));
    });
    form('pursuit-form',async()=>{await api('/pursuits',{method:'POST',body:JSON.stringify({action:'create',attributes:{title:$('pursuit-title').value,next_step:$('pursuit-step').value,origin:'operator invitation',commitment:false}})});$('pursuit-form').reset();await load();});
    form('backup-form',async()=>{
      notice('Gathering a checkpoint and encrypting the download…');let password=$('backup-password').value;$('backup-password').value='';
      const result=await api('/backups',{method:'POST',body:JSON.stringify({password})});password='';
      empty($('backup-download'));const a=link('Download encrypted home',result.download);a.download=result.filename;$('backup-download').append(a);notice('Your download is ready. Keep it and its password on another device.');
    });
    form('restore-form',async()=>{
      notice('Decrypting and checking the archive…');const body=new FormData();body.append('archive',$('restore-file').files[0]);body.append('password',$('restore-password').value);$('restore-password').value='';
      const result=await api('/restores/preview',{method:'POST',body});restore=result.id;$('restore-details').textContent=JSON.stringify(result,null,2);$('restore-preview').hidden=false;notice('Preview checked. Review the resident and source installation before restoring.');
    });
    $('restore-commit').onclick=()=>perform(async()=>{
      if(!$('source-stopped').checked)throw new Error('Confirm the source installation is stopped.');
      $('restore-commit').disabled=true;
      try{const result=await api('/restores/'+encodeURIComponent(restore)+'/commit',{method:'POST',body:JSON.stringify({source_stopped:true,authority:$('restore-authority').value})});$('restore-details').textContent=JSON.stringify(result,null,2);notice(result.activation_required?'Restored to a new instance. Use the displayed configuration to continue.':'Restore committed. Sign in again after the supervisor switches homes.');}
      finally{$('restore-commit').disabled=false;}
    });
    $('refresh-evidence').onclick=()=>perform(async()=>{$('evidence').textContent=JSON.stringify(await api('/runtime/events'),null,2);});
    $('load-archives').onclick=()=>perform(async()=>{
      const result=await api('/operator-archives');empty($('archives'),result.archives.length?'':'No predecessor archive is registered.');
      for(const entry of result.archives){const node=item('Preserved predecessor',entry.id+' · '+(entry.complete?'checkpoint complete':'preservation incomplete')),form=document.createElement('form'),label=document.createElement('label'),input=document.createElement('input'),submit=document.createElement('button');label.textContent='Export password · at least 12 characters';input.type='password';input.minLength=12;input.required=true;input.autocomplete='new-password';submit.textContent='Create encrypted archive';form.append(label,input,submit);form.onsubmit=event=>{event.preventDefault();perform(async()=>{submit.disabled=true;let password=input.value;input.value='';try{const result=await api('/operator-archives/'+encodeURIComponent(entry.id)+'/export',{method:'POST',body:JSON.stringify({password})});password='';node.append(link('Download predecessor archive',result.download));}finally{submit.disabled=false;}});};node.append(form);$('archives').append(node);}
    });
    $('load-grants').onclick=()=>perform(async()=>{
      const result=await api('/runtime/grants');empty($('grants'),result.tools.length?'':'No external service actions are configured.');
      for(const tool of result.tools){const node=item(tool.name,tool.scope);for(const [label,seconds]of[['Allow for one hour',3600],['Revoke',0]])node.append(button(label,async()=>{await api('/runtime/grants',{method:'POST',body:JSON.stringify({scope:tool.scope,seconds})});notice(seconds?'Delegated for one hour.':'Delegation revoked.');}));$('grants').append(node);}
    });
    $('enable-deployment').onclick=()=>perform(async()=>{await api('/deployments/settings',{method:'POST',body:JSON.stringify({enabled:true})});await load();});
    $('disable-deployment').onclick=()=>perform(async()=>{await api('/deployments/settings',{method:'POST',body:JSON.stringify({enabled:false})});await load();});
    setInterval(()=>{if(['map','journal','companions','settings'].includes(TamagiShell.view())&&!document.hidden)load().catch(()=>{});},30000);
  });
})();
