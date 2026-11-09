"""Read-only Journal views of completed turns and their actual tool receipts."""
from collections import Counter
import hashlib


def world_activity_event(runtime, record):
    data = record['data']
    execution = data.get('execution_id')
    receipts = [r for r in runtime.store.receipts(execution, limit=10000)
                if execution and r['actor'] == runtime.actor_id and r['started'] <= record['created']]
    receipts.reverse()
    wake = {k: data.get(k) for k in ('next_wake_at', 'wake_event', 'wait_reason')}
    for receipt in reversed(receipts):
        if receipt['tool'] == 'end_turn' and receipt['status'] == 'succeeded':
            terminal = (receipt['payload'].get('result', {}).get('data') or {}).get('value') or {}
            wake = {'next_wake_at': terminal.get('wake_at'), 'wake_event': terminal.get('wake_event'),
                    'wait_reason': terminal.get('wait_reason')}
            break
    counts = Counter(r['status'] for r in receipts)
    tools = list(dict.fromkeys(r['tool'] for r in receipts))
    status = data.get('status', 'completed')
    outcome = 'Turn ended with status: ' + status.replace('_', ' ') + '.'
    if status == 'waiting' and wake['wake_event']:
        outcome = 'Chose to wait for ' + wake['wake_event'].replace('_', ' ') + '.'
    summary = [outcome, f"Recorded tool calls: {len(receipts)} ({counts['succeeded']} succeeded, {counts['failed']} failed)."]
    if tools:
        summary.append('Tools used: ' + ', '.join(tools) + '.')
    other = {k: v for k, v in counts.items() if k not in {'succeeded', 'failed'}}
    if other:
        summary.append('Other recorded outcomes: ' + ', '.join(f'{k}: {v}' for k, v in sorted(other.items())) + '.')
    evaluations = {}
    for receipt in receipts:
        if receipt['tool'] != 'runtime_change' or receipt['status'] != 'succeeded':
            continue
        value = (receipt['payload'].get('result', {}).get('data') or {}).get('value') or {}
        job_id = value.get('job_id')
        job = runtime.store.get('evaluation_job', job_id) if job_id else None
        if job and job['actor'] == runtime.actor_id:
            evaluations[job_id] = {'id': job_id, 'status': job['data']['status']}
    if evaluations:
        states = Counter(job['status'] for job in evaluations.values())
        summary.append('Evaluation jobs now: ' + ', '.join(f'{count} {state}' for state, count in sorted(states.items())) + '.')
    run = runtime.store.run(execution) if execution else None
    metadata = {'status': status, 'execution_id': execution, **wake,
        'receipts': [r['id'] for r in receipts], 'skills_used': tools,
        'tool_count': len(receipts), 'tool_outcomes': dict(counts),
        'evaluation_jobs': list(evaluations.values()),
        'activity_summary': '\n'.join(summary), 'summary_source': 'recorded_execution',
        'location': data.get('location', 'Home'), 'mood': data.get('mood', ''),
        'timezone': runtime.config.timezone}
    if run:
        metadata['duration_seconds'] = round(max(0, record['created'] - run['created']), 1)
    return {'id': 'ml-world-' + hashlib.sha256(record['id'].encode()).hexdigest()[:24],
        'type': 'world_activity', 'source': 'autonomous', 'title': f"{data.get('location', 'Home')} — {status}",
        'content': data.get('text') or '', 'timestamp': record['created'], 'metadata': metadata}


def journal_events(runtime, limit=1000):
    with runtime.store.lock:
        records = [runtime.store._record(r) for r in runtime.store.db.execute(
            "SELECT * FROM records WHERE actor=? AND kind IN ('monologue','world_turn') ORDER BY created DESC,id DESC LIMIT ?",
            (runtime.actor_id, limit * 2))]
    turns = {r['data'].get('execution_id'): r for r in records if r['kind'] == 'world_turn'}
    events, represented = [], set()
    for record in records:
        if record['kind'] != 'monologue':
            continue
        event = record['data']
        execution = event.get('metadata', {}).get('execution_id')
        if event.get('type') == 'world_activity' and execution in turns:
            projected = world_activity_event(runtime, turns[execution])
            event = {**projected, **event, 'metadata': {**event.get('metadata', {}), **projected['metadata']}}
            represented.add(execution)
        events.append(event)
    for execution, record in turns.items():
        if execution not in represented:
            events.append(world_activity_event(runtime, record))
    return sorted(events, key=lambda e: (e['timestamp'], e['id']))[-limit:]
