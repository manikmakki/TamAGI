from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from backend.core.household.native_registry import NativeRegistry
from backend.core.household.resident_recovery import (
    ResidentArchive, migrate_resident, restore_resident, stopped_household)
from backend.core.runtime.store import RuntimeStore

out = Path.cwd()
password = (out / 'archive-password').read_text().strip()
kinds = ('actor', 'identity', 'memory', 'conversation', 'household_projection',
         'growth_recognition', 'growth_stage', 'artifact', 'graph', 'pursuit', 'personality', 'companion')


def lived_records(path):
    with closing(sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)) as db:
        result = {}
        for kind in kinds:
            values = db.execute('SELECT * FROM records WHERE kind=? ORDER BY id', (kind,)).fetchall()
            result[kind] = {'count': len(values), 'sha256': hashlib.sha256(
                json.dumps(values, sort_keys=True).encode()).hexdigest()}
        return result


results = []
for item in json.loads((out / 'archives.json').read_text()):
    actor = item['resident_id']
    home = out / ('recovered-' + item['name'].lower())
    assert home.is_relative_to(out) and not home.exists()
    archive = ResidentArchive(out / ('staging-' + item['name'].lower()))
    try:
        preview = archive.stage(item['path'], password)
        stage = archive.stages[preview['id']]
        expected = lived_records(Path(stage['directory']) / 'runtime.sqlite3')
        files = stage['manifest']['files']

        def verify(root):
            root = Path(root)
            assert root.is_relative_to(out)
            for name, value in files.items():
                if name == 'runtime.sqlite3':
                    continue
                path = root / name
                assert path.stat().st_size == value['size'], name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256'], name
            assert lived_records(root / 'runtime.sqlite3') == expected
            with closing(sqlite3.connect('file:' + str(root / 'runtime.sqlite3') + '?mode=ro', uri=True)) as db:
                resident = json.loads(db.execute("SELECT value FROM meta WHERE key='resident_id'").fetchone()[0])
                assert resident == actor

        migrated = migrate_resident(archive, preview, home, release=item['release'],
            confirmed=True, source_stopped=True, authority='guarded')
        registry = NativeRegistry(home)
        root = Path(registry.descriptors[actor]['root'])
        assert list(registry.descriptors) == [actor]
        registry.close()
        verify(root)
        # Destructive changes are confined to the inactive migrated copy.
        identity = root / 'workspace/IDENTITY.md'
        identity.write_text('DISPOSABLE PRIVATE REGRESSION')
        store = RuntimeStore(root / 'runtime.sqlite3')
        memories = store.list('memory', actor, limit=1)
        if memories:
            memory = memories[0]
            store.put('memory', memory['id'], actor, {**memory['data'], 'content': 'DISPOSABLE MEMORY REGRESSION'})
        store.close()
        preview = archive.stage(item['path'], password)
        with stopped_household(home, damaged_resident=actor) as registry:
            restored = restore_resident(registry, archive, preview, actor,
                confirmed=True, source_stopped=True, authority='guarded')
        verify(restored['root'])
        assert identity.read_text() == 'DISPOSABLE PRIVATE REGRESSION'
        root = Path(restored['root'])
        assert root.is_relative_to(home / 'residents')
        shutil.rmtree(root)
        preview = archive.stage(item['path'], password)
        with stopped_household(home, damaged_resident=actor) as registry:
            restored = restore_resident(registry, archive, preview, actor,
                confirmed=True, source_stopped=True, authority='guarded')
        verify(restored['root'])
        registry = NativeRegistry(home)
        assert list(registry.descriptors) == [actor]
        registry.close()
        results.append({'name': item['name'], 'resident_id': actor, 'passed': True,
            'migration': True, 'private_regression_recovery': True, 'complete_private_data_loss_recovery': True,
            'verified_files': len(files) - 1, 'preserved_records': expected,
            'active_workers_started': 0, 'real_model_calls': 0})
    finally:
        archive.cleanup()
report = {'passed': True, 'source': 'encrypted checkpoints of the actual Cedar/Willow/River test household',
          'destructive_target': 'inactive disposable migrations only', 'residents': results}
(out / 'retained-data-report.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'passed': True, 'residents': [r['name'] for r in results], 'scenarios_per_resident': 3}))
