"""Explicit public bytes, version attribution and private import boundaries."""
import base64
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from backend.core.household.contracts import AccessDenied, Conflict, ProtocolError
from backend.core.household.native_artifacts import NativeArtifacts, local_path, verified_files
from backend.core.household.store import HouseholdStore
from backend.core.llm import LLMResponse, ToolCall
import test_household_native_app as fixture


def encoded_file(content):
    return {'data': base64.b64encode(content).decode(), 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


class PublicArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='public-artifacts-', dir='/tmp')
        self.root = Path(self.temp.name)
        self.store = HouseholdStore(self.root / 'household.sqlite3')
        self.active = True
        def active(actor, execution):
            if not self.active: raise AccessDenied('Stopped')
            return None, {'helper': False, 'actor_id': actor}, None
        self.artifacts = NativeArtifacts(SimpleNamespace(store=self.store, resolve=lambda actor: actor), SimpleNamespace(_active=active))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def publish(self, actor='Cedar', command='first', **changes):
        return self.artifacts.dispatch(actor, {'action': 'publish', 'command_id': command,
            'local_execution_id': 'active-execution', 'title': 'Dice roller', 'description': 'A shared project',
            'files': {'main.py': encoded_file(b'print(2)\n')}, **changes})

    def test_public_versions_preserve_bytes_authorship_stale_write_and_replay(self):
        first = self.publish()
        self.assertEqual(self.publish(), first)
        self.assertEqual(len(self.store.list('shared_artifact_version')), 1)
        second = self.publish('River', 'update', artifact_id=first['artifact_id'], expected=1,
                              files={'main.py': encoded_file(b'print(3)\n')})
        self.assertEqual(second['author_id'], 'River')
        self.assertEqual(second['version'], 2)
        with self.assertRaises(Conflict):
            self.publish('Willow', 'stale', artifact_id=first['artifact_id'], expected=1)
        self.assertEqual(verified_files(self.artifacts.version(first['artifact_id'], 1)['data']['files'])['main.py'], b'print(2)\n')
        self.store.close()
        self.store = HouseholdStore(self.root / 'household.sqlite3')
        self.artifacts.store = self.store
        self.assertEqual(self.artifacts.listed()[0]['version'], 2)
        self.assertEqual(verified_files(self.artifacts.version(first['artifact_id'], 1)['data']['files'])['main.py'], b'print(2)\n')

    def test_invalid_bytes_paths_and_stopped_publication_leave_no_shared_effect(self):
        for files in ({'../escape': encoded_file(b'x')}, {'a': encoded_file(b'x'), 'a/b': encoded_file(b'y')},
                      {'main.py': {**encoded_file(b'x'), 'sha256': 'bad'}}):
            with self.assertRaises(ProtocolError): self.publish(files=files)
        self.active = False
        with self.assertRaises(AccessDenied): self.publish()
        self.assertFalse(self.store.list('shared_artifact'))
        self.assertFalse(self.store.list('shared_artifact_version'))

    def test_local_publication_and_import_paths_reject_symlinks_and_escapes(self):
        workspace = self.root / 'workspace'
        workspace.mkdir()
        (workspace / 'link').symlink_to(self.root, target_is_directory=True)
        for path in ('../outside', '/absolute', 'link/household.sqlite3'):
            with self.assertRaises((AccessDenied, ProtocolError)): local_path(workspace, path)


class NativePublicArtifactTests(unittest.IsolatedAsyncioTestCase):
    resident_names = ['Cedar', 'Willow', 'River']
    asyncSetUp = fixture.NativeAppTests.asyncSetUp
    asyncTearDown = fixture.NativeAppTests.asyncTearDown
    request = fixture.NativeAppTests.request
    wait_until = fixture.NativeAppTests.wait_until
    completed = fixture.NativeAppTests.completed

    async def tool(self, actor, key, **arguments):
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall(key, 'household_artifact', arguments)]),
                                   LLMResponse(content='@Tester Finished the artifact operation.')]
        work = await self.request('POST', '/api/chat/requests', actor=actor, expected=202,
            json={'request_id': key, 'message': 'Please handle the shared artifact operation.'})
        await self.completed(work['work_id'])
        return work

    def receipts(self, actor):
        path = Path(self.registry.descriptors[actor]['root']) / 'runtime.sqlite3'
        with closing(sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)) as db:
            return [(status, json.loads(payload)) for status, payload in db.execute(
                "SELECT status,payload FROM receipts WHERE tool='household_artifact' ORDER BY started")]

    async def test_nonparticipant_imports_published_exact_bytes_without_private_file_sharing(self):
        third = list(self.registry.descriptors)[2]
        cedar = Path(self.registry.descriptors[self.first]['root']) / 'workspace'
        river = Path(self.registry.descriptors[third]['root']) / 'workspace'
        project = cedar / 'workshop/diceroller'
        project.mkdir(parents=True)
        source = b'import random\nprint(random.randint(1, 6))\n'
        (project / 'main.py').write_bytes(source)
        (project / 'README.md').write_text('Roll one die.\n')
        (cedar / 'private-note.txt').write_text('PRIVATE_NOT_SHARED')
        self.assertEqual((await self.request('GET', '/api/household/artifacts'))['artifacts'], [])
        await self.tool(self.first, 'publish', action='publish', path='workshop/diceroller', title='Dice roller')
        artifact = (await self.request('GET', '/api/household/artifacts'))['artifacts'][0]
        self.assertEqual(set(artifact['files']), {'main.py', 'README.md'})
        self.assertNotIn('data', artifact['files']['main.py'])
        (project / 'main.py').write_text('A later private draft.\n')
        await self.tool(third, 'import', action='import', artifact_id=artifact['artifact_id'], version=1, destination='workshop/shared-dice')
        self.assertEqual((river / 'workshop/shared-dice/main.py').read_bytes(), source)
        self.assertFalse((river / 'private-note.txt').exists())
        self.assertFalse((Path(self.registry.descriptors[self.second]['root']) / 'workspace/workshop/shared-dice').exists())
        self.assertEqual(self.receipts(third)[-1][0], 'succeeded')
        response = await self.http.get(f"/api/household/artifacts/{artifact['artifact_id']}/versions/1/files/main.py")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, source)
        await self.tool(third, 'reimport', action='import', artifact_id=artifact['artifact_id'], version=1, destination='workshop/shared-dice')
        self.assertTrue(self.receipts(third)[-1][1]['result']['data']['value']['already_imported'])
        (river / 'workshop/shared-dice/main.py').write_text('A deliberately edited local copy.\n')
        await self.tool(third, 'collision', action='import', artifact_id=artifact['artifact_id'], version=1, destination='workshop/shared-dice')
        self.assertEqual(self.receipts(third)[-1][0], 'failed')
        self.assertIn('deliberately edited', (river / 'workshop/shared-dice/main.py').read_text())
        await self.tool(third, 'river-version', action='publish', path='workshop/shared-dice', title='Dice roller', artifact_id=artifact['artifact_id'], expected=1)
        latest = (await self.request('GET', '/api/household/artifacts'))['artifacts'][0]
        self.assertEqual((latest['version'], latest['author_id']), (2, third))
        response = await self.http.get(f"/api/household/artifacts/{artifact['artifact_id']}/versions/1/files/main.py")
        self.assertEqual(response.content, source)
