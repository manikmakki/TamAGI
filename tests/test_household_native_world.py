"""Household clock semantics and actual autonomous resident worker integration."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from backend.config import TamAGIConfig
from backend.core.household.native_scheduler import NativeWorldScheduler
from backend.core.household.store import HouseholdStore
from backend.core.llm import LLMResponse, ToolCall
import test_household_native_app as fixture


class HouseholdClockTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='world-clock-', dir='/tmp')
        self.now = 1800000000.0
        self.store = HouseholdStore(Path(self.temp.name) / 'household.sqlite3', clock=lambda: self.now)
        self.config = TamAGIConfig()
        self.config.runtime.timezone = 'UTC'
        self.sleep_until, self.busy, self.selected = {}, set(), []
        async def ready(actor):
            return {'world_due': self.sleep_until.get(actor, 0) <= self.now, 'pending': actor in self.busy}
        class Host:
            def __init__(self, actor): self.actor = actor
            async def json(host, *_): return await ready(host.actor)
        def submit(actor, now):
            self.selected.append(actor)
            return {'work_id': 'tick-' + str(len(self.selected)), 'resident_id': actor}
        self.manager = SimpleNamespace(store=self.store, hosts={a: Host(a) for a in ('Cedar', 'Willow', 'River')},
                                       records=lambda: [], submit_world=submit)
        self.scheduler = NativeWorldScheduler(self.manager, self.config)

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    async def scheduled(self):
        self.now = self.scheduler.status()['next_at']
        return await self.scheduler.opportunity()

    async def test_rotation_and_one_household_opportunity_per_quarter_hour(self):
        for _ in range(4):
            await self.scheduled()
            self.assertEqual((await self.scheduler.opportunity())['status'], 'waiting')
        self.assertEqual(self.selected, ['Cedar', 'Willow', 'River', 'Cedar'])

    async def test_only_awake_resident_receives_all_four_ticks_and_no_catchup_debt(self):
        first = self.scheduler.status()['next_at']
        self.sleep_until = {'Willow': first + 3600, 'River': first + 3600}
        for _ in range(4): await self.scheduled()
        self.assertEqual(self.selected, ['Cedar'] * 4)
        await self.scheduled()
        self.assertEqual(self.selected[-1], 'Willow')
        await self.scheduled()
        self.assertEqual(self.selected[-1], 'River')

    async def test_all_resting_or_busy_skips_without_inference_or_cursor_change(self):
        self.sleep_until = {a: self.now + 7200 for a in self.manager.hosts}
        self.assertEqual((await self.scheduled())['reason'], 'no_eligible_resident')
        self.assertFalse(self.selected)
        self.assertIsNone(self.scheduler.status()['last_resident_id'])
        self.sleep_until.clear()
        self.busy.add('Cedar')
        await self.scheduled()
        self.assertEqual(self.selected, ['Willow'])

    async def test_persisted_rotation_and_late_restart_does_not_replay_missed_slots(self):
        await self.scheduled()
        previous = self.scheduler.status()['next_at']
        self.now = previous + 3 * 3600
        self.scheduler = NativeWorldScheduler(self.manager, self.config)
        await self.scheduler.opportunity()
        self.assertEqual(self.selected, ['Cedar', 'Willow'])
        self.assertGreater(self.scheduler.status()['next_at'], self.now)
        self.assertEqual((await self.scheduler.opportunity())['status'], 'waiting')


class NativeWorldTests(unittest.IsolatedAsyncioTestCase):
    resident_names = ['Cedar', 'Willow', 'River']
    asyncSetUp = fixture.NativeAppTests.asyncSetUp
    asyncTearDown = fixture.NativeAppTests.asyncTearDown
    request = fixture.NativeAppTests.request
    wait_until = fixture.NativeAppTests.wait_until

    async def tick(self, manual=True):
        work = await self.app.state.world_scheduler.opportunity(manual=manual)
        self.assertEqual(work['status'], 'queued', work)
        await self.wait_until(lambda: self.app.state.interactions.get(work['work_id'])['data']['status']
                             not in {'queued', 'running'})
        data = self.app.state.interactions.get(work['work_id'])['data']
        self.assertNotEqual(data['status'], 'failed', data)
        return data

    async def test_due_household_clock_dispatches_once_without_manual_override(self):
        scheduler = self.app.state.world_scheduler
        scheduler.config.autonomy.enabled = True
        now = scheduler.status()['next_at']
        self.registry.store.clock = lambda: now
        work = await self.tick(manual=False)
        self.assertEqual(work['resident_id'], self.first)
        self.assertTrue(work['result']['activation_started'])
        self.assertEqual((await scheduler.opportunity())['status'], 'waiting')
        self.assertEqual(len(self.registry.store.list('native_world_exchange')), 1)

    async def test_real_operator_input_interrupts_world_work_without_replaying_late_tools(self):
        self.provider.gate.clear()
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('late-create', 'contact', {
            'action': 'create', 'participants': ['operator'], 'message': 'This late tool must never run.'})])]
        world = await self.app.state.world_scheduler.opportunity(manual=True)
        await self.provider.started.wait()
        human = await self.request('POST', '/api/chat/requests', actor=self.first, expected=202,
            json={'request_id': 'operator-interruption', 'message': 'Please come talk with me.'})
        await self.wait_until(lambda: self.app.state.interactions.get(world['work_id'])['data']['status'] == 'interrupted')
        self.provider.gate.set()
        await self.wait_until(lambda: self.app.state.interactions.get(human['work_id'])['data']['status'] == 'completed')
        messages = self.registry.store.list('message')
        self.assertFalse(any(m['data']['content'] == 'This late tool must never run.' for m in messages))
        self.assertEqual(len(self.registry.store.list('conversation')), 1)
        self.assertEqual([r['input_epoch'] for r in sorted(self.private(self.first, 'interaction'), key=lambda r: r['kind'] != 'world')], [0, 1])

    def private(self, actor, kind):
        path = Path(self.registry.descriptors[actor]['root']) / 'runtime.sqlite3'
        with closing(sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)) as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM records WHERE kind=?', (kind,))]

    async def test_real_world_turn_is_private_durable_and_not_a_human_chat(self):
        work = await self.tick()
        self.assertEqual(work['resident_id'], self.first)
        self.assertTrue(work['result']['activation_started'])
        self.assertEqual(len(self.private(self.first, 'world_turn')), 1)
        self.assertFalse(self.registry.store.list('conversation'))
        self.assertFalse(self.registry.store.list('notification_intent'))
        self.assertEqual([r['input_epoch'] for r in self.private(self.first, 'interaction')], [0])
        self.assertEqual((await self.request('GET', '/api/chat/requests'))['requests'], [])
        self.assertEqual((await self.request('GET', '/r/' + self.first + '/api/chat/requests'))['requests'], [])
        self.assertIn('no new user task', str([m.content for m in self.provider.calls[0]]))
        await self.tick()
        self.assertEqual(len(self.private(self.second, 'world_turn')), 1)

    async def test_world_conversation_charges_initiator_and_recipient_keeps_next_tick(self):
        original = self.provider.chat
        async def rest_after_contact(messages, **options):
            if len(messages) > 2 and 'Autonomous opportunity;' in str(messages[1].content):
                self.provider.responses.insert(0, LLMResponse(tool_calls=[ToolCall('rest-after-contact', 'end_turn', {'status': 'resting'})]))
            return await original(messages, **options)
        self.provider.chat = rest_after_contact
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('world-contact', 'contact', {
            'action': 'create', 'participants': ['operator', self.second], 'title': 'A shared idea',
            'mentions': ['operator', self.second], 'message': '@Tester @Willow Would you like to discuss this?'})])]
        first = await self.tick()
        self.assertEqual(first['status'], 'resting', 'The initiator may rest while its sponsored recipient replies')
        await self.wait_until(lambda: all(r['data']['status'] not in {'queued', 'running'}
                                         for r in self.app.state.interactions.records()))
        exchanges = self.registry.store.list('broker_exchange')
        self.assertEqual(len(exchanges), 1)
        charges = [r['data'] for r in self.registry.store.list('broker_charge')]
        self.assertTrue(any(c['actor_id'] == self.second for c in charges))
        self.assertTrue(all(c['sponsor_id'] == self.first for c in charges))
        self.assertTrue(all(c['exchange_id'] == first['exchange_id'] for c in charges))
        self.assertEqual([r['input_epoch'] for r in self.private(self.second, 'interaction')], [0])
        unseen = await self.request('GET', '/api/household/threads?unread=1')
        self.assertEqual(unseen['unread_count'], 1)
        self.assertGreater(unseen['threads'][0]['mention_count'], 0)
        self.assertEqual((await self.request('GET', '/api/household'))['addressed_thread_count'], 1)
        self.assertTrue(all(r['data']['author'] != 'operator' for r in self.registry.store.list('message')))
        second = await self.tick()
        self.assertEqual(second['resident_id'], self.second)
        later_charges = [r['data'] for r in self.registry.store.list('broker_charge') if r['data']['exchange_id'] == second['exchange_id']]
        self.assertTrue(later_charges)
        self.assertTrue(all(c['sponsor_id'] == self.second for c in later_charges))

    async def test_chosen_rest_skips_resident_and_stop_revokes_world_execution(self):
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('rest', 'end_turn', {'status': 'resting'})])]
        await self.tick()
        await self.tick()
        await self.tick()
        self.provider.gate.clear()
        self.provider.started.clear()
        work = await self.app.state.world_scheduler.opportunity(manual=True)
        await self.provider.started.wait()
        self.assertEqual(work['resident_id'], self.second, 'Cedar is still resting')
        await self.request('POST', '/api/household/world/' + work['work_id'] + '/stop')
        self.provider.gate.set()
        await self.wait_until(lambda: all(t.done() for t in self.app.state.model._tasks.values()))
        self.assertEqual(self.app.state.interactions.get(work['work_id'])['data']['status'], 'stopped')
        self.assertFalse(self.registry.store.list('conversation'))
