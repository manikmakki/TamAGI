"""Existing-thread invitations through actual workers and the single contact tool."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import unittest

import test_household_native_app as fixture
from backend.core.household.contracts import AccessDenied, Conflict
from backend.core.household.native_messaging import NativeMessaging
from backend.core.llm import LLMResponse, ToolCall


class NativeInvitationTests(unittest.IsolatedAsyncioTestCase):
    resident_names = ['Cedar', 'Willow', 'River']
    asyncSetUp = fixture.NativeAppTests.asyncSetUp
    asyncTearDown = fixture.NativeAppTests.asyncTearDown
    request = fixture.NativeAppTests.request
    wait_until = fixture.NativeAppTests.wait_until
    completed = fixture.NativeAppTests.completed

    def private_records(self, actor, kind):
        path = Path(self.registry.descriptors[actor]['root']) / 'runtime.sqlite3'
        with closing(sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)) as db:
            return [(identity, json.loads(payload)) for identity, payload in db.execute(
                'SELECT id,payload FROM records WHERE kind=? ORDER BY created,id', (kind,))]

    async def private_thread(self, with_image=False):
        image = ({'image_data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN2kAAAAASUVORK5CYII=',
                  'image_media_type': 'image/png'} if with_image else {})
        work = await self.request('POST', '/api/chat/requests', actor=self.second, expected=202,
            json={'request_id': 'original-thread', 'message': 'A quiet conversation with Willow.', **image})
        await self.completed(work['work_id'])
        thread = work['conversation_id']
        work = await self.request('POST', '/api/chat/requests', actor=self.second, expected=202,
            json={'request_id': 'private-history', 'conversation_id': thread, 'message': 'PRIVATE_HISTORY_X9: previous private discussion.'})
        await self.completed(work['work_id'])
        return thread

    def invite(self):
        return LLMResponse(tool_calls=[ToolCall('invite-cedar', 'contact', {
            'action': 'invite', 'participants': [self.first], 'message': 'Cedar, would you join this conversation?'})])

    def answer(self, decision):
        return LLMResponse(tool_calls=[ToolCall('answer-invitation', 'contact', {'action': 'respond', 'decision': decision})])

    async def submit_invite(self, thread):
        return await self.request('POST', '/api/chat/requests', actor=self.second, expected=202,
            json={'request_id': 'invite-current-thread', 'conversation_id': thread, 'message': 'Please invite Cedar into this thread.'})

    async def test_cedar_invites_willow_with_redundant_mentions_in_existing_thread(self):
        opened = await self.request('POST', '/api/chat/requests', actor=self.first, expected=202,
            json={'request_id': 'cedar-thread', 'message': 'Good morning Cedar.'})
        await self.completed(opened['work_id'])
        thread = opened['conversation_id']
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('invite-willow', 'contact', {
            'action': 'invite', 'conversation_id': thread, 'participants': [self.second],
            'mentions': [self.second], 'message': 'Hey Willow! Want to join us for a bit?'})]),
            LLMResponse(content='@Tester Invitation sent.'), self.answer('accept'),
            LLMResponse(content='@Tester Willow joined this conversation.')]
        work = await self.request('POST', '/api/chat/requests', actor=self.first, expected=202,
            json={'request_id': 'invite-willow', 'conversation_id': thread, 'message': 'Please invite Willow here.'})
        await self.completed(work['work_id'])
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertEqual(history['participants'], ['operator', self.first, self.second])
        self.assertEqual(history['invitations'][0]['status'], 'joined')
        self.assertEqual(len(self.registry.store.list('conversation')), 1)
        notice = next(r['data'] for r in self.registry.store.list('message')
                      if r['data']['content'] == 'Hey Willow! Want to join us for a bit?')
        self.assertEqual(notice['addressees'], [], 'An unjoined recipient receives a card, not a normal reply offer')
        self.assertTrue(notice['explicit_mentions'], 'The notice must not fall back to broadcasting reply offers')
        self.assertEqual(history['messages'][-1]['metadata']['author_id'], self.second)

    async def test_invalid_invite_mentions_give_repair_guidance_without_side_effects(self):
        thread = await self.private_thread()
        third = list(self.registry.descriptors)[2]
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('invalid-invite', 'contact', {
            'action': 'invite', 'participants': [self.first], 'mentions': [third],
            'message': 'Please join us.'})]), self.invite(),
            LLMResponse(content='@Tester Corrected the invitation.'), self.answer('accept'),
            LLMResponse(content='@Tester Joined.')]
        original = self.provider.chat
        repairs = []
        async def inspect(messages, **options):
            if 'For invite, mentions may only repeat IDs in participants' in str([m.content for m in messages]):
                if not repairs:
                    self.assertFalse(self.registry.store.list('conversation_invitation'))
                    self.assertFalse(self.registry.store.list('native_invitation_offer'))
                repairs.append(messages)
            return await original(messages, **options)
        self.provider.chat = inspect
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        self.assertTrue(repairs, 'The real worker must pass specific argument guidance back to its model')
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertEqual(history['participants'], ['operator', self.second, self.first])
        self.assertNotIn(third, history['participants'])
        self.assertEqual(len(history['invitations']), 1)
        self.assertEqual(len(self.registry.store.list('conversation')), 1)

    async def test_accept_keeps_thread_and_shares_history_only_after_actual_choice(self):
        thread = await self.private_thread(with_image=True)
        initial_calls = len(self.provider.calls)
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation sent.'),
                                   self.answer('accept'), LLMResponse(content='@Tester I joined this existing conversation.')]
        observed = []
        original = self.provider.chat
        async def inspect(messages, **options):
            if 'Household invitation decision.' in str(messages[0].content):
                observed.append(messages)
                self.assertNotIn('PRIVATE_HISTORY_X9', str([m.content for m in messages]))
                self.assertIn('FULL prior history', messages[0].content)
                self.assertIn('Cedar, would you join', messages[-1].content)
                with self.assertRaises(AccessDenied):
                    self.app.state.interactions.conversations.history(self.first, thread)
                await self.request('GET', '/_native/conversations/' + thread, expected=403,
                                   headers={'X-Tamagi-Worker': self.app.state.hosts[self.first].token})
                attachment_id = self.registry.store.list('conversation_attachment')[0]['id']
                grant = next(r for r in self.registry.store.list('native_grant')
                             if r['data']['resident_id'] == self.first and r['data'].get('kind') == 'invitation')
                await self.request('GET', f'/_native/conversations/{thread}/attachments/{attachment_id}?grant_id={grant["id"]}',
                    expected=403, headers={'X-Tamagi-Worker': self.app.state.hosts[self.first].token})
                tools = [t['function']['name'] for t in options.get('tools', [])]
                self.assertEqual(set(tools), {'contact', 'household_contacts', 'end_turn'})
                self.assertNotIn('answer_invitation', tools)
            return await original(messages, **options)
        self.provider.chat = inspect
        work = await self.submit_invite(thread)
        done = await self.completed(work['work_id'])
        self.assertEqual(len(observed), 1)
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertEqual(history['participants'], ['operator', self.second, self.first])
        self.assertEqual(len(self.registry.store.list('conversation')), 1, 'Invitation must not fork the private thread')
        self.assertEqual(history['invitations'][0]['status'], 'joined')
        self.assertEqual([m['content'] for m in history['messages'] if m['role'] == 'user'],
                         ['A quiet conversation with Willow.', 'PRIVATE_HISTORY_X9: previous private discussion.', 'Please invite Cedar into this thread.'])
        self.assertEqual(history['messages'][-1]['metadata']['author_id'], self.first)
        self.assertIn('PRIVATE_HISTORY_X9', str([m.content for m in self.provider.calls[-1]]))
        self.assertTrue(any(isinstance(m.content, list) for m in self.provider.calls[-1]), 'The joined reply receives the prior image')
        self.assertEqual(len(self.provider.calls) - initial_calls, 4)
        exchange = done['work']['exchange_id']
        grants = [r['data'] for r in self.registry.store.list('native_grant') if r['data']['exchange_id'] == exchange]
        self.assertEqual(len(grants), 3)
        self.assertTrue(any(g.get('kind') == 'invitation' for g in grants))
        self.assertTrue(all(not g['first_response'] for g in grants if g['resident_id'] == self.first))
        self.assertEqual(len(self.registry.store.list('broker_exchange')), 3, 'Only the three actual user submissions open exchanges')
        inputs = self.private_records(self.first, 'interaction')
        self.assertEqual([data['input_epoch'] for _, data in inputs], [0, 0], 'Invitation and reply never simulate human input')
        boxes = self.private_records(self.first, 'household_outbox')
        decision = next(data['payload'] for _, data in boxes if data.get('payload', {}).get('decision') == 'accept')
        messaging = NativeMessaging(self.registry, self.app.state.model, self.app.state.interactions)
        before = len(self.registry.store.events())
        replay = messaging.contact(self.first, decision)
        self.assertTrue(replay['joined'])
        self.assertEqual(len(self.registry.store.events()), before, 'Lost acknowledgment replay adds no membership, work or message')
        with self.assertRaises(Conflict):
            messaging.contact(self.first, {**decision, 'decision': 'decline'})
        print('INVITATION_FLOW_PROOF ' + json.dumps({
            'fixture': 'actual native worker processes; injected provider',
            'original_conversation_id': thread,
            'inviter': 'Willow', 'invitee': 'Cedar',
            'delivered_card': json.loads(observed[0][-1].content.split('\n', 1)[1]),
            'pre_join_transcript_access': 'denied', 'pre_join_history_token_visible': False,
            'actual_response_action': decision['action'], 'actual_response_decision': decision['decision'],
            'reply_conversation_id': history['id'], 'post_join_history_token_visible': True,
            'post_join_participants': ['Operator', 'Willow', 'Cedar'],
            'provider_calls_including_decision_and_reply': 4,
            'exchange_ids': sorted({g['exchange_id'] for g in grants}),
            'invitee_input_epochs': [data['input_epoch'] for _, data in inputs],
            'reply_messages_by_invitee': sum(m['metadata']['author_id'] == self.first for m in history['messages']),
            'total_threads': len(self.registry.store.list('conversation')),
            'decision_replay_added_events': len(self.registry.store.events()) - before,
        }, sort_keys=True))

    async def test_decline_is_persisted_without_membership_or_transcript_delivery(self):
        thread = await self.private_thread()
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation sent.'), self.answer('decline')]
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertEqual(history['participants'], ['operator', self.second])
        self.assertEqual(history['invitations'][0]['status'], 'declined')
        self.assertFalse(self.private_records(self.first, 'household_projection'))
        self.assertNotIn('PRIVATE_HISTORY_X9', str([m.content for m in self.provider.calls[-1]]))
        with self.assertRaises(AccessDenied):
            self.app.state.interactions.conversations.history(self.first, thread)
        self.assertEqual(len(self.registry.store.list('conversation')), 1)

    async def test_plain_text_does_not_accept_and_browsing_does_not_retry_decision(self):
        thread = await self.private_thread()
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation sent.'),
                                   LLMResponse(content='Yes, I would love to join!')]
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        calls = len(self.provider.calls)
        for _ in range(3):
            history = await self.request('GET', '/api/conversations/' + thread)
            await self.request('GET', '/api/household')
        await asyncio.sleep(2.2)
        self.assertEqual(len(self.provider.calls), calls)
        self.assertEqual(history['invitations'][0]['status'], 'pending')
        self.assertNotIn(self.first, history['participants'])
        self.assertEqual(self.registry.store.list('native_invitation_offer')[0]['data']['status'], 'silent')

    async def test_decision_tool_scope_blocks_files_and_creating_another_thread(self):
        thread = await self.private_thread()
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation sent.'),
            LLMResponse(tool_calls=[ToolCall('blocked-write', 'write', {'path': 'workshop/forbidden.txt', 'content': 'bad'}),
                                   ToolCall('blocked-create', 'contact', {'action': 'create', 'participants': ['operator'], 'message': 'Wrong path'})]),
            self.answer('accept'), LLMResponse(content='@Tester Joined after choosing.')]
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        self.assertFalse((Path(self.registry.descriptors[self.first]['root']) / 'workspace/workshop/forbidden.txt').exists())
        self.assertEqual(len(self.registry.store.list('conversation')), 1)
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertIn(self.first, history['participants'])
        self.assertIn('Tool is not available', str([m.content for m in self.provider.calls[-2]]))

    async def test_stop_revokes_pending_decision_and_prevents_late_join(self):
        thread = await self.private_thread()
        started, gate = asyncio.Event(), asyncio.Event()
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation sent.'), self.answer('accept')]
        original = self.provider.chat
        async def hold(messages, **options):
            if 'Household invitation decision.' in str(messages[0].content):
                started.set()
                await gate.wait()
            return await original(messages, **options)
        self.provider.chat = hold
        work = await self.submit_invite(thread)
        try:
            await asyncio.wait_for(started.wait(), 15)
            await self.request('POST', f"/api/chat/requests/{work['work_id']}/stop")
        finally:
            gate.set()
        await self.wait_until(lambda: all(t.done() for t in self.app.state.model._tasks.values()))
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertNotIn(self.first, history['participants'])
        self.assertEqual(history['invitations'][0]['status'], 'pending')
        self.assertEqual(self.registry.store.list('native_invitation_offer')[0]['data']['status'], 'closed')

    async def test_inviting_many_has_independent_choices_and_no_extra_thread(self):
        thread = await self.private_thread()
        third = list(self.registry.descriptors)[2]
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('invite-many', 'contact', {
            'action': 'invite', 'participants': [self.first, third], 'message': 'Cedar and River, would you join us?'})]),
            LLMResponse(content='@Tester Both invitations are pending.'), self.answer('accept'), self.answer('decline'),
            LLMResponse(content='@Tester Cedar is here.')]
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertEqual(history['participants'], ['operator', self.second, self.first])
        self.assertEqual({i['participant_id']: i['status'] for i in history['invitations']}, {self.first: 'joined', third: 'declined'})
        self.assertEqual(len(self.registry.store.list('conversation')), 1)
        self.assertFalse(self.private_records(third, 'household_projection'))

    async def test_operator_participants_control_delivers_same_decision_flow(self):
        thread = await self.private_thread()
        self.provider.responses = [self.answer('accept'), LLMResponse(content='@Tester Cedar joined your thread.')]
        before = len([r for r in self.registry.store.list('message') if r['data']['author'] == 'operator'])
        response = await self.request('POST', f'/api/household/threads/{thread}/participants',
            json={'request_id': 'operator-invite', 'participants': [self.second, self.first]})
        self.assertEqual(response['conversation_id'], thread)
        await self.wait_until(lambda: self.registry.store.list('conversation_invitation')[0]['data']['status'] == 'joined')
        await self.wait_until(lambda: not any(r['data']['status'] in {'queued', 'running'} for r in self.app.state.interactions.records()))
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertIn(self.first, history['participants'])
        self.assertEqual(len([r for r in self.registry.store.list('message') if r['data']['author'] == 'operator']), before)
        self.assertEqual([data['input_epoch'] for _, data in self.private_records(self.first, 'interaction')], [0, 0])

    async def test_chosen_wait_defers_invitation_until_wake_without_new_allowance(self):
        self.provider.responses = [LLMResponse(tool_calls=[ToolCall('choose-wait', 'end_turn', {'status': 'waiting', 'wake_event': 'notebook_changed'})])]
        rest = await self.request('POST', '/api/chat/requests', actor=self.first, expected=202,
                                 json={'request_id': 'cedar-waits', 'message': 'Choose whether to wait.'})
        await self.completed(rest['work_id'])
        thread = await self.private_thread()
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation is waiting for Cedar.')]
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        calls = len(self.provider.calls)
        self.assertEqual(self.registry.store.list('native_invitation_offer')[0]['data']['status'], 'deferred')
        self.provider.responses = [self.answer('accept'), LLMResponse(content='@Tester I joined after waking.')]
        await self.request('POST', '/api/runtime/notebook', json={'text': 'A shared notebook event.', 'request_id': 'wake-invite'})
        await self.wait_until(lambda: self.registry.store.list('conversation_invitation')[0]['data']['status'] == 'joined')
        await self.wait_until(lambda: not any(r['data']['status'] in {'queued', 'running'} for r in self.app.state.interactions.records()))
        self.assertEqual(len(self.provider.calls), calls + 2)
        invocations = [r for r in self.registry.store.list('native_grant') if r['data']['resident_id'] == self.first and r['data'].get('invitation_id')]
        self.assertEqual({r['data']['exchange_id'] for r in invocations}, {work['exchange_id']})
        self.assertEqual(len(self.registry.store.list('broker_exchange')), 4)

    async def test_unanswered_decision_is_capped_without_forcing_membership(self):
        thread = await self.private_thread()
        blocked = LLMResponse(tool_calls=[ToolCall('try-create', 'contact', {
            'action': 'create', 'participants': ['operator'], 'message': 'Not an invitation answer.'})])
        self.provider.responses = [self.invite(), LLMResponse(content='@Tester Invitation sent.'), *[blocked for _ in range(4)]]
        calls = len(self.provider.calls)
        work = await self.submit_invite(thread)
        await self.completed(work['work_id'])
        self.assertEqual(len(self.provider.calls) - calls, 6)
        self.assertEqual(self.registry.store.list('native_invitation_offer')[0]['data']['status'], 'exhausted')
        history = await self.request('GET', '/api/conversations/' + thread)
        self.assertNotIn(self.first, history['participants'])
        self.assertEqual(history['invitations'][0]['status'], 'pending')
        self.assertEqual(len(self.registry.store.list('conversation')), 1)
