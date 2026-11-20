# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Replay OpenAI configuration changes at the input events that introduced them."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
import re
from typing import Any
from typing import TYPE_CHECKING

from ...events._rewind_events import _apply_rewinds
from ...events.event import Event

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse

_KEY = 'adk_openai_configuration'
# These standard-mode models default to medium. The update item requires a
# concrete effort even when the top-level request omitted the setting.
_FIELDS = {
    'reasoning': ('effort', 'medium'),
    'text': ('format', {'type': 'text'}),
}


@dataclass
class ConfigurationHistory:
  """The changes to replay and commit for one SDK call."""

  context: InvocationContext
  metadata: dict[str, Any]
  # Keys identify copied Content objects only for the lifetime of this call.
  before: dict[int, list[dict]] = field(default_factory=dict)
  skip: set[int] = field(default_factory=set)
  update: Event | None = None

  async def commit(self, response: LlmResponse) -> None:
    if response.error_code or not response.interaction_id:
      return
    if self.update is not None:
      await self.context.session_service.append_event(
          session=self.context.session, event=self.update
      )
    response.custom_metadata = {
        **(response.custom_metadata or {}),
        _KEY: self.metadata,
    }


def prepare_configuration_history(
    request: LlmRequest, kwargs: dict[str, Any], *, implicit_summary: bool
) -> ConfigurationHistory | None:
  """Keep the first request's prefix and convert subsequent changes to items."""
  if request._event_history is None:
    return None
  model = kwargs['model']
  reasoning = kwargs.get('reasoning') or {}
  if (
      not re.fullmatch(
          r'gpt-6(?:-(?:astra|sol|luna)|\.1-sol)(?:-\d{4}-\d{2}-\d{2})?', model
      )
      or reasoning.get('mode', 'standard') != 'standard'
      or kwargs.get('truncation') == 'auto'
      or kwargs.get('context_management')
      or (kwargs.get('multi_agent') or {}).get('enabled')
      or kwargs.get('conversation')
      or kwargs.get('extra_body')
      or 'input' in kwargs
  ):
    return None

  context, contents = request._event_history()
  events = _apply_rewinds(context.session.events)
  # Resolve anchors against the selected, copied Content objects. A callback
  # may have removed or replaced one since the history builder ran.
  selected = {id(content) for content in request.contents}
  visible = {
      key: id(value) for key, value in contents.items() if id(value) in selected
  }
  owned = [
      e
      for e in events
      if e.author == context.agent.name
      and e.branch == context.branch
      and e.isolation_scope == context.isolation_scope
  ]
  records = {
      e.id: e.custom_metadata[_KEY]
      for e in owned
      if e.custom_metadata and isinstance(e.custom_metadata.get(_KEY), dict)
  }
  parent_id = kwargs.get('previous_response_id')
  parent = next(
      (
          event
          for event in reversed(owned)
          if event.interaction_id
          and (
              event.interaction_id == parent_id
              if parent_id
              else event.id in visible
          )
      ),
      None,
  )
  anchor = next(
      (
          e.id
          for e in reversed(events)
          if e.id in visible
          and e.invocation_id == context.invocation_id
          and (e.author == 'user' or e.get_function_responses())
      ),
      None,
  )
  desired = {key: kwargs[key] for key in _FIELDS if kwargs.get(key) is not None}
  if parent is None:
    # Do not infer a baseline from an external response ID or incomplete history.
    if parent_id or not anchor or len(visible) != 1:
      return None
    return ConfigurationHistory(
        context,
        {'baseline': deepcopy(desired), 'before': anchor, 'model': model},
    )

  # Responses point to the first response (the baseline) and to the last
  # update for each active field. Large schemas are stored only when changed.
  previous = records.get(parent.id, {})
  root = previous.get('root', parent.id)
  initial = records.get(root, {})
  if initial.get('model') != model or (
      not parent_id and initial['before'] not in visible
  ):
    return None
  baseline = initial['baseline']
  history = ConfigurationHistory(context, {'root': root})
  active_refs = {}
  if parent_id:
    active_refs.update(previous.get('active', {}))
    if any(ref not in records for ref in active_refs.values()):
      return None
    # Only the new turn (or new tool results) belongs after this parent.
    past_parent = False
    for event in events:
      if event.id in visible and (
          not past_parent or event.invocation_id != context.invocation_id
      ):
        history.skip.add(visible[event.id])
      if event.id == parent.id:
        past_parent = True
  else:
    by_anchor = {}
    for event_id, record in records.items():
      if (
          'item' in record
          and record.get('root') == root
          and record['before'] in visible
      ):
        by_anchor.setdefault(visible[record['before']], []).append(event_id)
    for content in request.contents:
      for event_id in by_anchor.get(id(content), ()):
        item = records[event_id]['item']
        history.before.setdefault(id(content), []).append(item)
        active_refs.update({key: event_id for key in _FIELDS if key in item})

  change = {}
  for key, (name, default) in _FIELDS.items():
    active = (baseline.get(key) or {}).get(name, default)
    if key in active_refs:
      active = records[active_refs[key]]['item'][key][name]
    requested = (desired.get(key) or {}).get(name, default)
    if requested != active:
      change[key] = {name: requested}

  if change:
    # Keep caller-supplied histories and formats the update item cannot express
    # on their existing path. Only add an update at a fresh input boundary.
    if (
        not anchor
        or visible[anchor] != id(request.contents[-1])
        or visible[anchor] in history.skip
        or (
            'text' in change
            and change['text']['format']['type'] not in ('text', 'json_schema')
        )
    ):
      return None
    item = {'type': 'configuration_update', **change}
    history.before.setdefault(visible[anchor], []).append(item)
    history.update = Event(
        author=context.agent.name,
        invocation_id=context.invocation_id,
        branch=context.branch,
        isolation_scope=context.isolation_scope,
        custom_metadata={_KEY: {'root': root, 'before': anchor, 'item': item}},
    )
    active_refs.update({key: history.update.id for key in change})
  history.metadata['active'] = active_refs
  _restore_baseline(kwargs, baseline, implicit_summary)
  return history


def _restore_baseline(
    kwargs: dict, baseline: dict, implicit_summary: bool
) -> None:
  # Restore only update-managed fields. Independently configured options such
  # as text verbosity or a reasoning summary still belong to the caller.
  for key, (name, _) in _FIELDS.items():
    original = baseline.get(key, {})
    section = dict(kwargs.get(key) or {})
    restore = [name]
    if key == 'reasoning' and implicit_summary:
      # ADK synthesizes a summary option alongside an explicit effort.
      restore.append('summary')
    for field_name in restore:
      if field_name in original:
        section[field_name] = original[field_name]
      else:
        section.pop(field_name, None)
    if section or key in baseline:
      kwargs[key] = section
    else:
      kwargs.pop(key, None)
