"""AI agent workflow: the model reads, translates and reviews; the app does the pixels.

* ``settings``  - the ``agent`` config section
* ``store``     - state files under ``manga_translator_work/agent/`` and the long-term memory
* ``events``    - the JSONL event log and live listeners (Agents page)
* ``page``      - one page's ``*_translations.json`` as the agent edits it
* ``images``    - numbered page images, zoomed crops and thumbnails sent to the model
* ``tools``     - the tool registry the model calls
* ``runner``    - the conversation loop (``AgentRunner``)
* ``workflow``  - preprocess -> agent -> render, as one workflow mode
"""
