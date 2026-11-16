"""Jarvis's coding assistant: project awareness, explanations, debugging, reviews, tests and refactoring.

- ``project``   finds the project around a file, detects its language, framework and test runner, and builds a
                light, cached map of its files and symbols.
- ``snippets``  cuts out only the code that matters (a symbol, the lines around an error) within a token budget.
- ``runner``    runs a file or the test suite with the detected runner only (never a command the model wrote),
                without a shell, inside the project, with a timeout.
- ``edits``     turns the model's find/replace edits into a diff, and applies them with a backup and undo.
- ``brain``     the prompts for the text model; file contents always go in as untrusted data.
- ``highlight`` small syntax highlighter for the code card (no extra dependency).

The tools that use all this live in ``jarvis/tools/coding.py``.
"""
