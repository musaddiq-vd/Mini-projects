from jarvis.tools import tool


@tool("undo", "Undo the last change Jarvis made (file moves/writes, new folders, saved memories).")
def undo(ctx):
    return ctx.undo.undo()
