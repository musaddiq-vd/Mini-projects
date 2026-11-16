"""Voice tools for the agent team (see ``jarvis.agents.team``)."""

from __future__ import annotations

from jarvis.tools import tool


@tool(
    "run_agents",
    "Hand a bigger, multi-part job to Jarvis's agent team: a planner splits it into steps for a researcher (web), "
    "coder, system (PC tools) and writer (reports, e-mails). Use for requests like 'research X and write me a "
    "report', 'find the best Y and make a comparison', 'build a small app for Z and explain it'. It runs in the "
    "background with a live plan card; you'll get a notice when it's done. Not for simple one-tool requests.",
    {"goal": {"type": "string", "description": "The whole request, in the user's words"}},
    required=["goal"],
    slow=True,
)
def run_agents(ctx, goal: str, background: bool = True):
    from jarvis.agents import team

    goal = " ".join(str(goal).split())
    if len(goal) < 4:
        return {"error": "Tell me what the agents should do."}
    run = team.start(ctx, goal, background=background)
    if run is None:
        return {"error": "The agent team is already working on something. Say 'agent status' or 'cancel agents'."}
    if not background:
        snap = run.snapshot() | {"result": run.summary or run.error}
        return snap if run.status == team.DONE else snap | {"error": run.error}
    return {"run_id": run.id, "plan": "being made",
            "result": "The agent team has started; a plan card shows each step. Tell the user in one short "
                      "sentence. They can say 'stop' or 'cancel agents' to end it. Don't wait for it."}


@tool("agent_status", "What the agent team is doing now (or did last): the plan, each step's status and result.")
def agent_status(ctx):
    from jarvis.agents import team

    run = team.current()
    if run is None:
        return "The agent team hasn't run yet in this session."
    snap = run.snapshot()
    done = f"{snap['done_steps']} of {len(snap['steps'])} step(s) done"
    return snap | {"result": f"Status: {snap['status']}, {done}, {snap['tool_calls']} tool call(s), "
                             f"{int(snap['seconds'])} s."}


@tool("cancel_agents", "Stop the agent team or a running workflow right now ('cancel agents', 'stop the research').")
def cancel_agents(ctx):
    from jarvis.agents import team

    return "Stopping the agents now." if team.cancel("cancel_agents") else "Nothing is running."
