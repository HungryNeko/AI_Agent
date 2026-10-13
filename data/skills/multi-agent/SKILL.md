# Multi-Agent Work

Use this skill when a task could be split across sub-agents (`agent` tool), runs long, or touches files that several agents may use.

## When to delegate
- Delegate independent chunks: parallel research, exploring many files, long jobs (training, batch processing, crawling). Do not delegate trivial work.
- Up to 4 agents run at once per conversation (8 across the server). Give each a complete, self-contained prompt and say which report you want back.
- Before `spawn`, state the task list (`taskList`) and give every task its `taskId`. A new turn starts a fresh list, so ids like `t1` can be reused.

## wait=true or wait=false
- `wait=true` (default): blocks up to about 90 seconds and returns the results. Use it for quick tasks whose results you need right now. Agents still running at the timeout are handed over to the background.
- `wait=false`, or ending your turn: for long jobs. The agent keeps running after the turn ends or is stopped, and when it finishes the conversation is woken with a `systemEvent:` message that carries the result. Never poll in a loop.
- A foreground agent (wait=true) is killed, with its child processes, when its turn closes while it still runs. Background agents are recorded in `_background.json` and survive.

## Shared files
- Every file operation takes a read/write lock; a locked file returns an error naming the holder, so work on something else and retry.
- A whole-file `write` is refused if the file changed after you last read it: read it again, merge, write again. `replace` and `append` always work on the current text, so use them for logs and shared files.
- Best of all: give parallel agents separate files and merge them yourself, or let sub-agents only return results and write shared files from the main agent.

## Python state between calls
- Every python call gets a fresh artifact directory, so a file written there is not visible to the next call.
- Keep state in `WORKSPACE` (a predefined path string, persistent for the conversation, separate for each sub-agent), or use `save_parameter` / `load_parameter`.
- Check artifacts with the absolute paths that python results return.
