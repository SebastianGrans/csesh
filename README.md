# csesh

A terminal browser for finding all your Claude and Copilot session.

Claude and Copilot sessions are bound to a specific folder/workspace. If you have a lot of worktrees
that can make it very hard to find an older session.

## Requirements

- [uv](https://docs.astral.sh/uv/). The script declares its own dependencies (Python 3.13+, Textual,
  RapidFuzz), so uv fetches them on first run.
- On Wayland: `wl-copy` (from `wl-clipboard`) for copying session IDs.
- `less` for reading a full transcript.

## Install

Assuming `~/.local/bin` is in your `PATH`, you can install it with these commands:

```sh
curl -o ~/.local/bin/csesh https://raw.githubusercontent.com/SebastianGrans/csesh/main/csesh.py
chmod +x ~/.local/bin/csesh
```

Or clone the repo and symlink it into your `PATH`: `ln -s "$PWD/csesh.py" ~/.local/bin/csesh`.