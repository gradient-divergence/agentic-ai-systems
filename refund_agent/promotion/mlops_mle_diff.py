"""Item and file differences, with prompt and contract changes named explicitly."""


def diff(incumbent, candidate):
    changes = []
    for before, after in zip(incumbent.components, candidate.components):
        for key in sorted(before.items.keys() | after.items.keys()):
            old, new = before.items.get(key), after.items.get(key)
            if old == new:
                continue
            if key == "agent_code_files":
                for path in sorted(old.keys() | new.keys()):
                    if old.get(path) != new.get(path):
                        changes.append({"component": "tools", "item": "agent_code", "path": path})
                        if path == "adaptation/prompt.py":
                            changes.append({"component": "prompt", "item": "code"})
                        if path == "contracts.py":
                            changes.append({"component": "tools", "item": "tool_contracts"})
            else:
                changes.append({"component": after.name, "item": key})
    return list({tuple(sorted(change.items())): change for change in changes}.values())
