"""
KeyMap MCP Tools — Meta-Stamp V3

Exposes KeyMap functionality as MCP (Model Context Protocol) tools so that
AI agents — including Codex, Claude, GPT-4, and any MCP-compatible client —
can programmatically query shortcut mappings and generate export scripts.

Tools:
  keymap_list_apps         → List all supported creative apps
  keymap_map_shortcuts     → Map shortcuts from one app to another
  keymap_export            → Generate a remapping script in any supported format
  keymap_get_shortcuts     → Get all shortcuts for a specific app
  keymap_search_action     → Find which shortcut performs a specific action
"""

from typing import Any
from app.services.keymap import (
    get_all_apps,
    get_shortcuts_for_app,
    map_shortcuts,
    export_mapping,
    APP_METADATA,
)


# ---------------------------------------------------------------------------
# Tool definitions (MCP schema format)
# ---------------------------------------------------------------------------

KEYMAP_MCP_TOOLS = [
    {
        "name": "keymap_list_apps",
        "description": (
            "List all creative software applications supported by the KeyMap system. "
            "Returns app IDs, names, types (DAW/NLE), vendors, and supported platforms. "
            "Use this first to discover valid app IDs for other KeyMap tools."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "keymap_map_shortcuts",
        "description": (
            "Map keyboard shortcuts from one creative application to another. "
            "Given a source app (where the user has muscle memory) and a target app "
            "(where they are currently working), returns a complete mapping of every "
            "action — showing the source shortcut, the equivalent target shortcut, "
            "and whether a direct mapping exists. "
            "Example: Map Pro Tools shortcuts onto Final Cut Pro to help an audio "
            "engineer who is learning video editing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_app": {
                    "type": "string",
                    "description": "App ID of the source application (user's existing muscle memory). E.g. 'pro_tools', 'logic_pro'",
                },
                "target_app": {
                    "type": "string",
                    "description": "App ID of the target application (where the user is working). E.g. 'final_cut_pro', 'capcut'",
                },
                "platform": {
                    "type": "string",
                    "enum": ["mac", "windows"],
                    "description": "Operating system platform. Defaults to 'mac'.",
                    "default": "mac",
                },
                "category_filter": {
                    "type": "string",
                    "description": "Optional: filter results to a specific category such as 'Transport', 'Edit', 'Tools', 'View', 'Markers', 'Tracks'.",
                },
            },
            "required": ["source_app", "target_app"],
        },
    },
    {
        "name": "keymap_export",
        "description": (
            "Generate a downloadable remapping script that physically remaps keyboard shortcuts "
            "from one app's layout to another. "
            "Supported formats: "
            "'autohotkey' (Windows .ahk script), "
            "'karabiner' (macOS Karabiner-Elements JSON), "
            "'keyboard_maestro' (macOS .kmmacros XML), "
            "'stream_deck' (Elgato Stream Deck profile JSON), "
            "'markdown' (human-readable reference table). "
            "Returns the script content as a string."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_app": {
                    "type": "string",
                    "description": "App ID of the source application.",
                },
                "target_app": {
                    "type": "string",
                    "description": "App ID of the target application.",
                },
                "platform": {
                    "type": "string",
                    "enum": ["mac", "windows"],
                    "default": "mac",
                },
                "format": {
                    "type": "string",
                    "enum": ["autohotkey", "karabiner", "keyboard_maestro", "stream_deck", "markdown"],
                    "description": "Export format for the remapping script.",
                },
            },
            "required": ["source_app", "target_app", "format"],
        },
    },
    {
        "name": "keymap_get_shortcuts",
        "description": (
            "Get all keyboard shortcuts for a specific creative application. "
            "Returns shortcuts organized by category with both Mac and Windows variants."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_id": {
                    "type": "string",
                    "description": "App ID to retrieve shortcuts for. Use keymap_list_apps to find valid IDs.",
                },
                "platform": {
                    "type": "string",
                    "enum": ["mac", "windows"],
                    "default": "mac",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter to a specific category.",
                },
            },
            "required": ["app_id"],
        },
    },
    {
        "name": "keymap_search_action",
        "description": (
            "Search for a specific action across all supported apps. "
            "Given an action name or description (e.g. 'split clip', 'add marker', 'loop'), "
            "returns which apps support it and what shortcut performs it in each app. "
            "Useful for answering questions like 'What is the equivalent of Cmd+E in Final Cut Pro?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Action name or description to search for. E.g. 'split clip', 'loop playback', 'add marker'",
                },
                "platform": {
                    "type": "string",
                    "enum": ["mac", "windows"],
                    "default": "mac",
                },
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_keymap_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """
    Dispatch a KeyMap MCP tool call and return the result.
    Called by the main MCP server handler.
    """

    if tool_name == "keymap_list_apps":
        return {
            "apps": get_all_apps(),
            "count": len(APP_METADATA),
            "tip": "Use app_id values as source_app / target_app in other keymap tools.",
        }

    elif tool_name == "keymap_map_shortcuts":
        source_app = arguments["source_app"]
        target_app = arguments["target_app"]
        platform = arguments.get("platform", "mac")
        category_filter = arguments.get("category_filter")

        if source_app not in APP_METADATA:
            return {"error": f"Unknown app: '{source_app}'. Call keymap_list_apps to see valid IDs."}
        if target_app not in APP_METADATA:
            return {"error": f"Unknown app: '{target_app}'. Call keymap_list_apps to see valid IDs."}

        mappings = map_shortcuts(source_app, target_app, platform)
        if category_filter:
            mappings = [m for m in mappings if m["category"].lower() == category_filter.lower()]

        mapped_count = sum(1 for m in mappings if m["mapped"])
        total_count = len(mappings)

        return {
            "source_app": source_app,
            "source_name": APP_METADATA[source_app]["name"],
            "target_app": target_app,
            "target_name": APP_METADATA[target_app]["name"],
            "platform": platform,
            "total_shortcuts": total_count,
            "mapped_shortcuts": mapped_count,
            "coverage_pct": round(mapped_count / total_count * 100, 1) if total_count else 0,
            "mappings": mappings,
        }

    elif tool_name == "keymap_export":
        source_app = arguments["source_app"]
        target_app = arguments["target_app"]
        platform = arguments.get("platform", "mac")
        fmt = arguments["format"]

        if source_app not in APP_METADATA:
            return {"error": f"Unknown app: '{source_app}'."}
        if target_app not in APP_METADATA:
            return {"error": f"Unknown app: '{target_app}'."}

        mappings = map_shortcuts(source_app, target_app, platform)
        source_name = APP_METADATA[source_app]["name"]
        target_name = APP_METADATA[target_app]["name"]

        try:
            content, filename, mime_type = export_mapping(
                mappings=mappings,
                source_app=source_app,
                target_app=target_app,
                source_name=source_name,
                target_name=target_name,
                format=fmt,
                platform=platform,
            )
            return {
                "filename": filename,
                "mime_type": mime_type,
                "content": content,
                "instructions": _get_install_instructions(fmt, target_name),
            }
        except ValueError as e:
            return {"error": str(e)}

    elif tool_name == "keymap_get_shortcuts":
        app_id = arguments["app_id"]
        platform = arguments.get("platform", "mac")
        category = arguments.get("category")

        if app_id not in APP_METADATA:
            return {"error": f"Unknown app: '{app_id}'."}

        shortcuts = get_shortcuts_for_app(app_id)
        shortcuts = [s for s in shortcuts if s.get(platform) is not None]
        if category:
            shortcuts = [s for s in shortcuts if s["category"].lower() == category.lower()]

        return {
            "app_id": app_id,
            "app_name": APP_METADATA[app_id]["name"],
            "platform": platform,
            "count": len(shortcuts),
            "shortcuts": shortcuts,
        }

    elif tool_name == "keymap_search_action":
        query = arguments["query"].lower()
        platform = arguments.get("platform", "mac")

        results = []
        for app_id, shortcuts in __import__(
            "app.services.keymap.shortcut_db", fromlist=["SHORTCUT_DB"]
        ).SHORTCUT_DB.items():
            for s in shortcuts:
                if query in s["label"].lower() or query in s["action_id"].lower():
                    key = s.get(platform)
                    if key:
                        results.append({
                            "app_id": app_id,
                            "app_name": APP_METADATA[app_id]["name"],
                            "action_id": s["action_id"],
                            "label": s["label"],
                            "shortcut": key,
                            "category": s["category"],
                        })

        return {
            "query": query,
            "platform": platform,
            "count": len(results),
            "results": results,
        }

    else:
        return {"error": f"Unknown KeyMap tool: '{tool_name}'"}


def _get_install_instructions(fmt: str, target_name: str) -> str:
    instructions = {
        "autohotkey": (
            f"1. Install AutoHotkey v2 from autohotkey.com\n"
            f"2. Double-click the .ahk file to activate\n"
            f"3. Script activates only when {target_name} is focused"
        ),
        "karabiner": (
            f"1. Open Karabiner-Elements → Complex Modifications → Add rule\n"
            f"2. Import the .json file\n"
            f"3. Rule activates only when {target_name} is the frontmost app"
        ),
        "keyboard_maestro": (
            f"1. Double-click the .kmmacros file\n"
            f"2. Keyboard Maestro imports the macro group automatically\n"
            f"3. Macros are scoped to {target_name}"
        ),
        "stream_deck": (
            f"1. Open Stream Deck app → Profiles → Import\n"
            f"2. Select the .streamDeckProfile file\n"
            f"3. Assign the profile to {target_name}"
        ),
        "markdown": "Print or keep open as a reference while learning the new app.",
    }
    return instructions.get(fmt, "See documentation for installation instructions.")
