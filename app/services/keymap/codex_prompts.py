"""
KeyMap Codex Prompt System — Meta-Stamp V3

Structured prompts for using OpenAI Codex / GPT-4 to:
1. Discover missing shortcut mappings (gap-filling)
2. Generate custom remapping scripts for unlisted apps
3. Explain what a shortcut does in plain language
4. Suggest workflow optimizations for creatives switching apps

These prompts are designed for use with the OpenAI API (gpt-4.1-mini or gpt-4.1-nano)
and can be called from the KeyMap API or directly from the frontend via the assistant.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# System prompt — shared context for all KeyMap Codex calls
# ---------------------------------------------------------------------------

KEYMAP_SYSTEM_PROMPT = """You are KeyMap, an expert assistant built into the Meta-Stamp platform.
Your specialty is keyboard shortcuts for professional creative software: DAWs (Pro Tools, Logic Pro),
NLEs (Final Cut Pro, Premiere Pro, CapCut, DaVinci Resolve), and hardware controllers (Stream Deck).

You help creative professionals — audio engineers, video editors, music producers — who switch between
different applications and struggle with muscle memory. Your job is to:
1. Map shortcuts from one app to their functional equivalent in another
2. Explain what a shortcut does in plain language
3. Fill gaps where no direct equivalent exists
4. Generate working remapping scripts (AutoHotkey, Karabiner, Keyboard Maestro)
5. Suggest workflow optimizations for the transition

Always be specific, accurate, and practical. When a direct equivalent doesn't exist,
suggest the closest alternative or a workaround."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_gap_fill_prompt(
    source_app: str,
    target_app: str,
    unmapped_actions: list[dict],
    platform: str = "mac",
) -> str:
    """
    Ask Codex to suggest equivalents for actions that have no direct mapping.
    """
    action_list = "\n".join(
        f"- {m['label']} (action_id: {m['action_id']}, source shortcut: {m['source_shortcut']})"
        for m in unmapped_actions[:20]  # Limit to 20 to stay within context
    )
    return f"""The following keyboard shortcuts from {source_app} have no direct equivalent in {target_app} on {platform}.

For each action, suggest:
1. The closest equivalent shortcut in {target_app} (if one exists)
2. A workaround if no direct equivalent exists
3. Whether this functionality requires a different workflow in {target_app}

Actions with no mapping:
{action_list}

Format your response as a JSON array:
[
  {{
    "action_id": "...",
    "label": "...",
    "suggested_shortcut": "...",
    "workaround": "...",
    "notes": "..."
  }}
]"""


def build_explain_shortcut_prompt(
    app_name: str,
    shortcut: str,
    action_label: str,
    context: Optional[str] = None,
) -> str:
    """
    Ask Codex to explain what a shortcut does in plain language.
    """
    ctx = f"\nContext: {context}" if context else ""
    return f"""Explain what the keyboard shortcut {shortcut} does in {app_name}.
Action name: {action_label}{ctx}

Provide:
1. A plain-language description of what this shortcut does
2. When you would use it in a typical workflow
3. Any important caveats or variations (e.g., different behavior in different modes)
4. The equivalent action in Pro Tools, Final Cut Pro, and Adobe Premiere Pro (if different)

Keep the response concise and practical — aimed at a professional creative who is learning the app."""


def build_custom_app_prompt(
    app_name: str,
    platform: str = "mac",
    categories: Optional[list[str]] = None,
) -> str:
    """
    Ask Codex to generate a shortcut database for an app not in our database.
    """
    cats = ", ".join(categories) if categories else "Transport, Edit, Tools, View, Markers, Tracks"
    return f"""Generate a comprehensive keyboard shortcut reference for {app_name} on {platform}.

Include shortcuts for these categories: {cats}

Format as a JSON array matching this schema exactly:
[
  {{
    "action_id": "snake_case_identifier",
    "label": "Human-readable action name",
    "mac": "Shortcut on Mac (or null)",
    "windows": "Shortcut on Windows (or null)",
    "category": "Category name"
  }}
]

Use this notation:
- Cmd, Shift, Opt, Ctrl for modifiers
- + to separate modifier+key (e.g. "Cmd+Shift+K")
- Spell out special keys: Space, Return, Delete, Escape, Tab, Left, Right, Up, Down
- Use null if the shortcut doesn't exist on that platform

Include at least 20 shortcuts covering the most important actions."""


def build_workflow_transition_prompt(
    source_app: str,
    target_app: str,
    user_role: str = "audio engineer",
    platform: str = "mac",
) -> str:
    """
    Ask Codex to generate a workflow transition guide for switching between apps.
    """
    return f"""Create a practical workflow transition guide for a {user_role} switching from {source_app} to {target_app} on {platform}.

Cover:
1. The 10 most important muscle-memory differences (shortcuts that are completely different)
2. The 5 biggest workflow paradigm shifts (concepts that work differently)
3. A 2-week practice plan for building new muscle memory
4. Which {source_app} shortcuts transfer directly to {target_app} (no relearning needed)
5. The 3 most common mistakes {source_app} users make in {target_app}

Format as a structured guide with clear sections. Be specific and practical."""


def build_stream_deck_layout_prompt(
    target_app: str,
    deck_size: str = "15-key",
    user_role: str = "video editor",
) -> str:
    """
    Ask Codex to suggest an optimal Stream Deck layout for a specific app.
    """
    return f"""Design an optimal Elgato Stream Deck {deck_size} layout for a {user_role} using {target_app}.

For each of the {deck_size.split('-')[0]} keys, specify:
1. The action/shortcut to assign
2. The suggested button label (max 10 chars)
3. Why this action deserves a dedicated hardware key

Prioritize:
- Actions used 20+ times per session
- Actions with complex multi-key shortcuts
- Actions that interrupt flow when missed

Format as a JSON array:
[
  {{
    "position": 0,
    "action": "Action name",
    "shortcut": "Keyboard shortcut",
    "label": "Button label",
    "reason": "Why this key deserves hardware"
  }}
]"""


# ---------------------------------------------------------------------------
# OpenAI API caller (async)
# ---------------------------------------------------------------------------

async def call_codex(
    prompt: str,
    system_prompt: str = KEYMAP_SYSTEM_PROMPT,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """
    Call the OpenAI API with a KeyMap prompt.
    Returns the response text.
    """
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error calling AI: {str(e)}"
