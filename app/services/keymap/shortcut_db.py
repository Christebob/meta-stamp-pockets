"""
KeyMap Shortcut Database — Meta-Stamp V3

Comprehensive shortcut database for professional creative applications.
Each shortcut is tagged with a semantic action_id that enables cross-app mapping.

Apps covered:
  - Pro Tools (Mac + Windows)
  - Final Cut Pro (Mac)
  - CapCut (Mac + Windows)
  - Adobe Premiere Pro (Mac + Windows)
  - Logic Pro X (Mac)
  - DaVinci Resolve (Mac + Windows)
  - Avid Media Composer (Mac + Windows)
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

ShortcutEntry = dict  # { action_id, label, mac, windows, category }

# ---------------------------------------------------------------------------
# Shortcut Database
# ---------------------------------------------------------------------------

SHORTCUT_DB: dict[str, list[ShortcutEntry]] = {

    # -----------------------------------------------------------------------
    # PRO TOOLS
    # -----------------------------------------------------------------------
    "pro_tools": [
        # Transport
        {"action_id": "play_stop",         "label": "Play / Stop",                 "mac": "Space",                    "windows": "Space",                    "category": "Transport"},
        {"action_id": "record",            "label": "Record",                      "mac": "Cmd+Space",                "windows": "Ctrl+Space",               "category": "Transport"},
        {"action_id": "rewind",            "label": "Rewind",                      "mac": "Num1",                     "windows": "Num1",                     "category": "Transport"},
        {"action_id": "fast_forward",      "label": "Fast Forward",                "mac": "Num2",                     "windows": "Num2",                     "category": "Transport"},
        {"action_id": "go_to_start",       "label": "Go to Start",                 "mac": "Return",                   "windows": "Enter",                    "category": "Transport"},
        {"action_id": "loop_playback",     "label": "Loop Playback",               "mac": "Cmd+Shift+L",              "windows": "Ctrl+Shift+L",             "category": "Transport"},
        # Edit
        {"action_id": "undo",              "label": "Undo",                        "mac": "Cmd+Z",                    "windows": "Ctrl+Z",                   "category": "Edit"},
        {"action_id": "redo",              "label": "Redo",                        "mac": "Cmd+Shift+Z",              "windows": "Ctrl+Shift+Z",             "category": "Edit"},
        {"action_id": "cut",               "label": "Cut",                         "mac": "Cmd+X",                    "windows": "Ctrl+X",                   "category": "Edit"},
        {"action_id": "copy",              "label": "Copy",                        "mac": "Cmd+C",                    "windows": "Ctrl+C",                   "category": "Edit"},
        {"action_id": "paste",             "label": "Paste",                       "mac": "Cmd+V",                    "windows": "Ctrl+V",                   "category": "Edit"},
        {"action_id": "select_all",        "label": "Select All",                  "mac": "Cmd+A",                    "windows": "Ctrl+A",                   "category": "Edit"},
        {"action_id": "split_clip",        "label": "Split / Separate Clip",       "mac": "Cmd+E",                    "windows": "Ctrl+E",                   "category": "Edit"},
        {"action_id": "trim_to_selection", "label": "Trim to Selection",           "mac": "Cmd+T",                    "windows": "Ctrl+T",                   "category": "Edit"},
        {"action_id": "nudge_left",        "label": "Nudge Left",                  "mac": "Comma",                    "windows": "Comma",                    "category": "Edit"},
        {"action_id": "nudge_right",       "label": "Nudge Right",                 "mac": "Period",                   "windows": "Period",                   "category": "Edit"},
        {"action_id": "consolidate_clip",  "label": "Consolidate Clip",            "mac": "Cmd+Shift+3",              "windows": "Ctrl+Shift+3",             "category": "Edit"},
        # Tools
        {"action_id": "tool_selector",     "label": "Selector Tool",               "mac": "F7",                       "windows": "F7",                       "category": "Tools"},
        {"action_id": "tool_grabber",      "label": "Grabber Tool",                "mac": "F8",                       "windows": "F8",                       "category": "Tools"},
        {"action_id": "tool_trimmer",      "label": "Trimmer Tool",                "mac": "F6",                       "windows": "F6",                       "category": "Tools"},
        {"action_id": "tool_pencil",       "label": "Pencil Tool",                 "mac": "F10",                      "windows": "F10",                      "category": "Tools"},
        {"action_id": "tool_zoom",         "label": "Zoom Tool",                   "mac": "Cmd+`",                    "windows": "Ctrl+`",                   "category": "Tools"},
        # View
        {"action_id": "zoom_in",           "label": "Zoom In (Horizontal)",        "mac": "Cmd+]",                    "windows": "Ctrl+]",                   "category": "View"},
        {"action_id": "zoom_out",          "label": "Zoom Out (Horizontal)",       "mac": "Cmd+[",                    "windows": "Ctrl+[",                   "category": "View"},
        {"action_id": "zoom_to_selection", "label": "Zoom to Selection",           "mac": "Cmd+Opt+F",                "windows": "Ctrl+Alt+F",               "category": "View"},
        {"action_id": "fit_tracks",        "label": "Fit Tracks in Window",        "mac": "Cmd+Opt+W",                "windows": "Ctrl+Alt+W",               "category": "View"},
        # Tracks
        {"action_id": "new_track",         "label": "New Track",                   "mac": "Cmd+Shift+N",              "windows": "Ctrl+Shift+N",             "category": "Tracks"},
        {"action_id": "group_tracks",      "label": "Group Tracks",                "mac": "Cmd+G",                    "windows": "Ctrl+G",                   "category": "Tracks"},
        {"action_id": "mute_track",        "label": "Mute Track",                  "mac": "Cmd+M",                    "windows": "Ctrl+M",                   "category": "Tracks"},
        {"action_id": "solo_track",        "label": "Solo Track",                  "mac": "Cmd+Opt+M",                "windows": "Ctrl+Alt+M",               "category": "Tracks"},
        # Markers
        {"action_id": "add_marker",        "label": "Add Marker",                  "mac": "Enter (Num)",              "windows": "Enter (Num)",              "category": "Markers"},
        {"action_id": "next_marker",       "label": "Next Marker",                 "mac": "Num9",                     "windows": "Num9",                     "category": "Markers"},
        {"action_id": "prev_marker",       "label": "Previous Marker",             "mac": "Num8",                     "windows": "Num8",                     "category": "Markers"},
        # Mix
        {"action_id": "show_mix_window",   "label": "Show Mix Window",             "mac": "Cmd+=",                    "windows": "Ctrl+=",                   "category": "Mix"},
        {"action_id": "show_edit_window",  "label": "Show Edit Window",            "mac": "Cmd+Shift+=",              "windows": "Ctrl+Shift+=",             "category": "Mix"},
    ],

    # -----------------------------------------------------------------------
    # FINAL CUT PRO
    # -----------------------------------------------------------------------
    "final_cut_pro": [
        # Transport
        {"action_id": "play_stop",         "label": "Play / Pause",                "mac": "Space",                    "windows": None,                       "category": "Transport"},
        {"action_id": "record",            "label": "Record Voiceover",            "mac": "Opt+Cmd+8",                "windows": None,                       "category": "Transport"},
        {"action_id": "go_to_start",       "label": "Go to Beginning",             "mac": "Fn+Left / Home",           "windows": None,                       "category": "Transport"},
        {"action_id": "go_to_end",         "label": "Go to End",                   "mac": "Fn+Right / End",           "windows": None,                       "category": "Transport"},
        {"action_id": "play_from_start",   "label": "Play from Beginning",         "mac": "Shift+Cmd+J",              "windows": None,                       "category": "Transport"},
        {"action_id": "loop_playback",     "label": "Loop Playback",               "mac": "Cmd+L",                    "windows": None,                       "category": "Transport"},
        {"action_id": "play_selection",    "label": "Play Selection",              "mac": "/",                        "windows": None,                       "category": "Transport"},
        # Edit
        {"action_id": "undo",              "label": "Undo",                        "mac": "Cmd+Z",                    "windows": None,                       "category": "Edit"},
        {"action_id": "redo",              "label": "Redo",                        "mac": "Cmd+Shift+Z",              "windows": None,                       "category": "Edit"},
        {"action_id": "cut",               "label": "Cut",                         "mac": "Cmd+X",                    "windows": None,                       "category": "Edit"},
        {"action_id": "copy",              "label": "Copy",                        "mac": "Cmd+C",                    "windows": None,                       "category": "Edit"},
        {"action_id": "paste",             "label": "Paste",                       "mac": "Cmd+V",                    "windows": None,                       "category": "Edit"},
        {"action_id": "select_all",        "label": "Select All",                  "mac": "Cmd+A",                    "windows": None,                       "category": "Edit"},
        {"action_id": "split_clip",        "label": "Blade / Split Clip",          "mac": "Cmd+B",                    "windows": None,                       "category": "Edit"},
        {"action_id": "trim_to_selection", "label": "Trim to Selection",           "mac": "Opt+\\",                   "windows": None,                       "category": "Edit"},
        {"action_id": "nudge_left",        "label": "Nudge Left 1 Frame",          "mac": "Comma",                    "windows": None,                       "category": "Edit"},
        {"action_id": "nudge_right",       "label": "Nudge Right 1 Frame",         "mac": "Period",                   "windows": None,                       "category": "Edit"},
        {"action_id": "ripple_delete",     "label": "Ripple Delete",               "mac": "Shift+Delete",             "windows": None,                       "category": "Edit"},
        {"action_id": "lift_from_primary", "label": "Lift from Primary Storyline", "mac": "Opt+Cmd+Up",               "windows": None,                       "category": "Edit"},
        {"action_id": "overwrite",         "label": "Overwrite to Primary",        "mac": "D",                        "windows": None,                       "category": "Edit"},
        {"action_id": "connect_clip",      "label": "Connect Clip",                "mac": "Q",                        "windows": None,                       "category": "Edit"},
        {"action_id": "append_to_storyline","label": "Append to Storyline",        "mac": "E",                        "windows": None,                       "category": "Edit"},
        {"action_id": "insert_clip",       "label": "Insert Clip",                 "mac": "W",                        "windows": None,                       "category": "Edit"},
        # Tools
        {"action_id": "tool_selector",     "label": "Select Tool",                 "mac": "A",                        "windows": None,                       "category": "Tools"},
        {"action_id": "tool_trimmer",      "label": "Trim Tool",                   "mac": "T",                        "windows": None,                       "category": "Tools"},
        {"action_id": "tool_blade",        "label": "Blade Tool",                  "mac": "B",                        "windows": None,                       "category": "Tools"},
        {"action_id": "tool_zoom",         "label": "Zoom Tool",                   "mac": "Z",                        "windows": None,                       "category": "Tools"},
        {"action_id": "tool_hand",         "label": "Hand Tool",                   "mac": "H",                        "windows": None,                       "category": "Tools"},
        # View
        {"action_id": "zoom_in",           "label": "Zoom In",                     "mac": "Cmd+=",                    "windows": None,                       "category": "View"},
        {"action_id": "zoom_out",          "label": "Zoom Out",                    "mac": "Cmd+-",                    "windows": None,                       "category": "View"},
        {"action_id": "zoom_to_selection", "label": "Zoom to Selection",           "mac": "Shift+Z",                  "windows": None,                       "category": "View"},
        {"action_id": "fit_tracks",        "label": "Fit All in Window",           "mac": "Shift+Z",                  "windows": None,                       "category": "View"},
        # Markers
        {"action_id": "add_marker",        "label": "Add Marker",                  "mac": "M",                        "windows": None,                       "category": "Markers"},
        {"action_id": "next_marker",       "label": "Next Marker",                 "mac": "Ctrl+`",                   "windows": None,                       "category": "Markers"},
        {"action_id": "prev_marker",       "label": "Previous Marker",             "mac": "Ctrl+Shift+`",             "windows": None,                       "category": "Markers"},
        # Tracks / Timeline
        {"action_id": "new_track",         "label": "New Storyline",               "mac": "Opt+Cmd+Y",                "windows": None,                       "category": "Tracks"},
        {"action_id": "mute_track",        "label": "Mute / Unmute Clip",          "mac": "V",                        "windows": None,                       "category": "Tracks"},
        {"action_id": "show_audio_lanes",  "label": "Show Audio Lanes",            "mac": "Ctrl+Opt+S",               "windows": None,                       "category": "Tracks"},
        # Color
        {"action_id": "show_color_board",  "label": "Show Color Board",            "mac": "Cmd+6",                    "windows": None,                       "category": "Color"},
        {"action_id": "balance_color",     "label": "Balance Color",               "mac": "Opt+Cmd+B",                "windows": None,                       "category": "Color"},
    ],

    # -----------------------------------------------------------------------
    # CAPCUT (Desktop)
    # -----------------------------------------------------------------------
    "capcut": [
        # Transport
        {"action_id": "play_stop",         "label": "Play / Pause",                "mac": "Space",                    "windows": "Space",                    "category": "Transport"},
        {"action_id": "go_to_start",       "label": "Go to Start",                 "mac": "Home",                     "windows": "Home",                     "category": "Transport"},
        {"action_id": "go_to_end",         "label": "Go to End",                   "mac": "End",                      "windows": "End",                      "category": "Transport"},
        # Edit
        {"action_id": "undo",              "label": "Undo",                        "mac": "Cmd+Z",                    "windows": "Ctrl+Z",                   "category": "Edit"},
        {"action_id": "redo",              "label": "Redo",                        "mac": "Cmd+Shift+Z",              "windows": "Ctrl+Shift+Z",             "category": "Edit"},
        {"action_id": "cut",               "label": "Cut",                         "mac": "Cmd+X",                    "windows": "Ctrl+X",                   "category": "Edit"},
        {"action_id": "copy",              "label": "Copy",                        "mac": "Cmd+C",                    "windows": "Ctrl+C",                   "category": "Edit"},
        {"action_id": "paste",             "label": "Paste",                       "mac": "Cmd+V",                    "windows": "Ctrl+V",                   "category": "Edit"},
        {"action_id": "select_all",        "label": "Select All",                  "mac": "Cmd+A",                    "windows": "Ctrl+A",                   "category": "Edit"},
        {"action_id": "split_clip",        "label": "Split Clip",                  "mac": "Cmd+B",                    "windows": "Ctrl+B",                   "category": "Edit"},
        {"action_id": "delete_clip",       "label": "Delete Clip",                 "mac": "Delete",                   "windows": "Delete",                   "category": "Edit"},
        {"action_id": "nudge_left",        "label": "Move Left 1 Frame",           "mac": "Left",                     "windows": "Left",                     "category": "Edit"},
        {"action_id": "nudge_right",       "label": "Move Right 1 Frame",          "mac": "Right",                    "windows": "Right",                    "category": "Edit"},
        # Tools
        {"action_id": "tool_selector",     "label": "Select Tool",                 "mac": "V",                        "windows": "V",                        "category": "Tools"},
        {"action_id": "tool_blade",        "label": "Split / Blade",               "mac": "Cmd+B",                    "windows": "Ctrl+B",                   "category": "Tools"},
        # View
        {"action_id": "zoom_in",           "label": "Zoom In Timeline",            "mac": "Cmd+=",                    "windows": "Ctrl+=",                   "category": "View"},
        {"action_id": "zoom_out",          "label": "Zoom Out Timeline",           "mac": "Cmd+-",                    "windows": "Ctrl+-",                   "category": "View"},
        {"action_id": "fit_tracks",        "label": "Fit Timeline to Window",      "mac": "Cmd+Shift+F",              "windows": "Ctrl+Shift+F",             "category": "View"},
        # Export
        {"action_id": "export",            "label": "Export",                      "mac": "Cmd+E",                    "windows": "Ctrl+E",                   "category": "Export"},
    ],

    # -----------------------------------------------------------------------
    # ADOBE PREMIERE PRO
    # -----------------------------------------------------------------------
    "premiere_pro": [
        # Transport
        {"action_id": "play_stop",         "label": "Play / Stop",                 "mac": "Space / K",                "windows": "Space / K",                "category": "Transport"},
        {"action_id": "record",            "label": "Record",                      "mac": "Shift+Cmd+R",              "windows": "Shift+Ctrl+R",             "category": "Transport"},
        {"action_id": "go_to_start",       "label": "Go to In Point",              "mac": "Shift+I",                  "windows": "Shift+I",                  "category": "Transport"},
        {"action_id": "rewind",            "label": "Shuttle Left (J)",            "mac": "J",                        "windows": "J",                        "category": "Transport"},
        {"action_id": "fast_forward",      "label": "Shuttle Right (L)",           "mac": "L",                        "windows": "L",                        "category": "Transport"},
        {"action_id": "loop_playback",     "label": "Loop",                        "mac": "Ctrl+L",                   "windows": "Ctrl+L",                   "category": "Transport"},
        # Edit
        {"action_id": "undo",              "label": "Undo",                        "mac": "Cmd+Z",                    "windows": "Ctrl+Z",                   "category": "Edit"},
        {"action_id": "redo",              "label": "Redo",                        "mac": "Cmd+Shift+Z",              "windows": "Ctrl+Shift+Z",             "category": "Edit"},
        {"action_id": "cut",               "label": "Cut",                         "mac": "Cmd+X",                    "windows": "Ctrl+X",                   "category": "Edit"},
        {"action_id": "copy",              "label": "Copy",                        "mac": "Cmd+C",                    "windows": "Ctrl+C",                   "category": "Edit"},
        {"action_id": "paste",             "label": "Paste",                       "mac": "Cmd+V",                    "windows": "Ctrl+V",                   "category": "Edit"},
        {"action_id": "select_all",        "label": "Select All",                  "mac": "Cmd+A",                    "windows": "Ctrl+A",                   "category": "Edit"},
        {"action_id": "split_clip",        "label": "Razor / Add Edit",            "mac": "Cmd+K",                    "windows": "Ctrl+K",                   "category": "Edit"},
        {"action_id": "ripple_delete",     "label": "Ripple Delete",               "mac": "Shift+Delete",             "windows": "Shift+Delete",             "category": "Edit"},
        {"action_id": "nudge_left",        "label": "Nudge Left 1 Frame",          "mac": "Opt+Left",                 "windows": "Alt+Left",                 "category": "Edit"},
        {"action_id": "nudge_right",       "label": "Nudge Right 1 Frame",         "mac": "Opt+Right",                "windows": "Alt+Right",                "category": "Edit"},
        {"action_id": "trim_to_selection", "label": "Trim In/Out to Playhead",     "mac": "Q / W",                    "windows": "Q / W",                    "category": "Edit"},
        # Tools
        {"action_id": "tool_selector",     "label": "Selection Tool",              "mac": "V",                        "windows": "V",                        "category": "Tools"},
        {"action_id": "tool_trimmer",      "label": "Ripple Edit Tool",            "mac": "B",                        "windows": "B",                        "category": "Tools"},
        {"action_id": "tool_blade",        "label": "Razor Tool",                  "mac": "C",                        "windows": "C",                        "category": "Tools"},
        {"action_id": "tool_zoom",         "label": "Zoom Tool",                   "mac": "Z",                        "windows": "Z",                        "category": "Tools"},
        {"action_id": "tool_hand",         "label": "Hand Tool",                   "mac": "H",                        "windows": "H",                        "category": "Tools"},
        # View
        {"action_id": "zoom_in",           "label": "Zoom In",                     "mac": "=",                        "windows": "=",                        "category": "View"},
        {"action_id": "zoom_out",          "label": "Zoom Out",                    "mac": "-",                        "windows": "-",                        "category": "View"},
        {"action_id": "zoom_to_selection", "label": "Fit Sequence in Window",      "mac": "\\",                       "windows": "\\",                       "category": "View"},
        # Markers
        {"action_id": "add_marker",        "label": "Add Marker",                  "mac": "M",                        "windows": "M",                        "category": "Markers"},
        {"action_id": "next_marker",       "label": "Next Marker",                 "mac": "Shift+M",                  "windows": "Shift+M",                  "category": "Markers"},
        # Tracks
        {"action_id": "new_track",         "label": "Add Track",                   "mac": "Shift+Cmd+T",              "windows": "Shift+Ctrl+T",             "category": "Tracks"},
        {"action_id": "mute_track",        "label": "Mute Track",                  "mac": "Shift+Cmd+M",              "windows": "Shift+Ctrl+M",             "category": "Tracks"},
    ],

    # -----------------------------------------------------------------------
    # LOGIC PRO X
    # -----------------------------------------------------------------------
    "logic_pro": [
        # Transport
        {"action_id": "play_stop",         "label": "Play / Stop",                 "mac": "Space",                    "windows": None,                       "category": "Transport"},
        {"action_id": "record",            "label": "Record",                      "mac": "R",                        "windows": None,                       "category": "Transport"},
        {"action_id": "go_to_start",       "label": "Go to Beginning",             "mac": "Return",                   "windows": None,                       "category": "Transport"},
        {"action_id": "rewind",            "label": "Rewind",                      "mac": "Num,",                     "windows": None,                       "category": "Transport"},
        {"action_id": "fast_forward",      "label": "Fast Forward",                "mac": "Num+",                     "windows": None,                       "category": "Transport"},
        {"action_id": "loop_playback",     "label": "Toggle Cycle",                "mac": "C",                        "windows": None,                       "category": "Transport"},
        # Edit
        {"action_id": "undo",              "label": "Undo",                        "mac": "Cmd+Z",                    "windows": None,                       "category": "Edit"},
        {"action_id": "redo",              "label": "Redo",                        "mac": "Cmd+Shift+Z",              "windows": None,                       "category": "Edit"},
        {"action_id": "cut",               "label": "Cut",                         "mac": "Cmd+X",                    "windows": None,                       "category": "Edit"},
        {"action_id": "copy",              "label": "Copy",                        "mac": "Cmd+C",                    "windows": None,                       "category": "Edit"},
        {"action_id": "paste",             "label": "Paste",                       "mac": "Cmd+V",                    "windows": None,                       "category": "Edit"},
        {"action_id": "select_all",        "label": "Select All",                  "mac": "Cmd+A",                    "windows": None,                       "category": "Edit"},
        {"action_id": "split_clip",        "label": "Split Region at Playhead",    "mac": "Cmd+T",                    "windows": None,                       "category": "Edit"},
        {"action_id": "nudge_left",        "label": "Nudge Left",                  "mac": "Opt+Left",                 "windows": None,                       "category": "Edit"},
        {"action_id": "nudge_right",       "label": "Nudge Right",                 "mac": "Opt+Right",                "windows": None,                       "category": "Edit"},
        {"action_id": "consolidate_clip",  "label": "Join Regions",                "mac": "Cmd+J",                    "windows": None,                       "category": "Edit"},
        # Tools
        {"action_id": "tool_selector",     "label": "Pointer Tool",                "mac": "T (then 1)",               "windows": None,                       "category": "Tools"},
        {"action_id": "tool_pencil",       "label": "Pencil Tool",                 "mac": "T (then 3)",               "windows": None,                       "category": "Tools"},
        {"action_id": "tool_trimmer",      "label": "Scissors Tool",               "mac": "T (then 4)",               "windows": None,                       "category": "Tools"},
        {"action_id": "tool_zoom",         "label": "Zoom Tool",                   "mac": "T (then 6)",               "windows": None,                       "category": "Tools"},
        # View
        {"action_id": "zoom_in",           "label": "Zoom In",                     "mac": "Cmd+Right",                "windows": None,                       "category": "View"},
        {"action_id": "zoom_out",          "label": "Zoom Out",                    "mac": "Cmd+Left",                 "windows": None,                       "category": "View"},
        {"action_id": "fit_tracks",        "label": "Fit All Tracks",              "mac": "Ctrl+Opt+Z",               "windows": None,                       "category": "View"},
        # Markers
        {"action_id": "add_marker",        "label": "Add Marker",                  "mac": "Opt+`",                    "windows": None,                       "category": "Markers"},
        {"action_id": "next_marker",       "label": "Next Marker",                 "mac": "Opt+Right",                "windows": None,                       "category": "Markers"},
        # Tracks
        {"action_id": "new_track",         "label": "New Track",                   "mac": "Opt+Cmd+N",                "windows": None,                       "category": "Tracks"},
        {"action_id": "mute_track",        "label": "Mute Track",                  "mac": "M",                        "windows": None,                       "category": "Tracks"},
        {"action_id": "solo_track",        "label": "Solo Track",                  "mac": "S",                        "windows": None,                       "category": "Tracks"},
    ],

    # -----------------------------------------------------------------------
    # DAVINCI RESOLVE
    # -----------------------------------------------------------------------
    "davinci_resolve": [
        # Transport
        {"action_id": "play_stop",         "label": "Play / Stop",                 "mac": "Space / K",                "windows": "Space / K",                "category": "Transport"},
        {"action_id": "rewind",            "label": "Shuttle Left (J)",            "mac": "J",                        "windows": "J",                        "category": "Transport"},
        {"action_id": "fast_forward",      "label": "Shuttle Right (L)",           "mac": "L",                        "windows": "L",                        "category": "Transport"},
        {"action_id": "go_to_start",       "label": "Go to Start",                 "mac": "Fn+Left",                  "windows": "Home",                     "category": "Transport"},
        {"action_id": "loop_playback",     "label": "Loop",                        "mac": "Cmd+/",                    "windows": "Ctrl+/",                   "category": "Transport"},
        # Edit
        {"action_id": "undo",              "label": "Undo",                        "mac": "Cmd+Z",                    "windows": "Ctrl+Z",                   "category": "Edit"},
        {"action_id": "redo",              "label": "Redo",                        "mac": "Cmd+Shift+Z",              "windows": "Ctrl+Shift+Z",             "category": "Edit"},
        {"action_id": "cut",               "label": "Cut",                         "mac": "Cmd+X",                    "windows": "Ctrl+X",                   "category": "Edit"},
        {"action_id": "copy",              "label": "Copy",                        "mac": "Cmd+C",                    "windows": "Ctrl+C",                   "category": "Edit"},
        {"action_id": "paste",             "label": "Paste",                       "mac": "Cmd+V",                    "windows": "Ctrl+V",                   "category": "Edit"},
        {"action_id": "select_all",        "label": "Select All",                  "mac": "Cmd+A",                    "windows": "Ctrl+A",                   "category": "Edit"},
        {"action_id": "split_clip",        "label": "Split Clip",                  "mac": "Cmd+\\",                   "windows": "Ctrl+\\",                  "category": "Edit"},
        {"action_id": "ripple_delete",     "label": "Ripple Delete",               "mac": "Shift+Delete",             "windows": "Shift+Delete",             "category": "Edit"},
        {"action_id": "nudge_left",        "label": "Nudge Left 1 Frame",          "mac": "Comma",                    "windows": "Comma",                    "category": "Edit"},
        {"action_id": "nudge_right",       "label": "Nudge Right 1 Frame",         "mac": "Period",                   "windows": "Period",                   "category": "Edit"},
        # Tools
        {"action_id": "tool_selector",     "label": "Selection Mode",              "mac": "A",                        "windows": "A",                        "category": "Tools"},
        {"action_id": "tool_trimmer",      "label": "Trim Mode",                   "mac": "T",                        "windows": "T",                        "category": "Tools"},
        {"action_id": "tool_blade",        "label": "Blade Mode",                  "mac": "B",                        "windows": "B",                        "category": "Tools"},
        {"action_id": "tool_zoom",         "label": "Dynamic Zoom",                "mac": "Shift+Z",                  "windows": "Shift+Z",                  "category": "Tools"},
        # View
        {"action_id": "zoom_in",           "label": "Zoom In",                     "mac": "Cmd+=",                    "windows": "Ctrl+=",                   "category": "View"},
        {"action_id": "zoom_out",          "label": "Zoom Out",                    "mac": "Cmd+-",                    "windows": "Ctrl+-",                   "category": "View"},
        {"action_id": "zoom_to_selection", "label": "Zoom to Fit",                 "mac": "Shift+Z",                  "windows": "Shift+Z",                  "category": "View"},
        # Markers
        {"action_id": "add_marker",        "label": "Add Marker",                  "mac": "M",                        "windows": "M",                        "category": "Markers"},
        {"action_id": "next_marker",       "label": "Next Marker",                 "mac": "Shift+M",                  "windows": "Shift+M",                  "category": "Markers"},
        # Tracks
        {"action_id": "new_track",         "label": "Add Video Track",             "mac": "Shift+Cmd+T",              "windows": "Shift+Ctrl+T",             "category": "Tracks"},
        {"action_id": "mute_track",        "label": "Mute Track",                  "mac": "Opt+M",                    "windows": "Alt+M",                    "category": "Tracks"},
        # Color
        {"action_id": "show_color_board",  "label": "Show Color Page",             "mac": "Shift+6",                  "windows": "Shift+6",                  "category": "Color"},
        {"action_id": "balance_color",     "label": "Auto Color",                  "mac": "Opt+Shift+C",              "windows": "Alt+Shift+C",              "category": "Color"},
    ],
}

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

APP_METADATA: dict[str, dict] = {
    "pro_tools": {
        "name": "Pro Tools",
        "vendor": "Avid",
        "type": "DAW",
        "platforms": ["mac", "windows"],
        "icon": "🎛️",
        "description": "Industry-standard digital audio workstation for music and post-production.",
    },
    "final_cut_pro": {
        "name": "Final Cut Pro",
        "vendor": "Apple",
        "type": "NLE",
        "platforms": ["mac"],
        "icon": "🎬",
        "description": "Professional non-linear video editor for macOS.",
    },
    "capcut": {
        "name": "CapCut",
        "vendor": "ByteDance",
        "type": "NLE",
        "platforms": ["mac", "windows"],
        "icon": "✂️",
        "description": "Fast, modern video editor popular with content creators.",
    },
    "premiere_pro": {
        "name": "Adobe Premiere Pro",
        "vendor": "Adobe",
        "type": "NLE",
        "platforms": ["mac", "windows"],
        "icon": "🎞️",
        "description": "Professional video editing software by Adobe.",
    },
    "logic_pro": {
        "name": "Logic Pro X",
        "vendor": "Apple",
        "type": "DAW",
        "platforms": ["mac"],
        "icon": "🎵",
        "description": "Professional digital audio workstation for macOS.",
    },
    "davinci_resolve": {
        "name": "DaVinci Resolve",
        "vendor": "Blackmagic Design",
        "type": "NLE/Color",
        "platforms": ["mac", "windows"],
        "icon": "🎨",
        "description": "Professional video editor and color grading suite.",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_all_apps() -> list[dict]:
    """Return list of all supported apps with metadata."""
    return [
        {"app_id": app_id, **meta}
        for app_id, meta in APP_METADATA.items()
    ]


def get_shortcuts_for_app(app_id: str) -> list[ShortcutEntry]:
    """Return all shortcuts for a given app."""
    return SHORTCUT_DB.get(app_id, [])


def get_shortcut_map(app_id: str) -> dict[str, ShortcutEntry]:
    """Return a dict keyed by action_id for fast lookup."""
    return {s["action_id"]: s for s in SHORTCUT_DB.get(app_id, [])}


def map_shortcuts(
    source_app: str,
    target_app: str,
    platform: str = "mac",
) -> list[dict]:
    """
    Map every shortcut from source_app to its equivalent in target_app.

    Returns a list of mapping objects:
    {
        action_id, label, category,
        source_shortcut, target_shortcut,
        mapped: bool  (False if no equivalent exists in target)
    }
    """
    source_map = get_shortcut_map(source_app)
    target_map = get_shortcut_map(target_app)

    results = []
    for action_id, source_entry in source_map.items():
        source_key = source_entry.get(platform)
        target_entry = target_map.get(action_id)
        target_key = target_entry.get(platform) if target_entry else None

        results.append({
            "action_id": action_id,
            "label": source_entry["label"],
            "category": source_entry["category"],
            "source_shortcut": source_key,
            "target_shortcut": target_key,
            "target_label": target_entry["label"] if target_entry else None,
            "mapped": target_key is not None,
        })

    # Sort: mapped first, then by category, then by label
    results.sort(key=lambda x: (not x["mapped"], x["category"], x["label"]))
    return results


def get_categories_for_app(app_id: str) -> list[str]:
    """Return sorted unique categories for an app."""
    shortcuts = get_shortcuts_for_app(app_id)
    return sorted(set(s["category"] for s in shortcuts))
