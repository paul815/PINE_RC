"""Which settings may come from outside this install, and in what shape.

Two doors let a value in without a person choosing it on this machine: a PATCH
from the page and "Restore settings" from a backup. Both go through
clean_setting(); a restore additionally only takes the keys listed in
RESTORABLE_SETTINGS.
"""

ALLOWED_FONT_FAMILIES = frozenset({'system', 'inter', 'source-sans-3', 'open-sans', 'roboto'})

# Wider than the four sizes the UI offers (12/13/15/17), so a value set by an
# older build still loads; narrow enough that it can only ever be a size.
FONT_SIZE_MIN = 8
FONT_SIZE_MAX = 32
DEFAULT_FONT_SIZE = '13'

PII_THRESHOLD_MIN = 0.3   # settings.html: <input type="range" min="0.3" max="0.9">
PII_THRESHOLD_MAX = 0.9

# Preferences that describe how a person likes to work and travel sensibly
# with their backup. Everything else stays with the machine that set it:
#   backup_path, backup_include_audio, auto_backup_* — where and what this
#     install writes on its own; an archive must not redirect that, least of
#     all to a network share;
#   stt_model_id — names a model that may not be installed here;
#   last_open_project_id, launcher and onboarding keys — this install's state;
#   export_default_prompt(_recording) — text that goes to an LLM together with
#     the transcript; a backup someone else made must not write it;
#   hf_token, models_path, projects_path — never in a backup to begin with.
RESTORABLE_SETTINGS = frozenset({
    'font_size',
    'font_family',
    'theme',
    'export_default_format',
    'export_default_comments',
    'export_default_tags',
    'export_default_project_details',
    'export_default_participant_details',
    'export_default_remove_pii',
    'export_default_include_prompt',
    'transcription_timeout_secs',
    'link_recordings',
    'backup_retention_count',
    'pii_threshold',
    'transcription_complete_sound_enabled',
    'transcription_complete_sound_volume',
})


def clean_font_size(value):
    """The size as a string of digits, or None if *value* is not a sensible one."""
    text = str(value if value is not None else '').strip()
    if not text.isdigit():
        return None
    size = int(text)
    if not FONT_SIZE_MIN <= size <= FONT_SIZE_MAX:
        return None
    return str(size)


def clean_setting(key, value):
    """Normalised *value* for *key*, or None to reject it.

    Covers the keys whose value lands somewhere it could do harm or break the
    page; the rest are stored as given, as before.
    """
    text = str(value if value is not None else '').strip()
    if key == 'font_size':
        return clean_font_size(text)
    if key == 'font_family':
        return text if text in ALLOWED_FONT_FAMILIES else None
    if key == 'transcription_complete_sound_volume':
        if not text.lstrip('-').isdigit():
            return None
        volume = int(text)
        return str(volume) if 0 <= volume <= 100 else None
    if key == 'pii_threshold':
        # The slider's range. Near 1.0 the detector reports nothing, which
        # turns "remove PII" off without a word.
        try:
            threshold = float(text)
        except ValueError:
            return None
        return text if PII_THRESHOLD_MIN <= threshold <= PII_THRESHOLD_MAX else None
    return text
