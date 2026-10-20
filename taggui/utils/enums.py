from enum import Enum


# `StrEnum` is a Python 3.11 feature that can be used here.
class AllTagsSortBy(str, Enum):
    FREQUENCY = 'Frequency'
    NAME = 'Name'
    LENGTH = 'Length'


class SortOrder(str, Enum):
    ASCENDING = 'Ascending'
    DESCENDING = 'Descending'


class CaptionPosition(str, Enum):
    BEFORE_FIRST_TAG = 'Insert before first tag'
    AFTER_LAST_TAG = 'Insert after last tag'
    OVERWRITE_FIRST_TAG = 'Overwrite first tag'
    OVERWRITE_ALL_TAGS = 'Overwrite all tags'
    DO_NOT_ADD = 'Do not add to tags'


class ThinkingMode(str, Enum):
    """
    Whether the captioning model should use thinking (chain-of-thought
    reasoning) when the model and its chat template support it. The value is
    passed to `llama-server` via its `--reasoning` argument.
    """
    AUTO = 'Model default (auto)'
    ENABLED = 'Enabled'
    DISABLED = 'Disabled'


class ReasoningEffort(str, Enum):
    """
    The reasoning effort level passed to the chat template via the
    `--reasoning-effort` argument of `llama-server`. Only applies to models
    whose templates support it.
    """
    DEFAULT = 'Model default'
    MINIMAL = 'Minimal'
    LOW = 'Low'
    MEDIUM = 'Medium'
    HIGH = 'High'
    XHIGH = 'xHigh'
    MAX = 'Max'
