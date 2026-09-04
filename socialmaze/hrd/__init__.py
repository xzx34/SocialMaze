"""Hidden Role Deduction (HRD), the core task of SocialMaze.

Modules
-------
rules      game definition: roles, variants, GameConfig, Statement
scenario   the Scenario record and its JSON schema
simulate   role assignment and statement generation (targeting policies)
solver     exhaustive uniqueness solver following Algorithm 1 of the paper
explain    natural-language reasoning chains derived from the solver trace
generate   dataset generation with rejection sampling on unique solvability
prompts    canonical system/user prompts for evaluation
parsing    extraction of the Final Judgment from model output
metrics    accuracy, confidence intervals and error decomposition
evaluate   querying models (incremental or final mode) with resume support
report     aggregation of a run directory into tables
io         JSONL / legacy JSON / HuggingFace row conversions
cli        ``python -m socialmaze.hrd`` entry point

Typical use::

    python -m socialmaze.hrd generate -n 6 --variant full -N 500 --out data/hrd/my.jsonl
    python -m socialmaze.hrd evaluate --data data/hrd/my.jsonl --models gpt-4o-mini --out runs/demo
    python -m socialmaze.hrd report runs/demo
"""

from .rules import (  # noqa: F401
    CRIMINAL,
    DISPLAYED_ROLE,
    INVESTIGATOR,
    LUNATIC,
    ROLES,
    RUMORMONGER,
    UNKNOWN,
    VARIANTS,
    GameConfig,
    Statement,
)
from .scenario import Answer, Scenario, TASK_NAME  # noqa: F401
