"""Optional diagnostic for total commitment effect.

Disabled by default in Round 2. It provides a small container for future audits
that compare pre- and post-commit answers without changing pipeline behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TotalCommitmentEffect:
    baseline_answer: str
    committed_answer: str
    answer_source: str
    enabled: bool = False

    def changed(self) -> bool:
        return str(self.baseline_answer).strip() != str(self.committed_answer).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "baseline_answer": self.baseline_answer,
            "committed_answer": self.committed_answer,
            "answer_source": self.answer_source,
            "changed": self.changed(),
        }
