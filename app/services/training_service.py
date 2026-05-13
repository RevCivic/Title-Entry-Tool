"""Service layer for training-related workflows.

Encapsulates the logic for querying annotation-queue priority order,
computing ground-truth dataset statistics, and providing correction data
for export and LLM prompt-tuning — keeping this logic out of both
``title_entry_tool.py`` function wrappers and Flask route handlers.
"""

from typing import Any, Dict, List

import psycopg2
import psycopg2.extras

from app.repositories.correction_repository import CorrectionRepository
from app.repositories.training_run_repository import TrainingRunRepository


class TrainingService:
    """Orchestrate training-data and annotation workflows.

    Parameters
    ----------
    connection:
        An active psycopg2 database connection.  The service does not own
        the connection lifetime; callers are responsible for closing it.
    """

    def __init__(self, connection: psycopg2.extensions.connection) -> None:
        self.connection = connection
        self._corrections = CorrectionRepository(connection)
        self._runs = TrainingRunRepository(connection)

    # ── Annotation queue ──────────────────────────────────────────────────────

    def get_annotation_queue(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return records needing annotation, ordered by priority.

        Priority order:
          1. Fewest ground-truth corrections (unannotated first).
          2. Unvalidated records before validated.
          3. Newest records first within each tier.

        Returns a list of plain dicts for template rendering compatibility.
        """
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    r.id, r.state, r.title_number, r.vin, r.vehicle_year,
                    r.make, r.model, r.color, r.is_validated, r.created_at,
                    COUNT(c.id) FILTER (WHERE c.is_ground_truth = 1) AS gt_count
                FROM title_records r
                LEFT JOIN corrections c ON c.record_id = r.id
                GROUP BY r.id
                ORDER BY
                    COUNT(c.id) FILTER (WHERE c.is_ground_truth = 1) ASC,
                    r.is_validated ASC,
                    r.id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            return [dict(r) for r in cursor.fetchall()]

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the ground-truth training dataset.

        Keys returned:
          total_gt_corrections  – int, total ground-truth correction count
          by_field              – list of {field_name, count} dicts
          by_state              – list of {state, count} dicts
          coverage              – dict with total_records, ge1, ge5, ge10
        """
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM corrections WHERE is_ground_truth = 1"
            )
            row = cursor.fetchone()
            total = int(row["total"]) if row else 0

            cursor.execute(
                """
                SELECT field_name, COUNT(*) AS count
                FROM corrections
                WHERE is_ground_truth = 1
                GROUP BY field_name
                ORDER BY count DESC
                """
            )
            by_field = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT r.state, COUNT(c.id) AS count
                FROM corrections c
                JOIN title_records r ON c.record_id = r.id
                WHERE c.is_ground_truth = 1
                GROUP BY r.state
                ORDER BY count DESC
                """
            )
            by_state = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_records,
                    COUNT(CASE WHEN gt_count >= 1 THEN 1 END) AS ge1,
                    COUNT(CASE WHEN gt_count >= 5 THEN 1 END) AS ge5,
                    COUNT(CASE WHEN gt_count >= 10 THEN 1 END) AS ge10
                FROM (
                    SELECT r.id,
                        COUNT(c.id) FILTER (WHERE c.is_ground_truth = 1) AS gt_count
                    FROM title_records r
                    LEFT JOIN corrections c ON c.record_id = r.id
                    GROUP BY r.id
                ) sub
                """
            )
            cov_row = cursor.fetchone()
            coverage = (
                {k: int(v) for k, v in dict(cov_row).items()}
                if cov_row
                else {"total_records": 0, "ge1": 0, "ge5": 0, "ge10": 0}
            )

        return {
            "total_gt_corrections": total,
            "by_field": by_field,
            "by_state": by_state,
            "coverage": coverage,
        }
