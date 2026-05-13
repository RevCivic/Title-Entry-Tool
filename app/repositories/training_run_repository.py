"""Repository for persisting and retrieving training-run records."""

from dataclasses import replace
from datetime import datetime
from typing import List

import psycopg2
import psycopg2.extras

from app.models.training_run import TrainingRun


class TrainingRunRepository:
    """Encapsulate database operations for :class:`TrainingRun`."""

    def __init__(self, connection: psycopg2.extensions.connection) -> None:
        self.connection = connection

    def create(self, run: TrainingRun) -> TrainingRun:
        """Insert a training-run record and return it with its assigned id."""
        exported_at = run.exported_at or datetime.utcnow().isoformat()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO training_runs (exported_at, sample_count, notes, export_path)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (exported_at, run.sample_count, run.notes or None, run.export_path),
            )
            run_id = cursor.fetchone()[0]
        self.connection.commit()
        return replace(run, id=run_id, exported_at=exported_at)

    def list(self) -> List[TrainingRun]:
        """Return all training runs, newest first."""
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM training_runs ORDER BY id DESC")
            rows = cursor.fetchall()
        return [TrainingRun.from_dict(dict(row)) for row in rows]
