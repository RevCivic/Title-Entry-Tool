"""Repository classes for the Title Entry Tool."""

from app.repositories.correction_repository import CorrectionRepository
from app.repositories.title_back_record_repository import TitleBackRecordRepository
from app.repositories.title_image_repository import TitleImageRepository
from app.repositories.title_record_repository import TitleRecordRepository
from app.repositories.training_run_repository import TrainingRunRepository

__all__ = [
    "CorrectionRepository",
    "TitleBackRecordRepository",
    "TitleImageRepository",
    "TitleRecordRepository",
    "TrainingRunRepository",
]
