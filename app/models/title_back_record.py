"""TitleBackRecord data model.

Represents the back-of-title (reassignment) data for a vehicle title,
capturing sale transaction details at the time of transfer.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TitleBackRecord:
    """Back-of-title reassignment data for a single title record."""

    id: Optional[int] = None
    title_record_id: Optional[int] = None
    odometer_at_sale: Optional[int] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    seller_name: Optional[str] = None
    sale_price: Optional[float] = None
    sale_date: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title_record_id": self.title_record_id,
            "odometer_at_sale": self.odometer_at_sale,
            "buyer_name": self.buyer_name,
            "buyer_address": self.buyer_address,
            "seller_name": self.seller_name,
            "sale_price": self.sale_price,
            "sale_date": self.sale_date,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TitleBackRecord":
        return cls(
            id=d.get("id"),
            title_record_id=d.get("title_record_id"),
            odometer_at_sale=d.get("odometer_at_sale"),
            buyer_name=d.get("buyer_name"),
            buyer_address=d.get("buyer_address"),
            seller_name=d.get("seller_name"),
            sale_price=(
                float(d["sale_price"]) if d.get("sale_price") is not None else None
            ),
            sale_date=d.get("sale_date"),
            notes=d.get("notes"),
            created_at=d.get("created_at") or "",
        )
