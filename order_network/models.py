from datetime import date

from pydantic import BaseModel, Field


class OrderLine(BaseModel):
    material_number: str = Field(description="6-digit material number")
    quantity: int = Field(ge=1)


class IgnoredNumber(BaseModel):
    value: str
    reason: str


class OrderRequest(BaseModel):
    customer: str | None = Field(default=None, description="Customer or company name if mentioned")
    lines: list[OrderLine]
    requested_delivery: date | None = None
    ignored_numbers: list[IgnoredNumber] = []


class ProductionJob(BaseModel):
    material_number: str
    description: str | None
    quantity: int
    production_line: str | None
    start_date: date | None
    finish_date: date | None
    lead_time_business_days: int | None
    note: str | None = None


class ProductionPlan(BaseModel):
    planned_on: date
    jobs: list[ProductionJob]
    latest_finish_date: date | None


class StockLine(BaseModel):
    material_number: str
    description: str | None
    known_material: bool
    requested: int
    available: int
    reserve_now: int
    shortfall: int
    warehouse: str | None


class StockReport(BaseModel):
    lines: list[StockLine]
    fully_available: bool
    production_request: list[OrderLine] = Field(description="Shortfall quantities that must be produced")
    production_plan: ProductionPlan | None = Field(
        default=None, description="Plan from the production planning agent for the shortfall quantities"
    )
    production_note: str | None = None
