import math
from dataclasses import dataclass
from datetime import date, timedelta

from order_network.models import OrderLine, ProductionJob, ProductionPlan


@dataclass(frozen=True)
class Routing:
    description: str
    line: str
    units_per_day: int
    setup_days: int


ROUTINGS = {
    "149449": Routing("Hydraulic pump housing", "CASTING-1", 12, 2),
    "255565": Routing("Gearbox output shaft 40 mm", "CNC-2", 60, 1),
    "252653": Routing("Flange coupling DN50", "CNC-1", 40, 1),
    "252654": Routing("Flange coupling DN65", "CNC-1", 30, 1),
    "310022": Routing("Servo drive mounting bracket", "SHEET-1", 100, 1),
    "418870": Routing("Conveyor roller 500 mm", "ASSEMBLY-1", 150, 0),
}

LINE_BACKLOG_DAYS = {"CASTING-1": 4, "CNC-1": 2, "CNC-2": 3, "SHEET-1": 1, "ASSEMBLY-1": 2}


def add_business_days(start: date, days: int) -> date:
    current = start
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current


def business_days_between(start: date, end: date) -> int:
    days, current = 0, start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


def plan_production(lines: list[OrderLine], today: date | None = None) -> ProductionPlan:
    """Jobs are queued per production line behind the current backlog and behind earlier jobs on the same line."""
    today = today or date.today()
    line_free_from = {line: add_business_days(today, backlog) for line, backlog in LINE_BACKLOG_DAYS.items()}
    jobs = []
    for order_line in lines:
        routing = ROUTINGS.get(order_line.material_number)
        if routing is None:
            jobs.append(ProductionJob(
                material_number=order_line.material_number, description=None, quantity=order_line.quantity,
                production_line=None, start_date=None, finish_date=None, lead_time_business_days=None,
                note="No routing found: material cannot be produced in-house.",
            ))
            continue
        start = line_free_from[routing.line]
        duration = routing.setup_days + math.ceil(order_line.quantity / routing.units_per_day)
        finish = add_business_days(start, duration)
        line_free_from[routing.line] = finish
        jobs.append(ProductionJob(
            material_number=order_line.material_number, description=routing.description, quantity=order_line.quantity,
            production_line=routing.line, start_date=start, finish_date=finish,
            lead_time_business_days=business_days_between(today, finish),
        ))
    finishes = [job.finish_date for job in jobs if job.finish_date]
    return ProductionPlan(planned_on=today, jobs=jobs, latest_finish_date=max(finishes) if finishes else None)
