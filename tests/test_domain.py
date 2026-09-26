from datetime import date

from order_network.domain.extraction import classify_numbers
from order_network.domain.inventory import check_stock, init_db
from order_network.domain.planning import add_business_days, business_days_between, plan_production
from order_network.models import OrderLine


def test_classify_numbers_filters_order_references():
    materials, ignored = classify_numbers(
        "Please send 50 x 149449 and 12 x 252654, also 255565. Ref 9938812, customer 90417, by 2026-10-30, "
        "1000 pcs of 1234567 at 12345.50 EUR."
    )
    assert materials == ["149449", "252654", "255565"]
    assert {i.value for i in ignored} == {"9938812", "90417", "1234567"}
    assert "starts with 9" in next(i.reason for i in ignored if i.value == "9938812")


def test_classify_numbers_deduplicates():
    materials, _ = classify_numbers("149449, again 149449")
    assert materials == ["149449"]


def test_check_stock(tmp_path):
    db = tmp_path / "inv.db"
    init_db(db)
    report = check_stock(db, [OrderLine(material_number="149449", quantity=50),
                              OrderLine(material_number="255565", quantity=100),
                              OrderLine(material_number="123456", quantity=1)])
    pump, shaft, unknown = report.lines
    assert (pump.available, pump.reserve_now, pump.shortfall) == (30, 30, 20)
    assert shaft.shortfall == 0 and shaft.warehouse == "Linz"
    assert not unknown.known_material and unknown.shortfall == 1
    assert report.fully_available is False
    assert [(l.material_number, l.quantity) for l in report.production_request] == [("149449", 20), ("123456", 1)]


def test_business_day_helpers():
    friday = date(2026, 9, 25)
    assert add_business_days(friday, 1) == date(2026, 9, 28)
    assert business_days_between(friday, date(2026, 10, 2)) == 5


def test_plan_production_queues_jobs_on_same_line():
    monday = date(2026, 9, 28)
    plan = plan_production([OrderLine(material_number="252653", quantity=40),
                            OrderLine(material_number="252654", quantity=12),
                            OrderLine(material_number="149449", quantity=20)], today=monday)
    dn50, dn65, pump = plan.jobs
    # CNC-1: backlog 2 days -> DN50 1 setup + 1 day, then DN65 queued behind it
    assert dn50.start_date == date(2026, 9, 30) and dn50.finish_date == date(2026, 10, 2)
    assert dn65.start_date == dn50.finish_date and dn65.lead_time_business_days == 6
    # CASTING-1: backlog 4, setup 2, ceil(20/12)=2 -> 8 business days
    assert pump.lead_time_business_days == 8
    assert plan.latest_finish_date == pump.finish_date


def test_plan_production_unknown_material():
    plan = plan_production([OrderLine(material_number="999999", quantity=5)])
    assert plan.jobs[0].production_line is None and plan.latest_finish_date is None
