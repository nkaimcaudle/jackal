import datetime


def date_diff(dt1: datetime.datetime, dt2: datetime.datetime) -> float:
    return (
        dt2.astimezone(datetime.UTC) - dt1.astimezone(datetime.UTC)
    ) / datetime.timedelta(days=365.25)


TDATE_PIVOT = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


def to_Tdate(dt: datetime.datetime) -> float:
    return date_diff(TDATE_PIVOT, dt)
