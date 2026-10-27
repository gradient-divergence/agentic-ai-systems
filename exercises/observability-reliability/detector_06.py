"""Starter: a daily average treats the closed market as missing quotes."""

from datetime import timedelta

from refund_agent.observability_reliability.silence import (
    SilenceDetector,
)


def rate_at(at):
    return 975


def quiet_hours(rate_per_hour, open_hours=325, expected_false=1):
    """Replace this daily-average timeout with the ten-week false-page threshold."""
    return 24 / 975


def run_detector(rate_function=rate_at, quiet_function=quiet_hours):
    from quotes_06 import START, quotes

    pages = []
    for venue in range(3):
        detector = None
        last = None
        for at in (*quotes(venue), START + timedelta(days=5)):
            if last is not None and detector is not None:
                boundary = last + timedelta(hours=detector.t_quiet)
                if boundary < at and rate_function(boundary) > 0:
                    alert = detector.check(boundary)
                    if alert is not None:
                        pages.append({"venue": venue, "at": alert.window_end})
            rate = rate_function(at)
            if rate > 0:
                detector = SilenceDetector(rate, quiet_function(rate))
                detector.arrival(at)
            else:
                detector = None
            last = at
    return pages


if __name__ == "__main__":
    print(run_detector())
