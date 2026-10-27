"""Starter: the allowance incorrectly uses one hour of traffic."""


def bank_budget():
    return {
        "allowance": 3000 * 0.001,
        "burn": 0.02 / 0.001,
        "hourly_page_minutes": 0.02 * 3000 * 0.001,
        "fault_bad": 3000 * 0.02 * 1.5,
    }
