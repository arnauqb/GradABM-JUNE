from pytest import fixture
from pathlib import Path
import yaml

from torch_june import Timer

"""
These are taken directly from JUNE.
"""


class TestTimer:
    @fixture(name="timer")
    def make_timer(self):
        timer = Timer(initial_day="2020-03-10", total_days=10)
        return timer

    def test__initial_parameters(self, timer):
        assert timer.shift == 0
        assert timer.is_weekend is False
        assert timer.day_of_week == "Tuesday"
        assert timer.day_type == "weekday"
        assert timer.date_str == "2020-03-10"

    def test__time_is_passing(self, timer):
        assert timer.now == 0
        next(timer)
        assert timer.now == 0.5
        assert timer.previous_date == timer.initial_date
        next(timer)
        assert timer.now == 1.0

    def test__time_reset(self, timer):
        start_time = timer.initial_date
        assert timer.date_str == "2020-03-10"
        next(timer)
        next(timer)
        assert timer.date_str == "2020-03-11"
        next(timer)
        next(timer)
        assert timer.day == 2
        assert timer.date_str == "2020-03-12"
        timer.reset()
        assert timer.day == 0
        assert timer.shift == 0
        assert timer.previous_date == start_time
        assert timer.date_str == "2020-03-10"
        next(timer)
        next(timer)
        next(timer)
        next(timer)
        assert timer.day == 2

    def test__weekend_transition(self, timer):
        for _ in range(0, 8):  # 5 days for 3 time steps per day
            next(timer)
        assert timer.is_weekend is True
        assert timer.day_type == "weekend"
        assert timer.activities == ("household",)
        next(timer)
        assert timer.day_type == "weekend"
        assert timer.is_weekend is True
        assert timer.activities == ("household",)
        next(timer)
        assert timer.is_weekend is False
        assert timer.day_type == "weekday"
        assert timer.activities == (
            "school",
            "household",
        )

    def test__read_from_file(self):
        timer = Timer.from_file()
        assert timer.date_str == "2022-02-01"
        assert timer.total_days == 90
        assert timer.weekday_step_duration == {0: 24}
        assert timer.weekend_step_duration == {0: 24}
        assert timer.weekday_activities == {
            0: [
                "company",
                "school",
                "university",
                "pub",
                "grocery",
                "gym",
                "cinema",
                "visit",
                "care_home",
                "household",
            ]
        }
        assert timer.weekend_activities == {
            0: ["pub", "grocery", "gym", "cinema", "visit", "care_home", "household"]
        }
        assert timer.day_of_week == "Tuesday"
        assert timer.day_type == "weekday"

    def test__activity_hierarchy(self, timer):
        activities = [
            "household",
            "company",
            "pub",
            "school",
        ]
        sorted = timer._apply_activity_hierarchy(activities)
        assert sorted == [
            "school",
            "company",
            "pub",
            "household",
        ]

    def test_get_activity_order(self, timer):
        assert timer.day_of_week == "Tuesday"
        assert set(timer.get_activity_order()) == set(
            [
                "school",
                "household",
            ]
        )
        next(timer)
        assert timer.get_activity_order() == [
            "pub",
            "household",
        ]
        next(timer)
        assert timer.get_activity_order() == [
            "school",
            "household",
        ]
        while not timer.is_weekend:
            next(timer)
        assert timer.is_weekend
        assert timer.get_activity_order() == [
            "household",
        ]
        next(timer)
        assert timer.get_activity_order() == [
            "household",
        ]
