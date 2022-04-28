import torch
import numpy as np

from torch_june.policies import Quarantine, QuarantinePolicies, quarantine_policies
from torch_june.timer import Timer
from torch_june.message_passing import InfectionPassing


class TestQuarantine:
    def test__match_symptoms(self):
        symptom_stages = torch.tensor([0, 1, 2, 3, 4])
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company", "household"),),
            weekend_activities=(("company", "household"),),
        )
        quarantine = Quarantine(
            stage_threshold=3, start_date="2022-02-01", end_date="2022-02-05"
        )
        quarantine_array = quarantine.apply(timer=timer, symptom_stages=symptom_stages)
        assert (quarantine_array == torch.tensor([1, 1, 1, 0, 0])).all()

    def test__integration(self, inf_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company",),),
            weekend_activities=(("company", "household"),),
        )
        inf_data["agent"]["transmission"] = inf_data["agent"]["transmission"] + 1.0
        n_agents = inf_data["agent"].id.shape[0]
        inf_data["agent"]["symptoms"]["current_stage"] = 5 * torch.ones(n_agents)
        mp = InfectionPassing(
            log_beta_company=torch.tensor(3.0), log_beta_household=torch.tensor(3.0)
        )
        quarantine = Quarantine(
            stage_threshold=3, start_date="2022-02-01", end_date="2022-03-15"
        )
        ret = mp(
            data=inf_data,
            timer=timer,
            quarantine_policies=QuarantinePolicies([quarantine]),
        )
        assert np.isclose(
            ret.sum().detach(), n_agents
        )  # No-one gets infected since they all quarantine
        while not timer.is_weekend:
            next(timer)
        ret = mp(
            data=inf_data,
            timer=timer,
            quarantine_policies=QuarantinePolicies([quarantine]),
        )
        assert np.isclose(
            ret.sum().detach().item(), 10.0
        )  # people living in the same household get infected, seed survives
