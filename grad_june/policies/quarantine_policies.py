import torch

from grad_june.policies.policies import PolicyCollection, Policy


class Quarantine(Policy):
    spec = "quarantine"

    def __init__(self, start_date, end_date, stage_threshold, device):
        super().__init__(start_date=start_date, end_date=end_date, device=device)
        self.stage_threshold = stage_threshold

    def apply(self, symptom_stages, timer):
        if self.is_active(timer.date):
            ret = (symptom_stages < self.stage_threshold).to(torch.float)
        else:
            ret = torch.ones(symptom_stages.shape, device=symptom_stages.device)
        return ret


class QuarantinePolicies(PolicyCollection):
    def __init__(self, policies):
        super().__init__(policies)
        self.quarantine_mask = 1.0

    def apply(self, symptom_stages, timer):
        self.quarantine_mask = torch.ones(
            symptom_stages.shape, device=symptom_stages.device
        )
        for policy in self.policies:
            self.quarantine_mask = self.quarantine_mask * policy.apply(
                symptom_stages=symptom_stages, timer=timer
            )
