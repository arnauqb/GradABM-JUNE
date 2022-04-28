import torch

from torch_june.policies.policies import PolicyCollection, Policy


class QuarantinePolicy(Policy):
    def __init__(self, start_date, end_date, stage_threshold):
        super().__init__(start_date=start_date, end_date=end_date)
        self.stage_threshold = stage_threshold

    def apply(self, symptom_stages, timer):
        if self.is_active(timer.date):
            ret = (symptom_stages < self.stage_threshold).to(torch.float)
        else:
            ret = torch.ones(symptom_stages.shape, device=symptom_stages.device)
        return ret


class QuarantinePolicies(PolicyCollection):
    def apply(self, symptom_stages, timer):
        ret = torch.ones(symptom_stages.shape, device=symptom_stages.device)
        for policy in self.policies:
            ret = ret * policy.apply(symptom_stages=symptom_stages, timer=timer)
        return ret
