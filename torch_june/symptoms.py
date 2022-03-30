import torch
import numpy as np
from torch import distributions as dist

from torch_june.utils import parse_age_probabilities


class SymptomsSampler:
    def __init__(
        self, stages, transition_probabilities, symptom_transition_times, recovery_times
    ):
        self.stages = stages
        self.transition_probabilities = self._parse_transition_probabilities(
            transition_probabilities
        )
        self.symptom_transition_times = self._parse_transition_times(
            symptom_transition_times
        )
        self.recovery_times = self._parse_transition_times(recovery_times)

    @classmethod
    def from_dict(cls, input_dict):
        return cls(**input_dict)

    def _parse_transition_probabilities(self, transition_probabilities):
        ret = torch.zeros((len(self.stages), 100))
        for i, stage in enumerate(self.stages):
            if stage not in transition_probabilities:
                continue
            ret[i] = torch.tensor(
                parse_age_probabilities(transition_probabilities[stage]),
                dtype=torch.float32,
            )
        return ret

    def _parse_transition_times(self, transition_times):
        ret = {}
        for i, stage in enumerate(self.stages):
            if stage not in transition_times:
                ret[i] = None
            else:
                dist_name = transition_times[stage].pop("distribution")
                dist_class = getattr(dist, dist_name)
                ret[i] = dist_class(**transition_times[stage])
        return ret

    def _get_need_to_transition(self, current_stages, time_to_next_stages, time):
        mask1 = time >= time_to_next_stages
        mask2 = current_stages < len(self.stages) - 1
        return mask1 * mask2

    def _get_prob_next_symptoms_stage(self, ages, current_stages):
        probs = self.transition_probabilities[current_stages, ages]
        return probs

    def sample_next_stages(self, ages, current_stages, time_to_next_stages, time):
        new_stages = current_stages.clone()
        new_times = time_to_next_stages.clone()
        probs = self._get_prob_next_symptoms_stage(ages, current_stages)
        mask_transition = self._get_need_to_transition(current_stages, time_to_next_stages, time)
        #print(probs)
        mask_symp_transition = torch.bernoulli(probs).to(torch.bool)
        mask_recovered_transition = ~mask_symp_transition
        for i, stage in enumerate(self.stages[:-1]): # skip dead
            #print(stage)
            mask_stage = current_stages == i
            mask_updating = mask_transition * mask_stage
            #print("needs updating")
            #print(mask_updating)

            mask_symp = mask_updating * mask_symp_transition
            #print("wins roll")
            #print(mask_symp)
            n_symp = mask_symp.sum()
            if n_symp > 0:
                if i < len(self.stages) - 2:
                    new_times[mask_symp] = new_times[mask_symp] + self.symptom_transition_times[i+1].sample((n_symp.item(),))
                new_stages[mask_symp] = new_stages[mask_symp] + 1

            mask_rec = mask_updating * mask_recovered_transition
            n_rec = mask_rec.sum()
            if n_rec > 0:
                new_times[mask_rec] = new_times[mask_rec] + self.recovery_times[i].sample((n_rec.item(),))
                new_stages[mask_rec] = torch.zeros(n_rec)
            #print("\n")
        #print("---")
        return new_stages, new_times

        





