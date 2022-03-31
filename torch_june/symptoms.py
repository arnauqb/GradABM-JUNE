import torch
from torch import distributions as dist

from torch_june.utils import parse_age_probabilities
from torch_june.default_parameters import make_parameters


class SymptomsSampler:
    def __init__(
        self,
        stages,
        stage_transition_probabilities,
        stage_transition_times,
        recovery_times,
        device,
    ):
        self.stages = stages
        self.stage_transition_probabilities = (
            self._parse_stage_transition_probabilities(
                stage_transition_probabilities, device=device
            )
        )
        self.stage_transition_times = self._parse_stage_times(
            stage_transition_times, device=device
        )
        self.recovery_times = self._parse_stage_times(recovery_times, device=device)

    @classmethod
    def from_dict(cls, input_dict, device="cpu"):
        return cls(**input_dict, device=device)

    @classmethod
    def from_default_parameters(cls, device="cpu"):
        return cls(**make_parameters()["symptoms"], device=device)

    def _parse_stage_transition_probabilities(
        self, stage_transition_probabilities, device
    ):
        ret = torch.zeros((len(self.stages), 100), device=device)
        for i, stage in enumerate(self.stages):
            if stage not in stage_transition_probabilities:
                continue
            ret[i] = torch.tensor(
                parse_age_probabilities(stage_transition_probabilities[stage]),
                dtype=torch.float32,
                device=device,
            )
        return ret

    def _parse_stage_times(self, stage_times, device):
        ret = {}
        for i, stage in enumerate(self.stages):
            if stage not in stage_times:
                ret[i] = None
            else:
                dist_name = stage_times[stage].pop("dist")
                dist_class = getattr(dist, dist_name)
                input = {
                    key: torch.tensor(value, device=device, dtype=torch.float)
                    for key, value in stage_times[stage].items()
                }
                ret[i] = dist_class(**input)
        return ret

    def _get_need_to_transition(self, current_stage, time_to_next_stage, time):
        mask1 = time >= time_to_next_stage
        mask2 = current_stage < len(self.stages) - 1
        return mask1 * mask2

    def _get_prob_next_symptoms_stage(self, ages, stages):
        probs = self.stage_transition_probabilities[stages, ages]
        return probs

    #@profile
    def sample_next_stage(
        self, ages, current_stage, next_stage, time_to_next_stage, time
    ):
        # new_next_stage = next_stage.clone()
        # new_times = time_to_next_stage.clone()

        # Check who has reached stage completion time and move them forward
        mask_transition = self._get_need_to_transition(
            current_stage, time_to_next_stage, time
        )
        current_stage[mask_transition] = next_stage[mask_transition]
        # Sample possible next stages
        probs = self._get_prob_next_symptoms_stage(ages, current_stage)
        mask_symp_stage = torch.bernoulli(probs).to(torch.bool)
        # These ones would recover
        mask_recovered_stage = ~mask_symp_stage
        for i in range(
            2, len(self.stages) - 1
        ):  # skip recovered, susceptible, and dead
            # Check people at this stage that need updating
            mask_stage = current_stage == i
            mask_updating = mask_stage * mask_transition

            # These people progress to another disease stage
            mask_symp = mask_updating * mask_symp_stage
            n_symp = mask_symp.sum()
            if n_symp > 0:
                next_stage[mask_symp] = next_stage[mask_symp] + 1
                time_to_next_stage[mask_symp] = time_to_next_stage[
                    mask_symp
                ] + self.stage_transition_times[i].sample((n_symp.item(),))

            # These people will recover
            mask_rec = mask_updating * mask_recovered_stage
            n_rec = mask_rec.sum()
            if n_rec > 0:
                next_stage[mask_rec] = torch.zeros(
                    n_rec, dtype=torch.long, device=next_stage.device
                )
                time_to_next_stage[mask_rec] = time_to_next_stage[
                    mask_rec
                ] + self.recovery_times[i].sample((n_rec.item(),))
        return current_stage, next_stage, time_to_next_stage


class SymptomsUpdater(torch.nn.Module):
    def __init__(self, symptoms_sampler):
        super().__init__()
        self.symptoms_sampler = symptoms_sampler

    def forward(self, data, timer, new_infected):
        time = timer.now
        symptoms = data["agent"].symptoms
        mask = new_infected.bool()
        symptoms["next_stage"][mask] = 2
        symptoms["time_to_next_stage"][mask] = time
        (
            current_stage,
            next_stage,
            time_to_next_stage,
        ) = self.symptoms_sampler.sample_next_stage(
            ages=data["agent"].age,
            current_stage=symptoms["current_stage"],
            next_stage=symptoms["next_stage"],
            time_to_next_stage=symptoms["time_to_next_stage"],
            time=time,
        )
        symptoms["current_stage"] = current_stage
        symptoms["next_stage"] = next_stage
        symptoms["time_to_next_stage"] = time_to_next_stage
        return symptoms
