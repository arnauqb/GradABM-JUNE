from operator import ne
import torch
import yaml

from grad_june.utils import parse_age_probabilities, parse_distribution
from grad_june.default_parameters import make_parameters
from grad_june.paths import default_config_path


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
        self.stages_ids = torch.arange(0, len(stages))

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
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        return cls(**params["symptoms"], device=params["system"]["device"])

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
                ret[i] = parse_distribution(stage_times[stage], device)
        return ret

    def _get_need_to_transition(self, current_stage, time_to_next_stage, time):
        """
        Gets a mask that is 1 for the agents that need to transition stage and 0
        for the ones that don't.
        """
        mask1 = time >= time_to_next_stage
        mask2 = current_stage < len(self.stages) - 1
        return mask1 * mask2

    def _get_prob_next_symptoms_stage(self, ages, stages):
        """
        Gets the probability of transitioning to the next symptoms stage,
        given the agents' ages and current stages.
        """
        probs = self.stage_transition_probabilities[stages, ages]
        return probs

    def sample_next_stage(
        self, ages, current_stage, next_stage, time_to_next_stage, time
    ):
        """
        Samples the next stage for each agent that needs updating.
        Returns the current stages, next stages, and the time to next stages.
        """
        # Check who has reached stage completion time and move them forward
        mask_transition = self._get_need_to_transition(
            current_stage, time_to_next_stage, time
        )
        n_agents = ages.shape[0]
        current_stage = current_stage - (current_stage - next_stage) * mask_transition
        # Sample possible next stages
        probs = self._get_prob_next_symptoms_stage(ages, current_stage.long())
        mask_symp_stage = torch.bernoulli(probs).to(torch.bool) # no dependence on parameters here.
        # These ones would recover
        mask_recovered_stage = ~mask_symp_stage
        for i in range(
            2, len(self.stages) - 1
        ):  # skip recovered, susceptible, and dead
            # Check people at this stage that need updating
            mask_stage = (current_stage == i)
            # this makes sure the gradient flows, since we lost it in the previous line.
            mask_stage = mask_stage * current_stage / i 
            mask_updating = mask_stage * mask_transition

            # These people progress to another disease stage
            mask_symp = mask_updating * mask_symp_stage
            n_symp = mask_symp.max()  # faster than sum
            if n_symp > 0:
                next_stage = next_stage + mask_symp
                time_to_next_stage = (
                    time_to_next_stage
                    + self.stage_transition_times[i].rsample((n_agents,)) * mask_symp
                )

            # These people will recover
            mask_rec = mask_updating * mask_recovered_stage
            n_rec = mask_rec.max()
            if n_rec > 0:
                next_stage = next_stage - next_stage * mask_rec  # Set to 0
                time_to_next_stage = (
                    time_to_next_stage
                    + self.recovery_times[i].rsample((n_agents,)) * mask_rec
                )
        return current_stage, next_stage, time_to_next_stage


class SymptomsUpdater(torch.nn.Module):
    """
    A PyTorch module for updating symptoms of agents in an epidemic simulation.

    Args:
        symptoms_sampler (SymptomsSampler): A sampler for generating new symptoms based on current symptoms and other factors.

    Attributes:
        symptoms_sampler (SymptomsSampler): A sampler for generating new symptoms based on current symptoms and other factors.

    Methods:
        from_file(cls, fpath=default_config_path): Class method for creating a SymptomsUpdater instance from a YAML file.
        from_parameters(cls, params): Class method for creating a SymptomsUpdater instance from a dictionary of parameters.
        forward(self, data, timer, new_infected): Method for updating symptoms of agents based on current symptoms and other factors.
        stages_ids(self): Property for getting the IDs of all possible symptom stages.

    Raises:
        TypeError: If symptoms_sampler is not an instance of SymptomsSampler.
        KeyError: If data does not contain the "agent" key.
        KeyError: If symptoms does not contain the "current_stage", "next_stage", or "time_to_next_stage" keys.

    Returns:
        A SymptomsUpdater instance.
    """

    def __init__(self, symptoms_sampler):
        super().__init__()
        if not isinstance(symptoms_sampler, SymptomsSampler):
            raise TypeError("symptoms_sampler must be an instance of SymptomsSampler.")
        self.symptoms_sampler = symptoms_sampler

    @classmethod
    def from_file(cls, fpath=default_config_path):
        """
        Class method for creating a SymptomsUpdater instance from a YAML file.

        Args:
            fpath (str): The path to the YAML file containing the parameters.

        Raises:
            FileNotFoundError: If the file at fpath does not exist.
            yaml.YAMLError: If the file at fpath is not a valid YAML file.
            KeyError: If the YAML file does not contain the "symptoms_sampler" key.

        Returns:
            A SymptomsUpdater instance.
        """
        try:
            with open(fpath, "r") as f:
                params = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at {fpath}.")
        except yaml.YAMLError:
            raise yaml.YAMLError(f"Invalid YAML file at {fpath}.")
        ss = SymptomsSampler.from_parameters(params)
        return cls(symptoms_sampler=ss)

    @classmethod
    def from_parameters(cls, params):
        """
        Class method for creating a SymptomsUpdater instance from a dictionary of parameters.

        Args:
            params (dict): A dictionary of parameters.

        Raises:

        Returns:
            A SymptomsUpdater instance.
        """
        ss = SymptomsSampler.from_parameters(params)
        return cls(symptoms_sampler=ss)

    def forward(self, data, timer, new_infected):
        """
        Method for updating symptoms of agents based on current symptoms and other factors.

        Args:
            data (dict): A dictionary containing information about the agents.
            timer (Timer): A timer object for keeping track of time in the simulation.
            new_infected (int): The number of new agents who have become infected.

        Raises:
            KeyError: If data does not contain the "agent" key.
            KeyError: If symptoms does not contain the "current_stage", "next_stage", or "time_to_next_stage" keys.

        Returns:
            A dictionary containing the updated symptoms of the agents.
        """
        try:
            symptoms = data["agent"].symptoms
        except KeyError:
            raise KeyError("data must contain the 'agent' key.")
        if not all(key in symptoms for key in ["current_stage", "next_stage", "time_to_next_stage"]):
            raise KeyError("symptoms must contain the 'current_stage', 'next_stage', and 'time_to_next_stage' keys.")
        time = timer.now
        symptoms["next_stage"] = symptoms["next_stage"] + new_infected * (
            2.0 - symptoms["next_stage"]
        )
        symptoms["time_to_next_stage"] = symptoms[
            "time_to_next_stage"
        ] + new_infected * (time - symptoms["time_to_next_stage"])
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

    @property
    def stages_ids(self):
        """
        Property for getting the IDs of all possible symptom stages.

        Returns:
            A list of integers representing the IDs of all possible symptom stages.
        """
        return self.symptoms_sampler.stages_ids
