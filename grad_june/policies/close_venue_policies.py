from grad_june.policies import Policy, PolicyCollection


class CloseVenue(Policy):
    spec = "close_venue"

    def __init__(self, start_date, end_date, names, device):
        super().__init__(start_date=start_date, end_date=end_date, device=device)
        self.edge_type_to_close = set([f"{name}" for name in names])

    def apply(self, edge_types, timer):
        if self.is_active(timer.date):
            return [edge for edge in edge_types if edge not in self.edge_type_to_close]
        else:
            return edge_types


class CloseVenuePolicies(PolicyCollection):
    def apply(self, edge_types, timer):
        for policy in self.policies:
            edge_types = policy.apply(edge_types=edge_types, timer=timer)
        return edge_types
