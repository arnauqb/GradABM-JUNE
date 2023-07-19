from grad_june.demographics import get_people_per_area

class TestGetPeoplePerArea:
    def test__get_people_per_area(self, data):
        agent_ids = data["agent"].id
        area_ids = data["agent"].area_id
        people_per_area = get_people_per_area(agent_ids, area_ids)
        assert len(people_per_area) == 10
        assert sum([len(people_per_area[k]) for k in people_per_area]) == 100
        district_ids = data["agent"].district_id
        people_per_district = get_people_per_area(agent_ids, district_ids)
        assert len(people_per_district) == 3
        assert sum([len(people_per_district[k]) for k in people_per_district]) == 100