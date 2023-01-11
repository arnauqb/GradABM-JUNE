import numpy as np


def convert_lognormal_parameters(mean, std):
    """
    Converts mean and std to loc and scale parmaeters for the LogNormal.
    """
    loc = np.log(mean**2 / np.sqrt(mean**2 + std**2))
    scale = np.sqrt(np.log(1 + std**2 / mean**2))
    return loc, scale


def make_parameters():
    ret = {}

    """
    Symptoms Parameters
    Taken from the Covasim paper
    """
    symptoms = {}
    stages = [
        "recovered",
        "susceptible",
        "exposed",
        "infectious",
        "symptomatic",
        "severe",
        "critical",
        "dead",
    ]
    symptoms["stages"] = stages

    # Symptom transition probabilities
    tprobs = {}
    tprobs["recovered"] = {"0-100": 0.0}
    tprobs["susceptible"] = {"0-100": 0.0}
    tprobs["exposed"] = {"0-100": 1.0}
    tprobs["infectious"] = {
        "0-10": 0.5,
        "10-20": 0.55,
        "20-30": 0.6,
        "30-40": 0.65,
        "40-50": 0.7,
        "50-60": 0.75,
        "60-70": 0.8,
        "70-80": 0.85,
        "80-90": 0.9,
        "90-100": 0.9,
    }
    tprobs["symptomatic"] = {
        "0-10": 0.0005,
        "10-20": 0.00165,
        "20-30": 0.00720,
        "30-40": 0.02080,
        "40-50": 0.03430,
        "50-60": 0.07650,
        "60-70": 0.13280,
        "70-80": 0.20655,
        "80-90": 0.24570,
        "90-100": 0.24570,
    }
    tprobs["severe"] = {
        "0-10": 0.00003,
        "10-20": 0.00008,
        "20-30": 0.00036,
        "30-40": 0.00104,
        "40-50": 0.00216,
        "50-60": 0.00933,
        "60-70": 0.03639,
        "70-80": 0.08923,
        "80-90": 0.17420,
        "90-100": 0.17420,
    }
    tprobs["critical"] = {
        "0-10": 0.00002,
        "10-20": 0.00002,
        "20-30": 0.0001,
        "30-40": 0.00032,
        "40-50": 0.00098,
        "50-60": 0.00265,
        "60-70": 0.00766,
        "70-80": 0.02439,
        "80-90": 0.08292,
        "90-100": 0.16190,
    }
    # Convert to relative probs to the previous stage.
    tprobs["critical"] = {
        key: tprobs["critical"][key] / tprobs["severe"][key]
        for key in tprobs["critical"]
    }
    tprobs["severe"] = {
        key: tprobs["severe"][key] / tprobs["symptomatic"][key]
        for key in tprobs["severe"]
    }
    tprobs["symptomatic"] = {
        key: tprobs["symptomatic"][key] / tprobs["infectious"][key]
        for key in tprobs["symptomatic"]
    }
    symptoms["stage_transition_probabilities"] = tprobs

    # Symptom transition times
    ttimes = {}
    loc, scale = convert_lognormal_parameters(4.5, 1.5)
    ttimes["exposed"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(1.1, 0.9)
    ttimes["infectious"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(6.6, 4.9)
    ttimes["symptomatic"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(1.5, 2.0)
    ttimes["severe"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(10.7, 4.8)
    ttimes["critical"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    symptoms["stage_transition_times"] = ttimes

    # Recovery times
    rtimes = {}
    # loc, scale = convert_lognormal_parameters(4.5, 1.5)
    # rtimes["recovered"] = {"dist": "LogNormal", "loc": loc, "scale" : scale}
    # loc, scale = convert_lognormal_parameters(4.5, 1.5)
    # rtimes["susceptible"] = {"dist": "LogNormal", "loc": loc, "scale" : scale}
    loc, scale = convert_lognormal_parameters(4.5, 1.5)
    rtimes["exposed"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(8.0, 2.0)
    rtimes["infectious"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(8.0, 2.0)
    rtimes["symptomatic"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(18.1, 6.3)
    rtimes["severe"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    loc, scale = convert_lognormal_parameters(18.1, 6.3)
    rtimes["critical"] = {"dist": "LogNormal", "loc": loc, "scale": scale}
    symptoms["recovery_times"] = rtimes
    ret["symptoms"] = symptoms

    # policy parameters
    policies = {}
    policies["interaction"] = {
        "social_distancing": {
            1: {
                "start_date": "2020-03-16",
                "end_date": "2020-03-24",
                "beta_factors": {
                    "leisure": 0.65,
                    "care_home": 0.65,
                    "school": 0.65,
                    "university": 0.65,
                    "company": 0.65,
                },
            },
            2: {
                "start_date": "2020-03-24",
                "end_date": "2020-05-11",
                "beta_factors": {
                    "leisure": 0.45,
                    "care_home": 0.45,
                    "school": 0.45,
                    "university": 0.45,
                    "company": 0.45,
                },
            },
            3: {
                "start_date": "2020-05-11",
                "end_date": "2020-07-04",
                "beta_factors": {
                    "leisure": 0.50,
                    "care_home": 0.50,
                    "school": 0.50,
                    "university": 0.50,
                    "company": 0.50,
                },
            },
        }
    }
    policies["close_venue"] = {
        "close_venue": {
            1: {
                "start_date": "2020-03-21",
                "end_date": "2020-07-04",
                "names": ["leisure", "school"],
            },
        }
    }
    policies["quarantine"] = {
        "quarantine": {
            "start_date": "2020-03-16",
            "end_date": "9999-03-24",
            "stage_threshold": 4,
        },
    }
    ret["policies"] = policies

    return ret
