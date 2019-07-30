from dscribe.descriptors import MBTR


def make_mbtr_desc(atomic_numbers, max_atomic_number, min_atomic_number,
                 min_distance, decay_factor):
    # print("Generating MBTRs with the following specs:\n {}".format(kwargs))

    # atomic_numbers = kwargs['atomic_numbers']
    # min_atomic_number = kwargs['min_atomic_number']
    # max_atomic_number = kwargs['max_atomic_number']
    # min_distance = kwargs['min_distance']
    # decay_factor = kwargs['decay_factor']

    mbtr_desc = MBTR(
        species=atomic_numbers,
        periodic=False,
        k1={
            "geometry": {
                "function": "atomic_number"
            },
            "grid": {
                "min": min_atomic_number,
                "max": max_atomic_number,
                "sigma": 0.2,
                "n": 200
            }
        },
        k2={
            "geometry": {
                "function": "inverse_distance"
            },
            "grid": {
                "min": 0,
                "max": 1 / min_distance,
                "sigma": 0.02,
                "n": 200,
            },
            "weighting": {
                "function": "exponential",
                "scale": decay_factor,
                "cutoff": 1e-3
            },
        },
        k3={
            "geometry": {
                "function": "angle"
            },
            "grid": {
                "min": -1.0,
                "max": 1.0,
                "sigma": 0.09,
                "n": 200,
            },
            "weighting": {
                "function": "exponential",
                "scale": decay_factor,
                "cutoff": 1e-3
            },
        },
        flatten = False
    )

    return mbtr_desc
