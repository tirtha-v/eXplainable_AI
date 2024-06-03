""" 
Data processing task for the OC20 dataset, we need to perform the following steps:
1. Extract systems containing O, H, and C1 adsorbates.
2. Extract the last frames of the given systems with multiple trajectories.
"""


def extract_post_hoc_data(data):
    req_data = []
    for j in range(len(data)):
        req_data = list(req_data)
        count_carbon = list(data[j].atomic_numbers).count(6)
        tags = list(data[j].tags.detach().numpy())
        atomic_numbers = data[j].atomic_numbers.detach().numpy()
        indices = [i for i, x in enumerate(tags) if x == 2]
        if (
            all(item in [1, 6, 8] for item in atomic_numbers[indices])
            and count_carbon <= 1
        ):
            req_data.append(data[j])

    return req_data


def extract_sr_data(data):
    req_data = []
    for j in range(len(data)):
        req_data = list(req_data)
        tags = list(data[j].tags.detach().numpy())
        atomic_numbers = data[j].atomic_numbers.detach().numpy()
        indices = [i for i, x in enumerate(tags) if x == 2]
        if all(item in [1] for item in atomic_numbers[indices]):
            req_data.append(data[j])
    return req_data


def extract_last_frame(req_data):
    sid_list = set(item.sid for item in req_data)
    last_frames = {
        u_sid: max(item.fid for item in req_data if item.sid == u_sid)
        for u_sid in tqdm(sid_list)
    }
    return last_frames


def get_material_data(mapping_path, api_key):
    """
    Summary and electronic structure data from the Materials Project API.

    Parameters:
    mapping_path (str): Path to the pickle file containing the system ID to Materials Project ID mapping.
    api_key (str): Your Materials Project API key.

    Returns:
    tuple: A tuple containing summary and electronic structure data.
    """
    # Load data from the pickle file
    with open(mapping_path, "rb") as file:
        mapping = pickle.load(file)

    # Extract SID and corresponding mp-ids
    mp_sid_dict = {}
    for sid, info in mapping.items():
        mp_sid_dict[sid] = info["bulk_mpid"]

    mpid_list = list(mp_sid_dict.values())

    with MPRester(api_key=api_key) as mpr:
        summary = mpr.materials.summary.search(material_ids=mpid_list)
        electronic = mpr.materials.electronic_structure.search(material_ids=mpid_list)

    return summary, electronic
