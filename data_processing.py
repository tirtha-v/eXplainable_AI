
def extract_post_hoc_data(data):
    """
    Extract systems with only O, H, and C1 category adsorbates.

    Parameters:
    data (list): List of system objects.

    Returns:
    list: List of filtered system objects with only O, H, and C1 adsorbates.
    """
    req_data = []
    for system in data:
        carbon_count = list(system.atomic_numbers).count(6)
        tags = list(system.tags.detach().numpy())
        atomic_numbers = system.atomic_numbers.detach().numpy()
        indices = [i for i, x in enumerate(tags) if x == 2]

        if all(item in [1, 6, 8] for item in atomic_numbers[indices]) and carbon_count <= 1:
            req_data.append(system)

    return req_data


def extract_sr_data(data):
    """
    Extract systems with only H adsorbates.

    Parameters:
    data (list): List of system objects.

    Returns:
    list: List of filtered system objects with only H adsorbates.
    """
    req_data = []
    for system in data:
        tags = list(system.tags.detach().numpy())
        atomic_numbers = system.atomic_numbers.detach().numpy()
        indices = [i for i, x in enumerate(tags) if x == 2]

        if all(item in [1] for item in atomic_numbers[indices]):
            req_data.append(system)

    return req_data


def extract_last_frame(req_data):
    """
    Extract the last frame id (fid) from each system id (sid).

    Parameters:
    req_data (list): List of filtered system objects.

    Returns:
    dict: Dictionary mapping system ID to the last frame ID.
    """
    sid_list = set(item.sid for item in req_data)
    last_frames = {sid: max(item.fid for item in req_data if item.sid == sid) for sid in tqdm(sid_list)}
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
    with open(mapping_path, 'rb') as file:
        mapping = pickle.load(file)

    # Extract SID and corresponding mp-ids
    mp_sid_dict = {}
    for sid, info in mapping.items():
        mp_sid_dict[sid] = info['bulk_mpid']

    mpid_list = list(mp_sid_dict.values())

    with MPRester(api_key=api_key) as mpr:
        summary = mpr.materials.summary.search(material_ids=mpid_list)
        electronic = mpr.materials.electronic_structure.search(material_ids=mpid_list)

    return summary, electronic, mp_sid_dict
