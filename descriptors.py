from utils import pyg2atoms, SiteAnalyzer
from data_processing import extract_data, extract_last_frame, get_material_data


def get_local_e(data):
    """
    Calculate the local electronegativity for each system in new_data.

    Parameters:
    data (list): List of system objects, each having attributes like 'tags', 'atomic_numbers', 'natoms', 'sid', etc.

    Returns:
    dict: Dictionary mapping system ID to local electronegativity value.
    """
    # Load electronegativity scale for atomic numbers from file
    scale = np.loadtxt("pauling_electronegativity.txt", delimiter="\t")
    scale_t = tf.convert_to_tensor(scale)

    local_e = {}

    # Iterate through each system in new_data
    for key in tqdm(range(len(data))):
        system = data[key]

        # Count the number of atoms with tag value 1 (N Value)
        count = np.count_nonzero(system.tags == 1)

        # Calculate Xj values
        electro = []
        for atom in range(system.natoms):
            if system.tags[atom] == 1:
                atomic_number = system.atomic_numbers[atom]
                index = np.where(scale_t[:, 0] == atomic_number)[0]
                if index.size > 0:  # Ensure index is found
                    electro_value = scale_t[int(index), 1] ** (1 / count)
                    electro.append(electro_value)

        # Calculate local electronegativity by multiplying all Xj ^ (1/N) values
        if electro:
            value = np.prod(electro)
            local_e[system.sid] = value

    return local_e


def get_ads_e(data):
    """
    Calculate the average electronegativity of adsorbate for each system in data.

    Parameters:
    data (list): List of system objects, each having attributes like 'tags', 'atomic_numbers', 'natoms', 'sid', etc.

    Returns:
    dict: Dictionary mapping system ID to the average electronegativity of its adsorbate molecule.
    """
    # Load electronegativity scale for atomic numbers from file
    scale = np.loadtxt("pauling_electronegativity.txt", delimiter="\t")
    scale_t = tf.convert_to_tensor(scale)

    avg_elec_adsorbate = {}

    # Iterate through each system in data
    for j in tqdm(range(len(data))):
        system = data[j]

        # Calculate electronegativity values for atoms tagged as adsorbates (tag value 2)
        ads_electro = []
        for i in range(system.natoms):
            if system.tags[i] == 2:
                atomic_number = system.atomic_numbers[i]
                index = np.where(scale_t[:, 0] == atomic_number)[0]
                if index.size > 0:  # Ensure index is found
                    ads_electro.append(scale_t[int(index), 1])

        # Calculate average electronegativity if ads_electro is not empty
        if ads_electro:
            avg_elec_adsorbate[system.sid] = mean(ads_electro)
        else:
            avg_elec_adsorbate[system.sid] = None  # Handle case with no adsorbates

    return avg_elec_adsorbate


def get_slab_e(data):
    """
    Calculate the average electronegativity of the slab atoms in the catalyst for each system in the dataset.

    Parameters:
    data (list): List of system objects, each having attributes like 'tags', 'atomic_numbers', 'natoms', 'sid', etc.

    Returns:
    dict: Dictionary mapping system ID to the average electronegativity of its slab atoms.
    """
    # Load electronegativity scale for atomic numbers from file
    scale = np.loadtxt("pauling_electronegativity.txt", delimiter="\t")
    scale_t = tf.convert_to_tensor(scale)

    avg_elec_slab = {}

    # Iterate through each system in data
    for j in tqdm(range(len(data))):
        system = data[j]
        number_of_atoms = system.natoms  # Number of atoms in the system

        # Calculate electronegativity values for atoms tagged as interacting slab atoms (tag value 1)
        surf_electro = []
        for i in range(number_of_atoms):
            if system.tags[i] == 1:
                atomic_number = system.atomic_numbers[i]
                index = np.where(scale_t[:, 0] == atomic_number)[0]
                if index.size > 0:  # Ensure index is found
                    surf_electro.append(scale_t[int(index), 1])

        # Calculate average electronegativity if surf_electro is not empty
        if surf_electro:
            avg_elec_slab[system.sid] = mean(surf_electro)
        else:
            avg_elec_slab[
                system.sid
            ] = None  # Handle case with no interacting slab atoms

    return avg_elec_slab


def get_eff_coord(data):
    """
    Calculate the average coordination number slab elements in catalyst for each system in data.

    Parameters:
    data (list): List of system objects, each having attributes.

    Returns:
    dict: Dictionary mapping system ID to its effective coordination number.
    """
    effective_coord_number = {}

    # Iterate through each system in data
    for i in tqdm(range(len(data))):
        # Convert pyg data to ase atoms object
        atoms = pyg2atoms(data[i])
        atom_obj = SiteAnalyzer(atoms)

        # Check for the presence of slab atoms
        check_slab = np.count_nonzero(atom_obj._find_binding_atoms_from_center())

        if check_slab != 0:
            # Get the slab atom indices
            slab_atoms_info = atom_obj._find_binding_atoms_from_center()[0]
            slab_atoms = len(slab_atoms_info["slab_atom_idxs"])

            indices = []
            coordination_numbers = []

            for idx in range(slab_atoms):
                slab_atom_idx = slab_atoms_info["slab_atom_idxs"][idx]
                indices.append(slab_atom_idx)

                # Get the connectivity information
                connectivity = atom_obj._get_connectivity(atoms, 1.0)
                coord_number = np.count_nonzero(connectivity[slab_atom_idx])
                coordination_numbers.append(coord_number)

            # Calculate average coordination number
            avg_coord_number = np.average(coordination_numbers)
            effective_coord_number[data[i].sid] = avg_coord_number
        else:
            effective_coord_number[data[i].sid] = 0.0

    return effective_coord_number


def get_center_coord(data):
    """
    Calculate the coordination number of center atom in adsorbate molecule for each system in data.

    Parameters:
    data (list): List of system objects, each having attributes.

    Returns:
    dict: Dictionary mapping system ID to the number of slab atoms coordinating with the center atom.
    """
    center_coord_number = {}

    # Iterate through each system in data
    for i in tqdm(range(len(data))):
        # Convert pyg data to ase atoms object
        atoms = pyg2atoms(data[i])
        atom_obj = SiteAnalyzer(atoms)

        # Check for the presence of binding atoms with the center atom
        binding_atoms_info = atom_obj._find_binding_atoms_from_center()
        num_binding_atoms = len(binding_atoms_info)

        if num_binding_atoms != 0:
            slab_atom_indices = binding_atoms_info[0]["slab_atom_idxs"]
            center_coord_number[data[i].sid] = len(slab_atom_indices)
        else:
            center_coord_number[data[i].sid] = 0

    return center_coord_number


def get_sum_atomic_adsorbate(data):
    """
    Calculate the sum of atomic numbers for adsorbate atoms in each system.

    Parameters:
    data (list): List of system objects.

    Returns:
    dict: Dictionary containing the sum of atomic numbers for adsorbate atoms in each system.
    """
    sum_atomic_adsorbate = {}

    for item in tqdm(data):
        atom_sum = sum(
            item.atomic_numbers[i] for i in range(item.natoms) if item.tags[i] == 2
        )
        sum_atomic_adsorbate[item.sid] = atom_sum

    return sum_atomic_adsorbate


def get_num_adsorbate(data):
    """
    Calculate the number of adsorbate atoms in each system.

    Parameters:
    data (list): List of system objects.

    Returns:
    dict: Dictionary containing the number of adsorbate atoms in each system.
    """
    num_adsorbate = {}

    for item in tqdm(data):
        count = sum(1 for i in range(item.natoms) if item.tags[i] == 2)
        num_adsorbate[item.sid] = count

    return num_adsorbate


def get_sites(data):
    """
    Calculate the type of adsorption site for each system in data.

    Parameters:
    data (list): List of system objects, each having attributes.

    Returns:
    dict: Dictionary mapping system ID to the site type.
    """
    sites = {}

    # Iterate through each system in data
    for i in tqdm(range(len(data))):
        # Convert pyg data to atoms object
        atoms = pyg2atoms(data[i])
        atom_obj = SiteAnalyzer(atoms)

        # Get the site type
        center_site_types = atom_obj.get_center_site_type()

        if center_site_types:
            sites[data[i].sid] = center_site_types[0]
        else:
            sites[data[i].sid] = 0

    print(len(sites))
    return sites


def get_density(summary, mp_sid_dict):
    """
    Get density values for required mpids.

    Parameters:
    summary (list): List of summary data obtained from the Materials Project API.
    mp_sid_dict (dict): Dictionary mapping system IDs to their Materials Project IDs.

    Returns:
    dict: Dictionary containing density values for the corresponding mpids.
    """
    density = {}
    for item in tqdm(summary):
        density[str(item.material_id)] = item.density

    final_density = {}
    for sid, mp in mp_sid_dict.items():
        for mpi, d in density.items():
            if mpi == mp:
                final_density[sid] = d

    return final_density


def get_H_f(summary, mp_sid_dict):
    """
    Get formation energy values for required mpids.

    Parameters:
    summary (list): List of summary data obtained from the Materials Project API.
    mp_sid_dict (dict): Dictionary mapping system IDs to their Materials Project IDs.

    Returns:
    dict: Dictionary containing formation energy values for the corresponding mpids.
    """
    form_eng = {}
    for item in tqdm(summary):
        form_eng[str(item.material_id)] = item.formation_energy_per_atom

    final_form_eng = {}
    for sid, mp in mp_sid_dict.items():
        for mpi, bg in form_eng.items():
            if mpi == mp:
                final_form_eng[sid] = bg

    return final_form_eng


def get_band_gap(electronic, mp_sid_dict):
    """
    Get band gap values for required mpids.

    Parameters:
    electronic (list): List of electronic structure data obtained from the Materials Project API.
    mp_sid_dict (dict): Dictionary mapping system IDs to their Materials Project IDs.

    Returns:
    dict: Dictionary containing band gap values for the corresponding mpids.
    """
    band_gaps = {}
    for item in tqdm(electronic):
        band_gaps[str(item.material_id)] = item.band_gap

    final_band_gap = {}
    for sid, mp in mp_sid_dict.items():
        for mpi, bg in band_gaps.items():
            if mpi == mp:
                final_band_gap[sid] = bg

    return final_band_gap


def get_space_groups(summary, mp_sid_dict):
    """
    Get space group numbers for required mpids.

    Parameters:
    summary (list): List of summary data obtained from the Materials Project API.
    mp_sid_dict (dict): Dictionary mapping system IDs to their Materials Project IDs.

    Returns:
    dict: Dictionary containing space group numbers for the corresponding mpids.
    """
    space_grp = {}
    for item in tqdm(summary):
        space_grp[str(item.material_id)] = item.symmetry.number

    final_space_grp = {}
    for sid, mp in mp_sid_dict.items():
        for mpi, sg in space_grp.items():
            if mpi == mp:
                final_space_grp[sid] = sg

    return final_space_grp


def get_miller_indices(mapping_path, data):
    """
    Get Miller indices for each system.

    Parameters:
    mapping_path (str): Path to the pickle file containing the system ID to Materials Project ID mapping.
    new_data (list): List of system objects.

    Returns:
    dict: Dictionary containing Miller indices for each system.
    """
    # Load the data from the pickle file
    with open(mapping_path, "rb") as file:
        mapping = pickle.load(file)

    miller_idx = {}
    for item in tqdm(data):
        random_sid = "random" + str(item.sid)
        m_idx = mapping[random_sid]["miller_index"]
        m_idx_str = "".join(str(x) for x in m_idx)
        miller_idx[item.sid] = int(m_idx_str)

    return miller_idx


def get_adsorption_energy(data, ref_path):
    """
    Calculate adsorption energy for each system ID using reference energy from a pickle file.

    Parameters:
    data (list): List of system objects.
    ref_path (str): Path to the pickle file containing reference energy.

    Returns:
    dict: Dictionary containing the adsorption energy for each system ID.
    """
    with open(ref_path, "rb") as file:
        ad_energy = pickle.load(file)

    E_adsorption = {}
    for item in tqdm(data):
        random_sid = "random" + str(item.sid)
        refE = ad_energy[random_sid]
        Eads = item.y - refE
        E_adsorption[item.sid] = Eads

    return E_adsorption
