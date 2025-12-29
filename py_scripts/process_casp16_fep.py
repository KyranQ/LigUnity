#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import pickle



TWO_LETTER_ELEMENTS = {
    "Cl",
    "Br",
    "Na",
    "Ca",
    "Si",
    "Al",
    "Li",
    "Mg",
    "Zn",
    "Fe",
    "Hg",
    "Sn",
    "As",
    "Se",
    "Ag",
    "Au",
    "Pt",
    "Pb",
    "Mn",
    "Co",
    "Ni",
    "Cu",
    "Cr",
    "Ga",
    "Ge",
    "Sr",
    "Ba",
    "Cd",
    "Bi",
}


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def normalize_element(element):
    if not element:
        return ""
    element = element.strip()
    if len(element) == 1:
        return element.upper()
    return element[0].upper() + element[1].lower()


def derive_element(atom_name):
    atom_name = atom_name.strip()
    letters = "".join([c for c in atom_name if c.isalpha()])
    if not letters:
        return ""
    if len(letters) >= 2:
        candidate = letters[:2].capitalize()
        if candidate in TWO_LETTER_ELEMENTS:
            return candidate
    return letters[0].upper()


def parse_pdb_atoms(path, record_types=("ATOM", "HETATM")):
    atoms = []
    with open(path, "r") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record not in record_types:
                continue
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21].strip()
            residue_number = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = normalize_element(line[76:78].strip())
            if not element:
                element = derive_element(atom_name)
            atoms.append(
                {
                    "atom_name": atom_name,
                    "element": element,
                    "residue_name": residue_name,
                    "chain_id": chain_id,
                    "residue_number": residue_number,
                    "coord": (x, y, z),
                }
            )
    return atoms


def extract_sequence(path):
    residues = []
    seen = set()
    for atom in parse_pdb_atoms(path, record_types=("ATOM",)):
        key = (atom["chain_id"], atom["residue_number"])
        if key in seen:
            continue
        seen.add(key)
        residues.append(atom["residue_name"])
    return "".join(AA3_TO_AA1.get(res, "X") for res in residues)


def build_pocket(protein_atoms, ligand_atoms, dist_threshold):
    pocket_residue_keys = set()
    threshold_sq = dist_threshold * dist_threshold
    ligand_coords = [atom["coord"] for atom in ligand_atoms]
    for atom in protein_atoms:
        px, py, pz = atom["coord"]
        for lx, ly, lz in ligand_coords:
            dx = px - lx
            dy = py - ly
            dz = pz - lz
            if dx * dx + dy * dy + dz * dz <= threshold_sq:
                pocket_residue_keys.add(
                    (atom["chain_id"], atom["residue_number"])
                )
                break
    pocket_atoms = []
    pocket_coords = []
    pocket_residue_type = []
    pocket_residue_name = []
    for atom in protein_atoms:
        key = (atom["chain_id"], atom["residue_number"])
        if key not in pocket_residue_keys:
            continue
        pocket_atoms.append(atom["atom_name"])
        pocket_coords.append(list(atom["coord"]))
        pocket_residue_type.append(atom["residue_name"])
        chain = atom["chain_id"]
        resnum = atom["residue_number"]
        pocket_residue_name.append(f"{chain}{resnum}" if chain else resnum)
    return {
        "pocket_atoms": pocket_atoms,
        "pocket_coordinates": pocket_coords,
        "pocket_residue_type": pocket_residue_type,
        "pocket_residue_name": pocket_residue_name,
    }


def write_lmdb(records, lmdb_path):
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lmdb is required to write LMDB files. Install it or run with --skip-lmdb."
        ) from exc
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=1_073_741_824,
    )
    with env.begin(write=True) as txn:
        for idx, record in enumerate(records):
            txn.put(str(idx).encode("ascii"), pickle.dumps(record))


def find_structure_paths(structure_root, target, prepared_dir):
    target_dir = os.path.join(structure_root, target, prepared_dir, target)
    protein_path = os.path.join(target_dir, "protein_aligned.pdb")
    if not os.path.exists(protein_path):
        raise FileNotFoundError(f"No protein pdb found in {target_dir}")
    ligand_candidates = glob.glob(os.path.join(target_dir, "ligand*.pdb"))
    if not ligand_candidates:
        raise FileNotFoundError(f"No ligand pdb found in {target_dir}")
    ligand_path = ligand_candidates[0]
    return protein_path, ligand_path


def load_affinity_rows(csv_path, allowed_tasks):
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Task"] not in allowed_tasks:
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="test_datasets/CASP16")
    parser.add_argument("--dist-threshold", type=float, default=6.0)
    parser.add_argument("--output-labels", default="casp16_fep_labels.json")
    parser.add_argument("--skip-lmdb", action="store_true", help="only write labels json")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    lmdb_root = os.path.join(dataset_root, "lmdbs")
    os.makedirs(lmdb_root, exist_ok=True)

    label_entries = {}

    csv_configs = [
        {
            "csv_path": os.path.join(dataset_root, "L1000_exper_affinity.csv"),
            "structure_root": os.path.join(dataset_root, "L1000_exper_struct"),
            "prepared_dir": "L1000_prepared",
            "allowed_tasks": {"PA"},
        },
        {
            "csv_path": os.path.join(dataset_root, "L3000_exper_affinity.csv"),
            "structure_root": os.path.join(dataset_root, "L3000_exper_struct"),
            "prepared_dir": "L3000_prepared",
            "allowed_tasks": {"PA", "A"},
        },
    ]

    missing_targets = []

    for cfg in csv_configs:
        rows = load_affinity_rows(cfg["csv_path"], cfg["allowed_tasks"])
        for row in rows:
            target = row["Target ID"]
            smiles = row["ligand_smiles"] if "ligand_smiles" in row else row["Structure"]
            affinity = float(row["binding_affinity"])

            try:
                protein_path, ligand_path = find_structure_paths(
                    cfg["structure_root"], target, cfg["prepared_dir"]
                )
            except FileNotFoundError:
                missing_targets.append(target)
                continue

            protein_atoms = parse_pdb_atoms(protein_path, record_types=("ATOM",))
            ligand_atoms = parse_pdb_atoms(ligand_path, record_types=("ATOM", "HETATM"))

            pocket_data = build_pocket(protein_atoms, ligand_atoms, args.dist_threshold)
            pocket_entry = {
                "pocket": target,
                "pocket_index": 1,
                **pocket_data,
            }
            if not args.skip_lmdb:
                write_lmdb([pocket_entry], os.path.join(lmdb_root, f"{target}.lmdb"))

            ligand_entry = {
                "atoms": [atom["element"] for atom in ligand_atoms],
                "coordinates": [list(atom["coord"]) for atom in ligand_atoms],
                "smi": smiles,
                "mol": None,
                "name": target,
            }
            ligand_entry = {
                **ligand_entry,
                "coordinates": [ligand_entry["coordinates"]],
            }
            if not args.skip_lmdb:
                write_lmdb([ligand_entry], os.path.join(lmdb_root, f"{target}_lig.lmdb"))

            if target not in label_entries:
                label_entries[target] = {
                    "pockets": [target],
                    "uniprot": target,
                    "sequence": extract_sequence(protein_path),
                    "ligands": [],
                }
            label_entries[target]["ligands"].append(
                {"act": affinity, "smi": smiles}
            )

    output_path = os.path.join(dataset_root, args.output_labels)
    with open(output_path, "w") as handle:
        json.dump(list(label_entries.values()), handle, indent=2)

    if missing_targets:
        missing_path = os.path.join(dataset_root, "casp16_missing_targets.json")
        with open(missing_path, "w") as handle:
            json.dump(sorted(set(missing_targets)), handle, indent=2)


if __name__ == "__main__":
    main()
