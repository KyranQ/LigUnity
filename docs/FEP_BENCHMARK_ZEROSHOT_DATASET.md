# FEP Benchmark Zero-Shot Dataset Documentation

This document explains where the FEP benchmark zero-shot dataset is called in LigUnity's code, how the data is loaded, and the format of the zero-shot dataset.

## 1. Where is the FEP benchmark zero-shot dataset called?

The FEP benchmark zero-shot dataset is called through the following code flow:

### Entry Point: `test.sh`

When you run:
```bash
bash test.sh FEP pocket_ranking ${path2weight} ${path2result}
```

The script (`test.sh`) invokes:
```bash
python ./unimol/test.py "./test_datasets" --user-dir ./unimol --valid-subset test \
       --results-path $results_path \
       --task test_task --loss rank_softmax --arch $arch \
       --path $weight_path \
       --test-task FEP
```

### Main Test Script: `unimol/test.py`

In `unimol/test.py` (lines 70-71), the FEP test task is handled:
```python
elif args.test_task == "FEP":
    task.test_fep(model)
```

### Task Implementation: `unimol/tasks/test_task.py`

The actual dataset loading and testing happens in `unimol/tasks/test_task.py`:

1. **`test_fep()` method** (lines 1203-1217): This is the main entry point for FEP zero-shot testing
   ```python
   def test_fep(self, model, **kwargs):
       labels_fep = json.load(
           open(f"{self.args.data}/FEP/fep_labels.json"))
       ligands_dict = {x["pockets"][0]: x for x in labels_fep}
       rho_list = []
       for i, target in enumerate(ligands_dict.keys()):
           if self.args.arch in ["DTA", "pocketregression"]:
               rho = self.test_fep_target_regression(target, model, ligands_dict[target])
           else:
               rho = self.test_fep_target(target, model, ligands_dict[target])
           rho_list.append(rho)
   ```

2. **`test_fep_target()` method** (lines 1088-1143): Loads ligand and pocket LMDB datasets
   ```python
   def test_fep_target(self, target, model, label_info, **kwargs):
       # Load ligand data
       data_path = f"{self.args.data}/FEP/lmdbs/{target}_lig.lmdb"
       mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
       
       # Load pocket data
       data_path = f"{self.args.data}/FEP/lmdbs/{target}.lmdb"
       pocket_dataset = self.load_pockets_dataset(data_path)
   ```

## 2. Which part of the code introduces/loads these datasets?

### Dataset Loading Methods

The datasets are loaded by these key methods in `unimol/tasks/test_task.py`:

1. **`load_mols_dataset()`** (lines 361-451): Loads ligand/molecule data from LMDB files
   - Reads atoms, coordinates, and SMILES from LMDB
   - Processes molecules by removing hydrogens and normalizing coordinates
   - Returns a nested dataset structure with:
     - `net_input`: molecular tokens, distances, edge types
     - `smi_name`: SMILES strings
     - `mol_len`: molecule lengths

2. **`load_pockets_dataset()`** (lines 453-537): Loads pocket data from LMDB files
   - Reads pocket atoms and coordinates from LMDB
   - Crops pockets to `max_pocket_atoms`
   - Returns a nested dataset structure with:
     - `net_input`: pocket tokens, distances, edge types, coordinates
     - `pocket_name`: pocket identifiers
     - `pocket_len`: pocket lengths

### Label Loading

Labels and sequence information are loaded from JSON files:
- **`test_datasets/FEP/fep_labels.json`**: Contains all FEP target information (sequences, ligands, activities)

## 3. FEP Zero-Shot Dataset Format

### Directory Structure

```
test_datasets/
├── FEP/
│   ├── fep_labels.json           # Labels and sequences for all targets
│   ├── FEP_sequence.csv          # Protein sequences
│   ├── ligands.lmdb              # All ligands in LMDB format
│   ├── proteins.lmdb             # All proteins in LMDB format
│   └── lmdbs/                    # Per-target LMDB files
│       ├── tnks2.lmdb            # Pocket data for TNKS2 target
│       ├── tnks2_lig.lmdb        # Ligand data for TNKS2 target
│       ├── p38.lmdb              # Pocket data for P38 target
│       ├── p38_lig.lmdb          # Ligand data for P38 target
│       └── ... (other targets)
└── FEP.json                      # FEP target metadata
```

### fep_labels.json Format

This is the main label file containing all FEP benchmark data:

```json
[
    {
        "pockets": ["tnks2"],           // Target pocket name(s)
        "uniprot": "Q9H2K2",            // UniProt ID
        "sequence": "MSGRRCAG...",      // Protein sequence
        "ligands": [
            {
                "act": 6.26874109866682,    // Activity value (pIC50 or similar)
                "smi": "O=c1[nH]c(-c2ccccc2)nc2ccccc12"  // SMILES string
            },
            // ... more ligands
        ]
    },
    // ... more targets
]
```

### FEP Targets in the Dataset

The FEP benchmark includes the following 16 targets:

| Target | UniProt ID | Description |
|--------|------------|-------------|
| tnks2  | Q9H2K2     | Tankyrase 2 |
| p38    | Q16539     | P38 MAP Kinase |
| hif2a  | Q99814     | HIF-2α |
| mcl1   | Q07820     | MCL1 (apoptosis regulator) |
| cdk8   | P49336     | Cyclin-Dependent Kinase 8 |
| cmet   | P08581     | c-Met receptor tyrosine kinase |
| tyk2   | P29597     | TYK2 kinase |
| pfkfb3 | Q16875     | PFKFB3 |
| cdk2   | P24941     | Cyclin-Dependent Kinase 2 |
| ptp1b  | P18031     | Protein Tyrosine Phosphatase 1B |
| jnk1   | P45983     | JNK1 kinase |
| shp2   | Q06124     | SHP2 phosphatase |
| bace   | P56817     | BACE1 (beta-secretase) |
| syk    | P43405     | SYK kinase |
| thrombin | P00734   | Thrombin |
| eg5    | P52732     | Kinesin-like protein EG5 |

### LMDB File Format

The LMDB files contain molecular and pocket structure data. The data is processed through specialized dataset classes (`AffinityMolDataset`, `AffinityPocketDataset`) that handle the data transformation.

#### Ligand LMDB (e.g., `tnks2_lig.lmdb`)
Each entry contains the following raw data:
- `atoms`: List of atom types (e.g., `["C", "N", "O", ...]`)
- `coordinates`: 3D coordinates as numpy array `(N_atoms, 3)`
- `smi`: SMILES string
- `label`: Activity value (optional, may not be present in all datasets)

The code loads this through `load_mols_dataset()` which:
1. Creates an `AffinityMolDataset` wrapper
2. Removes hydrogen atoms
3. Normalizes coordinates
4. Tokenizes atom types
5. Computes distance matrices and edge types

#### Pocket LMDB (e.g., `tnks2.lmdb`)
Each entry contains:
- `pocket_atoms`: List of residue-atom types (e.g., `["CA", "CB", "N", ...]`)
- `pocket_coordinates`: 3D coordinates as numpy array `(N_atoms, 3)`
- `pocket`: Pocket identifier
- `pocket_residue_name`: Residue names (optional)

The code loads this through `load_pockets_dataset()` which:
1. Creates an `AffinityPocketDataset` wrapper
2. Removes hydrogen atoms
3. Crops pocket to `max_pocket_atoms` (default 511)
4. Normalizes coordinates
5. Tokenizes residue-atom types
6. Computes distance matrices and edge types

### How to View the Dataset Format

#### Option 1: View JSON labels directly
```python
import json

# Load FEP labels
with open("test_datasets/FEP/fep_labels.json") as f:
    fep_labels = json.load(f)

# Print first target info
target = fep_labels[0]
print(f"Target: {target['pockets']}")
print(f"UniProt: {target['uniprot']}")
print(f"Sequence length: {len(target['sequence'])}")
print(f"Number of ligands: {len(target['ligands'])}")
print(f"First ligand: {target['ligands'][0]}")
```

#### Option 2: View LMDB data
```python
from unicore.data import LMDBDataset

# Load ligand dataset
lig_dataset = LMDBDataset("test_datasets/FEP/lmdbs/tnks2_lig.lmdb")
print(f"Number of ligands: {len(lig_dataset)}")
print(f"First ligand keys: {lig_dataset[0].keys()}")
print(f"First ligand SMILES: {lig_dataset[0]['smi']}")

# Load pocket dataset
pocket_dataset = LMDBDataset("test_datasets/FEP/lmdbs/tnks2.lmdb")
print(f"Number of pockets: {len(pocket_dataset)}")
print(f"First pocket keys: {pocket_dataset[0].keys()}")
```

## 4. Code Flow Summary

```
test.sh (FEP task)
    └── unimol/test.py::main()
            └── test_task.test_fep(model)
                    ├── Load labels: test_datasets/FEP/fep_labels.json
                    └── For each target:
                            ├── test_fep_target(target, model, label_info)
                            │       ├── load_mols_dataset(FEP/lmdbs/{target}_lig.lmdb)
                            │       ├── load_pockets_dataset(FEP/lmdbs/{target}.lmdb)
                            │       ├── Compute molecule embeddings
                            │       ├── Compute pocket embeddings with sequence
                            │       └── Calculate correlation (R²)
                            └── Report mean/median R² across all targets
```

## 5. Result Processing

After running zero-shot FEP tests, results are saved to:
- `{results_path}/FEP/{target}/saved_mols_embed.npy` - Molecule embeddings
- `{results_path}/FEP/{target}/saved_target_embed.npy` - Pocket embeddings
- `{results_path}/FEP/{target}/saved_labels.npy` - True activity values
- `{results_path}/FEP/{target}/saved_smis.json` - SMILES strings

The final ensemble results are computed using `ensemble_result.py`:
```bash
python ensemble_result.py zeroshot FEP
```

This combines predictions from pocket and protein ranking models to produce the final metrics.
