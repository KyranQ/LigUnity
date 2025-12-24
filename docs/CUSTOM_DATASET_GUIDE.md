# Guide: Testing LigUnity on Custom Datasets (e.g., CASP16)

This guide explains how to test LigUnity's affinity prediction capability on custom datasets, using CASP16 as an example.

## Overview

To test LigUnity on a new dataset like CASP16, you need to:
1. Download and prepare your data (proteins and ligands)
2. Convert the data to LMDB format
3. Run inference using LigUnity
4. Analyze the results

## Step 1: Download CASP16 Data

Download the CASP16 pharma ligands dataset from:
https://predictioncenter.org/download_area/CASP16/extra_experiments/pharma_ligands/

The CASP16 dataset typically includes:
- Protein structures (PDB format)
- Ligand structures (SDF/MOL2 format)
- Binding site information

## Step 2: Prepare Your Data

### Required Data Format

LigUnity requires two types of input:
1. **Ligand data**: SMILES strings or 3D structures
2. **Pocket data**: Protein pocket structures (PDB format) with a reference ligand to define the binding site

### Directory Structure
```
your_dataset/
├── proteins/
│   └── target1.pdb          # Protein structure
├── ligands/
│   └── target1_crystal.mol2 # Reference ligand to define pocket
├── test_ligands/
│   └── target1_ligands.sdf  # Ligands to test (or JSON with SMILES)
└── labels/
    └── target1_labels.json  # Ground truth activities (optional)
```

## Step 3: Convert Data to LMDB Format

Use the `py_scripts/write_case_study.py` script to convert your data:

### Convert Ligands to LMDB
```python
# Method 1: From SMILES list (JSON file)
python py_scripts/write_case_study.py mol ligands.json output_ligands.lmdb

# ligands.json format:
# ["CC(=O)Nc1ccc...", "COc1ccc...", ...]
```

### Convert Pocket to LMDB
```python
# From PDB + reference ligand (MOL2)
python py_scripts/write_case_study.py pocket protein.pdb crystal_ligand.mol2 output_pocket.lmdb
```

### Custom Script for CASP16

Here's a complete Python script to process CASP16 data:

```python
import json
import os
import pickle
import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2

def gen_conformation(mol, num_conf=1, num_worker=4):
    """Generate 3D conformation for a molecule."""
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        mol = Chem.RemoveHs(mol)
    except:
        return None
    if mol.GetNumConformers() == 0:
        return None
    return mol

def process_ligand_sdf(sdf_path, output_lmdb_path):
    """Convert SDF file to LMDB format for LigUnity."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    data = []
    
    for mol in suppl:
        if mol is None:
            continue
        mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol)
        
        # Get 3D coordinates
        if mol.GetNumConformers() > 0:
            coords = mol.GetConformer().GetPositions()
        else:
            mol = gen_conformation(Chem.MolFromSmiles(smi))
            if mol is None:
                continue
            coords = mol.GetConformer().GetPositions()
        
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        
        data.append({
            'atoms': atoms,
            'coordinates': [np.array(coords)],
            'smi': smi,
            'mol': mol,
            'label': 1,  # placeholder
        })
    
    write_lmdb(data, output_lmdb_path)
    print(f"Processed {len(data)} ligands to {output_lmdb_path}")

def process_pocket_pdb(pdb_path, ligand_path, output_lmdb_path, pocket_name="demo", raid=6.0):
    """Extract pocket from PDB using reference ligand and save to LMDB."""
    
    # Read protein
    pdb_df = PandasPdb().read_pdb(pdb_path)
    protein_coords = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    protein_atoms = pdb_df.df['ATOM']['atom_name'].tolist()
    protein_residues = (pdb_df.df['ATOM']['chain_id'] + 
                        pdb_df.df['ATOM']['residue_number'].astype(str)).tolist()
    protein_residue_types = pdb_df.df['ATOM']['residue_name'].tolist()
    
    # Read reference ligand to define pocket
    if ligand_path.endswith('.mol2'):
        mol2_df = PandasMol2().read_mol2(ligand_path)
        ligand_coords = mol2_df.df[['x', 'y', 'z']].values
    elif ligand_path.endswith('.sdf'):
        mol = next(Chem.SDMolSupplier(ligand_path))
        ligand_coords = mol.GetConformer().GetPositions()
    else:
        raise ValueError("Ligand must be .mol2 or .sdf format")
    
    # Find pocket residues within radius of ligand
    pocket_residues = set()
    for p_coord, res_name in zip(protein_coords, protein_residues):
        for l_coord in ligand_coords:
            if np.linalg.norm(p_coord - l_coord) < raid:
                pocket_residues.add(res_name)
                break
    
    # Extract pocket atoms
    pocket_indices = [i for i, r in enumerate(protein_residues) if r in pocket_residues]
    pocket_data = {
        'pocket': pocket_name,
        'pocket_index': 1,
        'pocket_atoms': [protein_atoms[i] for i in pocket_indices],
        'pocket_coordinates': [protein_coords[i] for i in pocket_indices],
        'pocket_residue_type': [protein_residue_types[i] for i in pocket_indices],
        'pocket_residue_name': [protein_residues[i] for i in pocket_indices],
    }
    
    write_lmdb([pocket_data], output_lmdb_path)
    print(f"Extracted pocket with {len(pocket_indices)} atoms to {output_lmdb_path}")

def write_lmdb(data, lmdb_path):
    """Write data to LMDB format."""
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, 
                    lock=False, map_size=1099511627776)
    
    with env.begin(write=True) as txn:
        for i, d in enumerate(data):
            txn.put(str(i).encode('ascii'), pickle.dumps(d))
    
    env.close()

# Example usage for CASP16:
if __name__ == "__main__":
    # Process a CASP16 target
    target_name = "T0001"  # Replace with actual target name
    
    # 1. Convert ligands
    process_ligand_sdf(
        f"casp16_data/{target_name}/ligands.sdf",
        f"processed/{target_name}_lig.lmdb"
    )
    
    # 2. Extract pocket
    process_pocket_pdb(
        f"casp16_data/{target_name}/protein.pdb",
        f"casp16_data/{target_name}/crystal_ligand.mol2",
        f"processed/{target_name}.lmdb",
        pocket_name=target_name
    )
```

## Step 4: Run LigUnity Inference

### Option A: Using Demo Mode (Single Target)

For testing on a single protein-ligand pair:

```bash
# Set paths
lig_file="path/to/target_lig.lmdb"
prot_file="path/to/target.lmdb"
uniprot="P00000"  # UniProt ID (or placeholder if unknown)
arch="pocket_ranking"  # or "protein_ranking"
weight_path="path/to/checkpoint.pt"
results_path="./results/casp16_demo"

# Run inference
bash test_zeroshot_demo.sh $lig_file $prot_file $uniprot $arch $weight_path $results_path
```

### Option B: Adding CASP16 as a New Test Task

For systematic testing, add CASP16 to the test pipeline:

1. **Create data directory structure:**
```
test_datasets/
└── CASP16/
    ├── casp16_labels.json    # Labels file
    └── lmdbs/
        ├── target1.lmdb      # Pocket data
        ├── target1_lig.lmdb  # Ligand data
        ├── target2.lmdb
        └── target2_lig.lmdb
```

2. **Create labels file (`casp16_labels.json`):**
```json
[
    {
        "pockets": ["target1"],
        "uniprot": "P00001",
        "sequence": "MKTAYIAKQR...",
        "ligands": [
            {"act": 7.5, "smi": "CC(=O)Nc1ccc..."},
            {"act": 6.2, "smi": "COc1ccc..."}
        ]
    },
    ...
]
```

3. **Add test method to `unimol/tasks/test_task.py`:**
```python
def test_casp16(self, model, **kwargs):
    labels_casp16 = json.load(
        open(f"{self.args.data}/CASP16/casp16_labels.json"))
    ligands_dict = {x["pockets"][0]: x for x in labels_casp16}
    rho_list = []
    
    for target in ligands_dict.keys():
        if self.args.arch in ["DTA", "pocketregression"]:
            rho = self.test_casp16_target_regression(target, model, ligands_dict[target])
        else:
            rho = self.test_casp16_target(target, model, ligands_dict[target])
        rho_list.append(rho)
    
    print(f"CASP16 Mean R²: {np.mean(rho_list)}")

def test_casp16_target(self, target, model, label_info, **kwargs):
    # Similar to test_fep_target()
    data_path = f"{self.args.data}/CASP16/lmdbs/{target}_lig.lmdb"
    mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
    # ... rest of the inference code
```

4. **Add to `unimol/test.py`:**
```python
elif args.test_task == "CASP16":
    task.test_casp16(model)
```

5. **Run the test:**
```bash
bash test.sh CASP16 pocket_ranking ${weight_path} ./results/casp16
```

## Step 5: Analyze Results

After inference, results are saved as numpy files:

```python
import numpy as np
import json

# Load results
mol_embeds = np.load("results/saved_mols_embed.npy")
pocket_embeds = np.load("results/saved_target_embed.npy")
smis = json.load(open("results/saved_smis.json"))

# Compute affinity scores (cosine similarity)
scores = pocket_embeds @ mol_embeds.T
affinity_scores = scores.max(axis=0)  # Max over pocket conformations

# Rank ligands by predicted affinity
ranked_indices = np.argsort(affinity_scores)[::-1]
for i in ranked_indices[:10]:
    print(f"Rank {i+1}: {smis[i]} (score: {affinity_scores[i]:.4f})")
```

### Computing Metrics

If you have ground truth activities:
```python
from scipy import stats

# Load ground truth
true_activities = [label["act"] for label in labels_data["ligands"]]

# Compute correlation
pearson_r = stats.pearsonr(true_activities, affinity_scores).statistic
spearman_r = stats.spearmanr(true_activities, affinity_scores).statistic

print(f"Pearson R: {pearson_r:.4f}")
print(f"Spearman R: {spearman_r:.4f}")
print(f"R²: {max(pearson_r, 0)**2:.4f}")
```

## Tips for CASP16

1. **Get UniProt IDs**: Use UniProt's mapping service to find UniProt IDs for CASP16 targets
2. **Pocket definition**: If no reference ligand is provided, use binding site prediction tools
3. **Sequence retrieval**: Sequences can be fetched automatically using the `get_uniprot_seq()` function
4. **Ensemble predictions**: For best results, use both `pocket_ranking` and `protein_ranking` models and ensemble the results

## Troubleshooting

### Common Issues

1. **"Pocket too large"**: Reduce the `raid` parameter (default: 6Å) or increase `--max-pocket-atoms`
2. **"Cannot generate conformation"**: Some SMILES may fail; filter out problematic molecules
3. **"UniProt sequence not found"**: Manually provide the sequence in labels file

### Memory Issues

For large datasets, process in batches:
```bash
# Split your data and run multiple times
for subset in subset1 subset2 subset3; do
    bash test_zeroshot_demo.sh ${subset}_lig.lmdb pocket.lmdb uniprot arch weight ./results/${subset}
done
```

## References

- CASP16 Pharma Ligands: https://predictioncenter.org/download_area/CASP16/extra_experiments/pharma_ligands/
- LigUnity Paper: https://doi.org/10.1016/j.patter.2025.101371
- LigUnity Checkpoints: https://huggingface.co/fengb/LigUnity_pocket_ranking
