#!/usr/bin/env python3
"""
CASP16 数据处理脚本 - 将 CASP16 数据转换为 LigUnity LMDB 格式

使用方法:
    python py_scripts/process_casp16.py --casp16_dir test_datasets/CASP16 --output_dir test_datasets/CASP16/lmdbs

此脚本处理 CASP16 数据集并生成 LigUnity 所需的 LMDB 文件。
"""

import argparse
import json
import os
import pickle
import sys
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lmdb
except ImportError:
    print("请安装 lmdb: pip install lmdb")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("请安装 rdkit: pip install rdkit")
    sys.exit(1)

try:
    from biopandas.pdb import PandasPdb
except ImportError:
    print("请安装 biopandas: pip install biopandas")
    sys.exit(1)


def gen_conformation(mol, num_conf=1, num_worker=4):
    """为分子生成 3D 构象。"""
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None
    return mol


def read_pdb_protein(pdb_path: str) -> Dict:
    """读取 PDB 蛋白质文件。"""
    pdb_df = PandasPdb().read_pdb(pdb_path)
    
    coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    atom_type = pdb_df.df['ATOM']['atom_name'].tolist()
    residue_name = (pdb_df.df['ATOM']['chain_id'] + 
                   pdb_df.df['ATOM']['residue_number'].astype(str)).tolist()
    residue_type = pdb_df.df['ATOM']['residue_name'].tolist()
    
    return {
        'coord': np.array(coord),
        'atom_type': atom_type,
        'residue_name': residue_name,
        'residue_type': residue_type
    }


def read_pdb_ligand(pdb_path: str) -> np.ndarray:
    """读取 PDB 配体文件并返回坐标。"""
    pdb_df = PandasPdb().read_pdb(pdb_path)
    
    if len(pdb_df.df['HETATM']) > 0:
        coord = pdb_df.df['HETATM'][['x_coord', 'y_coord', 'z_coord']].values
    else:
        coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    
    return np.array(coord)


def extract_pocket(protein: Dict, ligand_coords: np.ndarray, raid: float = 6.0) -> Dict:
    """使用参考配体从蛋白质中提取口袋。"""
    protein_coord = protein['coord']
    protein_residue_name = protein['residue_name']
    
    pocket_residues = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coords)):
            if np.linalg.norm(protein_coord[i] - ligand_coords[j]) < raid:
                pocket_residues.add(protein_residue_name[i])
                break
    
    pocket_indices = [i for i, r in enumerate(protein_residue_name) if r in pocket_residues]
    
    return {
        'pocket_atoms': [protein['atom_type'][i] for i in pocket_indices],
        'pocket_coordinates': [protein['coord'][i] for i in pocket_indices],
        'pocket_residue_type': [protein['residue_type'][i] for i in pocket_indices],
        'pocket_residue_name': [protein['residue_name'][i] for i in pocket_indices],
    }


def smiles_to_mol_data(smi: str) -> Optional[Dict]:
    """将 SMILES 转换为分子数据。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    
    mol = gen_conformation(mol)
    if mol is None:
        return None
    
    coords = mol.GetConformer().GetPositions()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    
    return {
        'atoms': atoms,
        'coordinates': [np.array(coords)],
        'smi': smi,
        'mol': mol,
        'label': 1,
    }


def write_lmdb(data: List[Dict], lmdb_path: str):
    """将数据写入 LMDB 格式。"""
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, 
                    lock=False, map_size=1099511627776)
    
    with env.begin(write=True) as txn:
        for i, d in enumerate(data):
            txn.put(str(i).encode('ascii'), pickle.dumps(d))
    
    env.close()
    print(f"已保存 {len(data)} 条数据到 {lmdb_path}")


def process_casp16_target(target_id: str, casp16_dir: str, output_dir: str, 
                          affinity_df: pd.DataFrame, smiles_dict: Dict[str, str]):
    """处理单个 CASP16 目标。"""
    
    # 确定目标所属的系列 (L1000, L2000, L3000, L4000)
    target_num = int(target_id[1:])
    if target_num < 2000:
        series = "L1000"
    elif target_num < 3000:
        series = "L2000"
    elif target_num < 4000:
        series = "L3000"
    else:
        series = "L4000"
    
    # 查找蛋白质和配体结构
    struct_dir = os.path.join(casp16_dir, f"{series}_exper_struct", target_id, f"{series}_prepared", target_id)
    
    if not os.path.exists(struct_dir):
        print(f"警告: 找不到目标 {target_id} 的结构目录: {struct_dir}")
        return None
    
    # 查找蛋白质 PDB 文件
    protein_pdb = os.path.join(struct_dir, "protein_aligned.pdb")
    if not os.path.exists(protein_pdb):
        # 尝试其他文件名
        pdb_files = glob(os.path.join(struct_dir, "protein*.pdb"))
        if pdb_files:
            protein_pdb = pdb_files[0]
        else:
            print(f"警告: 找不到目标 {target_id} 的蛋白质 PDB 文件")
            return None
    
    # 查找配体 PDB 文件
    ligand_pdb_files = glob(os.path.join(struct_dir, "ligand*.pdb"))
    if not ligand_pdb_files:
        print(f"警告: 找不到目标 {target_id} 的配体 PDB 文件")
        return None
    
    ligand_pdb = ligand_pdb_files[0]
    
    print(f"处理目标 {target_id}:")
    print(f"  蛋白质: {protein_pdb}")
    print(f"  配体: {ligand_pdb}")
    
    # 读取蛋白质和配体
    protein = read_pdb_protein(protein_pdb)
    ligand_coords = read_pdb_ligand(ligand_pdb)
    
    # 提取口袋
    pocket_data = extract_pocket(protein, ligand_coords)
    pocket_data['pocket'] = target_id
    pocket_data['pocket_index'] = 1
    
    # 保存口袋 LMDB
    pocket_lmdb_path = os.path.join(output_dir, f"{target_id}.lmdb")
    write_lmdb([pocket_data], pocket_lmdb_path)
    
    # 获取配体 SMILES
    if target_id in smiles_dict:
        smi = smiles_dict[target_id]
        mol_data = smiles_to_mol_data(smi)
        if mol_data:
            mol_data_list = [mol_data]
            lig_lmdb_path = os.path.join(output_dir, f"{target_id}_lig.lmdb")
            write_lmdb(mol_data_list, lig_lmdb_path)
        else:
            print(f"  警告: 无法处理 SMILES: {smi}")
    else:
        print(f"  警告: 找不到目标 {target_id} 的 SMILES")
    
    # 获取活性值
    target_row = affinity_df[affinity_df['Target ID'] == target_id]
    if len(target_row) > 0:
        activity = target_row['binding_affinity'].values[0]
        smi = target_row['ligand_smiles'].values[0] if 'ligand_smiles' in target_row.columns else smiles_dict.get(target_id, "")
    else:
        activity = None
        smi = smiles_dict.get(target_id, "")
    
    return {
        'pockets': [target_id],
        'uniprot': target_id,  # CASP16 没有 UniProt ID，使用目标 ID 代替
        'sequence': "",  # 如果需要序列，可以从 PDB 提取
        'ligands': [{'act': activity, 'smi': smi}] if activity else [{'act': 0, 'smi': smi}]
    }


def load_casp16_smiles(casp16_dir: str) -> Dict[str, str]:
    """加载所有 CASP16 SMILES。
    
    支持两种目录结构:
    1. L3000_SMILES/L3001.tsv (直接在目录下)
    2. L3000_SMILES/L3000/L3001.tsv (在子目录下)
    """
    smiles_dict = {}
    
    for series in ["L1000", "L2000", "L3000", "L4000"]:
        smiles_dir = os.path.join(casp16_dir, f"{series}_SMILES")
        if not os.path.exists(smiles_dir):
            continue
        
        # 直接在目录下查找 TSV 文件
        for tsv_file in glob(os.path.join(smiles_dir, "*.tsv")):
            target_id = os.path.basename(tsv_file).replace(".tsv", "")
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                if 'SMILES' in df.columns:
                    smiles_dict[target_id] = df['SMILES'].values[0]
            except Exception as e:
                print(f"警告: 无法读取 {tsv_file}: {e}")
        
        # 在子目录下查找 TSV 文件 (例如 L3000_SMILES/L3000/L3001.tsv)
        for tsv_file in glob(os.path.join(smiles_dir, "*", "*.tsv")):
            target_id = os.path.basename(tsv_file).replace(".tsv", "")
            if target_id not in smiles_dict:  # 避免重复
                try:
                    df = pd.read_csv(tsv_file, sep='\t')
                    if 'SMILES' in df.columns:
                        smiles_dict[target_id] = df['SMILES'].values[0]
                except Exception as e:
                    print(f"警告: 无法读取 {tsv_file}: {e}")
    
    return smiles_dict


def load_casp16_affinity(casp16_dir: str) -> pd.DataFrame:
    """加载所有 CASP16 活性数据。"""
    affinity_dfs = []
    
    for csv_file in glob(os.path.join(casp16_dir, "*_exper_affinity.csv")):
        try:
            df = pd.read_csv(csv_file)
            affinity_dfs.append(df)
        except Exception as e:
            print(f"警告: 无法读取 {csv_file}: {e}")
    
    if affinity_dfs:
        return pd.concat(affinity_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def create_casp16_labels_json(labels: List[Dict], output_path: str):
    """创建 CASP16 标签 JSON 文件。"""
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"已保存标签到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description='处理 CASP16 数据集用于 LigUnity')
    parser.add_argument('--casp16_dir', type=str, default='test_datasets/CASP16',
                        help='CASP16 数据目录')
    parser.add_argument('--output_dir', type=str, default='test_datasets/CASP16/lmdbs',
                        help='输出 LMDB 目录')
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='要处理的目标列表 (例如: L1001 L1002)，不指定则处理所有')
    parser.add_argument('--series', type=str, nargs='+', default=['L1000', 'L2000', 'L3000', 'L4000'],
                        help='要处理的系列 (L1000, L2000, L3000, L4000)')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 SMILES 和活性数据
    print("加载 SMILES 数据...")
    smiles_dict = load_casp16_smiles(args.casp16_dir)
    print(f"已加载 {len(smiles_dict)} 个 SMILES")
    
    print("加载活性数据...")
    affinity_df = load_casp16_affinity(args.casp16_dir)
    print(f"已加载 {len(affinity_df)} 条活性数据")
    
    # 确定要处理的目标
    if args.targets:
        targets = args.targets
    else:
        # 从目录结构中获取所有目标
        targets = []
        for series in args.series:
            struct_dir = os.path.join(args.casp16_dir, f"{series}_exper_struct")
            if os.path.exists(struct_dir):
                targets.extend([d for d in os.listdir(struct_dir) if d.startswith('L')])
    
    targets = sorted(set(targets))
    print(f"将处理 {len(targets)} 个目标")
    
    # 处理每个目标
    labels = []
    for target_id in targets:
        label = process_casp16_target(
            target_id, args.casp16_dir, args.output_dir, 
            affinity_df, smiles_dict
        )
        if label:
            labels.append(label)
    
    # 保存标签 JSON
    if labels:
        labels_path = os.path.join(args.output_dir, 'casp16_labels.json')
        create_casp16_labels_json(labels, labels_path)
    
    print(f"\n处理完成！共处理 {len(labels)} 个目标")
    print(f"LMDB 文件保存在: {args.output_dir}")
    print(f"\n运行 LigUnity zero-shot 推理:")
    print(f"  bash test.sh CASP16 pocket_ranking ${{weight_path}} ./results/casp16")


if __name__ == "__main__":
    main()
