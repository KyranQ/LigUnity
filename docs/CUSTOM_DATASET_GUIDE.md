# 指南：在自定义数据集上测试 LigUnity（以 CASP16 为例）

本指南说明如何在自定义数据集上测试 LigUnity 的亲和力预测能力，以 CASP16 数据集为例。

---

## 🚀 专门针对 L1000 和 L3000 系列的详细步骤指南

如果您只需要使用 CASP16 的 **L1000** 和 **L3000** 系列进行亲和力预测，请按以下步骤操作：

### 前提条件

确保您已安装以下依赖：
```bash
pip install lmdb rdkit biopandas pandas numpy scipy
```

### 第 1 步：确认数据已上传

确保您的 CASP16 数据位于 `test_datasets/CASP16/` 目录下，结构如下：
```
test_datasets/CASP16/
├── L1000_exper_affinity.csv      # L1000 活性数据（包含 binding_affinity 列）
├── L3000_exper_affinity.csv      # L3000 活性数据
├── L1000_SMILES/                 # L1000 系列的 SMILES 文件
│   ├── L1001.tsv
│   ├── L1002.tsv
│   └── ...
├── L3000_SMILES/                 # L3000 系列的 SMILES 文件
│   ├── L3001.tsv
│   └── ...
├── L1000_exper_struct/           # L1000 系列的结构文件
│   ├── L1001/
│   │   └── L1000_prepared/
│   │       └── L1001/
│   │           ├── protein_aligned.pdb
│   │           └── ligand_*.pdb
│   └── ...
└── L3000_exper_struct/           # L3000 系列的结构文件
    └── ...
```

### 第 2 步：运行数据处理脚本（只处理 L1000 和 L3000）

在 LigUnity 根目录下运行：

```bash
# 只处理 L1000 和 L3000 系列
python py_scripts/process_casp16.py \
    --casp16_dir test_datasets/CASP16 \
    --output_dir test_datasets/CASP16/lmdbs \
    --series L1000 L3000
```

**此命令会：**
1. 读取 `L1000_exper_struct/` 和 `L3000_exper_struct/` 中的蛋白质和配体 PDB 文件
2. 读取 `L1000_SMILES/` 和 `L3000_SMILES/` 中的 SMILES 文件
3. 读取 `L1000_exper_affinity.csv` 和 `L3000_exper_affinity.csv` 中的活性数据
4. 在 `test_datasets/CASP16/lmdbs/` 目录下生成：
   - 每个目标的口袋 LMDB 文件（例如 `L1001.lmdb`）
   - 每个目标的配体 LMDB 文件（例如 `L1001_lig.lmdb`）
   - 标签文件 `casp16_labels.json`

**验证生成的文件：**
```bash
ls -la test_datasets/CASP16/lmdbs/

# 应该看到类似：
# L1001.lmdb
# L1001_lig.lmdb
# L1002.lmdb
# L1002_lig.lmdb
# ...
# L3001.lmdb
# L3001_lig.lmdb
# ...
# casp16_labels.json
```

### 第 3 步：下载模型权重

从 HuggingFace 下载 LigUnity 模型权重：
```bash
# 方法 1：使用 wget（推荐）
wget https://huggingface.co/fengb/LigUnity_pocket_ranking/resolve/main/checkpoint.pt -O ./checkpoint_pocket.pt

# 方法 2：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download fengb/LigUnity_pocket_ranking checkpoint.pt --local-dir ./
# 下载后文件位于 ./checkpoint.pt，需要重命名
mv ./checkpoint.pt ./checkpoint_pocket.pt
```

### 第 4 步：运行 LigUnity Zero-Shot 推理

```bash
# 设置模型权重路径（确保路径与上一步下载的位置一致）
weight_path="./checkpoint_pocket.pt"

# 运行 CASP16 测试
bash test.sh CASP16 pocket_ranking ${weight_path} ./results/casp16
```

**预期输出：**
```
处理目标 L1001...
  L1001: R² = 0.xxxx
处理目标 L1002...
  L1002: R² = 0.xxxx
...
CASP16 结果: Mean R² = 0.xxxx, Median R² = 0.xxxx
```

### 第 5 步：查看和分析结果

结果保存在 `./results/casp16/CASP16/{target}/` 目录下。

**使用 Python 分析结果：**

```python
import numpy as np
import json
from scipy import stats
import os

# 设置结果路径
results_base = "./results/casp16/CASP16"

# 遍历所有目标
targets = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]

all_results = []
for target in sorted(targets):
    target_dir = os.path.join(results_base, target)
    
    # 加载数据
    mol_embeds = np.load(os.path.join(target_dir, "saved_mols_embed.npy"))
    pocket_embeds = np.load(os.path.join(target_dir, "saved_target_embed.npy"))
    labels = np.load(os.path.join(target_dir, "saved_labels.npy"))
    with open(os.path.join(target_dir, "saved_smis.json")) as f:
        smiles = json.load(f)
    
    # 计算预测分数
    scores = pocket_embeds @ mol_embeds.T
    pred_scores = scores.max(axis=0)
    
    # 计算相关性
    pearson_r = stats.pearsonr(labels, pred_scores).statistic
    spearman_r = stats.spearmanr(labels, pred_scores).statistic
    r2 = max(pearson_r, 0) ** 2
    
    print(f"{target}: Pearson R = {pearson_r:.4f}, Spearman R = {spearman_r:.4f}, R² = {r2:.4f}")
    all_results.append({'target': target, 'pearson': pearson_r, 'spearman': spearman_r, 'r2': r2})

# 汇总统计
print(f"\n===== 汇总 =====")
print(f"目标总数: {len(all_results)}")
print(f"平均 Pearson R: {np.mean([r['pearson'] for r in all_results]):.4f}")
print(f"平均 Spearman R: {np.mean([r['spearman'] for r in all_results]):.4f}")
print(f"平均 R²: {np.mean([r['r2'] for r in all_results]):.4f}")
```

### 第 6 步（可选）：使用集成模型提高性能

为获得最佳结果，可以同时使用 pocket_ranking 和 protein_ranking 模型：

```bash
# 下载 protein_ranking 模型
wget https://huggingface.co/fengb/LigUnity_protein_ranking/resolve/main/checkpoint.pt -O checkpoint_protein.pt

# 运行 protein_ranking 模型
bash test.sh CASP16 protein_ranking ./checkpoint_protein.pt ./results/casp16_protein

# 集成两个模型的结果
python ensemble_result.py zeroshot CASP16
```

---

## 快速开始：处理已上传的 CASP16 数据集

如果您已经将 CASP16 数据上传到 `test_datasets/CASP16` 目录，请按以下步骤操作：

### 步骤 1：运行数据处理脚本

```bash
# 处理所有 CASP16 目标
python py_scripts/process_casp16.py --casp16_dir test_datasets/CASP16 --output_dir test_datasets/CASP16/lmdbs

# 或者只处理特定系列
python py_scripts/process_casp16.py --casp16_dir test_datasets/CASP16 --output_dir test_datasets/CASP16/lmdbs --series L1000 L3000
```

这将：
- 读取 `L1000_exper_struct/`, `L2000_exper_struct/` 等目录中的蛋白质和配体 PDB 文件
- 读取 `L1000_SMILES/`, `L2000_SMILES/` 等目录中的 SMILES 文件
- 读取 `L1000_exper_affinity.csv`, `L3000_exper_affinity.csv` 中的活性数据
- 生成 LMDB 格式的口袋和配体文件
- 创建 `casp16_labels.json` 标签文件

### 步骤 2：运行 LigUnity Zero-Shot 推理

```bash
# 设置模型权重路径
weight_path="path/to/checkpoint.pt"  # 从 HuggingFace 下载

# 运行 CASP16 测试
bash test.sh CASP16 pocket_ranking ${weight_path} ./results/casp16
```

### 步骤 3：查看结果

结果保存在 `./results/casp16/CASP16/{target}/` 目录下：
- `saved_mols_embed.npy`: 分子嵌入
- `saved_target_embed.npy`: 口袋嵌入
- `saved_labels.npy`: 真实活性值
- `saved_smis.json`: SMILES 列表

---

## 详细说明

## 概述

要在新数据集（如 CASP16）上测试 LigUnity，您需要：
1. 下载并准备数据（蛋白质和配体）
2. 将数据转换为 LMDB 格式
3. 使用 LigUnity 运行推理
4. 分析结果

## 第一步：下载 CASP16 数据

从以下链接下载 CASP16 药物配体数据集：
https://predictioncenter.org/download_area/CASP16/extra_experiments/pharma_ligands/

CASP16 数据集包括：
- 蛋白质结构（PDB 格式）
- 配体结构（PDB 格式）
- SMILES 文件（TSV 格式）
- 活性数据（CSV 格式）

## CASP16 数据集结构

```
test_datasets/CASP16/
├── L1000_exper_affinity.csv         # L1000 系列活性数据
├── L3000_exper_affinity.csv         # L3000 系列活性数据
├── L1000_SMILES/                    # L1000 系列 SMILES
│   ├── L1001.tsv
│   ├── L1002.tsv
│   └── ...
├── L1000_exper_struct/              # L1000 系列结构
│   ├── L1001/
│   │   └── L1000_prepared/
│   │       └── L1001/
│   │           ├── protein_aligned.pdb  # 蛋白质结构
│   │           └── ligand_*.pdb         # 配体结构
│   ├── L1002/
│   └── ...
├── L2000_SMILES/
├── L2000_exper_struct/
├── L3000_SMILES/
├── L3000_exper_struct/
├── L4000_SMILES/
└── L4000_exper_struct/
```

## 第二步：准备数据

### 所需数据格式

LigUnity 需要两种类型的输入：
1. **配体数据**：SMILES 字符串或 3D 结构
2. **口袋数据**：蛋白质口袋结构（PDB 格式），需要参考配体来定义结合位点

### 目录结构
```
your_dataset/
├── proteins/
│   └── target1.pdb          # 蛋白质结构
├── ligands/
│   └── target1_crystal.mol2 # 参考配体（用于定义口袋）
├── test_ligands/
│   └── target1_ligands.sdf  # 待测试的配体（或包含 SMILES 的 JSON 文件）
└── labels/
    └── target1_labels.json  # 真实活性值（可选）
```

## 第三步：转换数据为 LMDB 格式

使用 `py_scripts/write_case_study.py` 脚本转换数据：

### 将配体转换为 LMDB
```python
# 方法1：从 SMILES 列表（JSON 文件）
python py_scripts/write_case_study.py mol ligands.json output_ligands.lmdb

# ligands.json 格式：
# ["CC(=O)Nc1ccc...", "COc1ccc...", ...]
```

### 将口袋转换为 LMDB
```python
# 从 PDB + 参考配体（MOL2）
python py_scripts/write_case_study.py pocket protein.pdb crystal_ligand.mol2 output_pocket.lmdb
```

### CASP16 自定义处理脚本

以下是处理 CASP16 数据的完整 Python 脚本：

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
    """为分子生成 3D 构象。"""
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
    """将 SDF 文件转换为 LigUnity 使用的 LMDB 格式。"""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    data = []
    
    for mol in suppl:
        if mol is None:
            continue
        mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol)
        
        # 获取 3D 坐标
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
            'label': 1,  # 占位符
        })
    
    write_lmdb(data, output_lmdb_path)
    print(f"已处理 {len(data)} 个配体到 {output_lmdb_path}")

def process_pocket_pdb(pdb_path, ligand_path, output_lmdb_path, pocket_name="demo", raid=6.0):
    """使用参考配体从 PDB 中提取口袋并保存为 LMDB。"""
    
    # 读取蛋白质
    pdb_df = PandasPdb().read_pdb(pdb_path)
    protein_coords = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].values
    protein_atoms = pdb_df.df['ATOM']['atom_name'].tolist()
    protein_residues = (pdb_df.df['ATOM']['chain_id'] + 
                        pdb_df.df['ATOM']['residue_number'].astype(str)).tolist()
    protein_residue_types = pdb_df.df['ATOM']['residue_name'].tolist()
    
    # 读取参考配体以定义口袋
    if ligand_path.endswith('.mol2'):
        mol2_df = PandasMol2().read_mol2(ligand_path)
        ligand_coords = mol2_df.df[['x', 'y', 'z']].values
    elif ligand_path.endswith('.sdf'):
        mol = next(Chem.SDMolSupplier(ligand_path))
        ligand_coords = mol.GetConformer().GetPositions()
    else:
        raise ValueError("配体必须是 .mol2 或 .sdf 格式")
    
    # 查找配体半径范围内的口袋残基
    pocket_residues = set()
    for p_coord, res_name in zip(protein_coords, protein_residues):
        for l_coord in ligand_coords:
            if np.linalg.norm(p_coord - l_coord) < raid:
                pocket_residues.add(res_name)
                break
    
    # 提取口袋原子
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
    print(f"已提取包含 {len(pocket_indices)} 个原子的口袋到 {output_lmdb_path}")

def write_lmdb(data, lmdb_path):
    """将数据写入 LMDB 格式。"""
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, 
                    lock=False, map_size=1099511627776)
    
    with env.begin(write=True) as txn:
        for i, d in enumerate(data):
            txn.put(str(i).encode('ascii'), pickle.dumps(d))
    
    env.close()

# CASP16 使用示例：
if __name__ == "__main__":
    # 处理一个 CASP16 目标
    target_name = "T0001"  # 替换为实际目标名称
    
    # 1. 转换配体
    process_ligand_sdf(
        f"casp16_data/{target_name}/ligands.sdf",
        f"processed/{target_name}_lig.lmdb"
    )
    
    # 2. 提取口袋
    process_pocket_pdb(
        f"casp16_data/{target_name}/protein.pdb",
        f"casp16_data/{target_name}/crystal_ligand.mol2",
        f"processed/{target_name}.lmdb",
        pocket_name=target_name
    )
```

## 第四步：运行 LigUnity 推理

### 方式 A：使用演示模式（单个目标）

用于测试单个蛋白质-配体对：

```bash
# 设置路径
lig_file="path/to/target_lig.lmdb"
prot_file="path/to/target.lmdb"
uniprot="P00000"  # UniProt ID（如未知可使用占位符）
arch="pocket_ranking"  # 或 "protein_ranking"
weight_path="path/to/checkpoint.pt"
results_path="./results/casp16_demo"

# 运行推理
bash test_zeroshot_demo.sh $lig_file $prot_file $uniprot $arch $weight_path $results_path
```

### 方式 B：将 CASP16 添加为新的测试任务

用于系统性测试，将 CASP16 添加到测试流程：

1. **创建数据目录结构：**
```
test_datasets/
└── CASP16/
    ├── casp16_labels.json    # 标签文件
    └── lmdbs/
        ├── target1.lmdb      # 口袋数据
        ├── target1_lig.lmdb  # 配体数据
        ├── target2.lmdb
        └── target2_lig.lmdb
```

2. **创建标签文件（`casp16_labels.json`）：**
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

3. **在 `unimol/tasks/test_task.py` 中添加测试方法：**
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
    
    print(f"CASP16 平均 R²: {np.mean(rho_list)}")

def test_casp16_target(self, target, model, label_info, **kwargs):
    # 类似于 test_fep_target()
    data_path = f"{self.args.data}/CASP16/lmdbs/{target}_lig.lmdb"
    mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
    # ... 其余推理代码
```

4. **在 `unimol/test.py` 中添加：**
```python
elif args.test_task == "CASP16":
    task.test_casp16(model)
```

5. **运行测试：**
```bash
bash test.sh CASP16 pocket_ranking ${weight_path} ./results/casp16
```

## 第五步：分析结果

推理完成后，结果保存为 numpy 文件：

```python
import numpy as np
import json

# 加载结果
mol_embeds = np.load("results/saved_mols_embed.npy")
pocket_embeds = np.load("results/saved_target_embed.npy")
smis = json.load(open("results/saved_smis.json"))

# 计算亲和力分数（余弦相似度）
scores = pocket_embeds @ mol_embeds.T
affinity_scores = scores.max(axis=0)  # 取口袋构象的最大值

# 按预测亲和力对配体排序
ranked_indices = np.argsort(affinity_scores)[::-1]
for i in ranked_indices[:10]:
    print(f"排名 {i+1}: {smis[i]} (分数: {affinity_scores[i]:.4f})")
```

### 计算评估指标

如果有真实活性值：
```python
from scipy import stats

# 加载真实值
true_activities = [label["act"] for label in labels_data["ligands"]]

# 计算相关性
pearson_r = stats.pearsonr(true_activities, affinity_scores).statistic
spearman_r = stats.spearmanr(true_activities, affinity_scores).statistic

print(f"Pearson R: {pearson_r:.4f}")
print(f"Spearman R: {spearman_r:.4f}")
print(f"R²: {max(pearson_r, 0)**2:.4f}")
```

## CASP16 使用技巧

1. **获取 UniProt ID**：使用 UniProt 的映射服务查找 CASP16 目标的 UniProt ID
2. **口袋定义**：如果没有提供参考配体，使用结合位点预测工具
3. **序列获取**：可以使用 `get_uniprot_seq()` 函数自动获取序列
4. **集成预测**：为获得最佳结果，同时使用 `pocket_ranking` 和 `protein_ranking` 模型并集成结果

## 故障排除

### 常见问题

1. **"口袋太大"**：减小 `raid` 参数（默认：6Å）或增加 `--max-pocket-atoms`
2. **"无法生成构象"**：某些 SMILES 可能失败；过滤掉有问题的分子
3. **"找不到 UniProt 序列"**：在标签文件中手动提供序列

### 内存问题

对于大型数据集，分批处理：
```bash
# 分割数据并多次运行
for subset in subset1 subset2 subset3; do
    bash test_zeroshot_demo.sh ${subset}_lig.lmdb pocket.lmdb uniprot arch weight ./results/${subset}
done
```

## 参考资料

- CASP16 药物配体：https://predictioncenter.org/download_area/CASP16/extra_experiments/pharma_ligands/
- LigUnity 论文：https://doi.org/10.1016/j.patter.2025.101371
- LigUnity 模型权重：https://huggingface.co/fengb/LigUnity_pocket_ranking
