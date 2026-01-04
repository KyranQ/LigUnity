# FEP 基准 Zero-Shot 数据集文档

本文档说明 FEP 基准 zero-shot 数据集在 LigUnity 代码中的调用位置、数据加载方式以及数据集格式。

## 1. FEP 基准 zero-shot 数据集在哪里被调用？

FEP 基准 zero-shot 数据集通过以下代码流程被调用：

### 入口点：`test.sh`

运行以下命令时：
```bash
bash test.sh FEP pocket_ranking ${path2weight} ${path2result}
```

脚本 (`test.sh`) 会调用：
```bash
python ./unimol/test.py "./test_datasets" --user-dir ./unimol --valid-subset test \
       --results-path $results_path \
       --task test_task --loss rank_softmax --arch $arch \
       --path $weight_path \
       --test-task FEP
```

### 主测试脚本：`unimol/test.py`

在 `unimol/test.py` 中（第 70-71 行），FEP 测试任务的处理：
```python
elif args.test_task == "FEP":
    task.test_fep(model)
```

### 任务实现：`unimol/tasks/test_task.py`

实际的数据集加载和测试在 `unimol/tasks/test_task.py` 中进行：

1. **`test_fep()` 方法**（第 1203-1217 行）：FEP zero-shot 测试的主入口
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

2. **`test_fep_target()` 方法**（第 1088-1143 行）：加载配体和口袋 LMDB 数据集
   ```python
   def test_fep_target(self, target, model, label_info, **kwargs):
       # 加载配体数据
       data_path = f"{self.args.data}/FEP/lmdbs/{target}_lig.lmdb"
       mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
       
       # 加载口袋数据
       data_path = f"{self.args.data}/FEP/lmdbs/{target}.lmdb"
       pocket_dataset = self.load_pockets_dataset(data_path)
   ```

## 2. 哪部分代码引入/加载这些数据集？

### 数据集加载方法

数据集由 `unimol/tasks/test_task.py` 中的以下关键方法加载：

1. **`load_mols_dataset()`**（第 361-451 行）：从 LMDB 文件加载配体/分子数据
   - 从 LMDB 读取原子、坐标和 SMILES
   - 通过移除氢原子和归一化坐标来处理分子
   - 返回包含以下内容的嵌套数据集结构：
     - `net_input`：分子 token、距离、边类型
     - `smi_name`：SMILES 字符串
     - `mol_len`：分子长度

2. **`load_pockets_dataset()`**（第 453-537 行）：从 LMDB 文件加载口袋数据
   - 从 LMDB 读取口袋原子和坐标
   - 将口袋裁剪到 `max_pocket_atoms`
   - 返回包含以下内容的嵌套数据集结构：
     - `net_input`：口袋 token、距离、边类型、坐标
     - `pocket_name`：口袋标识符
     - `pocket_len`：口袋长度

### 标签加载

标签和序列信息从 JSON 文件加载：
- **`test_datasets/FEP/fep_labels.json`**：包含所有 FEP 目标信息（序列、配体、活性）

## 3. FEP Zero-Shot 数据集格式

### 目录结构

```
test_datasets/
├── FEP/
│   ├── fep_labels.json           # 所有目标的标签和序列
│   ├── FEP_sequence.csv          # 蛋白质序列
│   ├── ligands.lmdb              # LMDB 格式的所有配体
│   ├── proteins.lmdb             # LMDB 格式的所有蛋白质
│   └── lmdbs/                    # 按目标分类的 LMDB 文件
│       ├── tnks2.lmdb            # TNKS2 目标的口袋数据
│       ├── tnks2_lig.lmdb        # TNKS2 目标的配体数据
│       ├── p38.lmdb              # P38 目标的口袋数据
│       ├── p38_lig.lmdb          # P38 目标的配体数据
│       └── ...（其他目标）
└── FEP.json                      # FEP 目标元数据
```

### fep_labels.json 格式

这是包含所有 FEP 基准数据的主标签文件：

```json
[
    {
        "pockets": ["tnks2"],           // 目标口袋名称
        "uniprot": "Q9H2K2",            // UniProt ID
        "sequence": "MSGRRCAG...",      // 蛋白质序列
        "ligands": [
            {
                "act": 6.26874109866682,    // 活性值（pIC50 或类似值）
                "smi": "O=c1[nH]c(-c2ccccc2)nc2ccccc12"  // SMILES 字符串
            },
            // ... 更多配体
        ]
    },
    // ... 更多目标
]
```

### 数据集中的 FEP 目标

FEP 基准包含以下 16 个目标：

| 目标 | UniProt ID | 描述 |
|------|------------|------|
| tnks2  | Q9H2K2     | Tankyrase 2 |
| p38    | Q16539     | P38 MAP 激酶 |
| hif2a  | Q99814     | HIF-2α |
| mcl1   | Q07820     | MCL1（凋亡调节因子）|
| cdk8   | P49336     | 周期蛋白依赖性激酶 8 |
| cmet   | P08581     | c-Met 受体酪氨酸激酶 |
| tyk2   | P29597     | TYK2 激酶 |
| pfkfb3 | Q16875     | PFKFB3 |
| cdk2   | P24941     | 周期蛋白依赖性激酶 2 |
| ptp1b  | P18031     | 蛋白酪氨酸磷酸酶 1B |
| jnk1   | P45983     | JNK1 激酶 |
| shp2   | Q06124     | SHP2 磷酸酶 |
| bace   | P56817     | BACE1（β-分泌酶）|
| syk    | P43405     | SYK 激酶 |
| thrombin | P00734   | 凝血酶 |
| eg5    | P52732     | 驱动蛋白 EG5 |

### LMDB 文件格式

LMDB 文件包含分子和口袋结构数据。数据通过专门的数据集类（`AffinityMolDataset`、`AffinityPocketDataset`）进行处理和转换。

#### 配体 LMDB（例如 `tnks2_lig.lmdb`）
每个条目包含以下原始数据：
- `atoms`：原子类型列表（例如 `["C", "N", "O", ...]`）
- `coordinates`：3D 坐标，numpy 数组 `(N_atoms, 3)`
- `smi`：SMILES 字符串
- `label`：活性值（可选，可能不存在于所有数据集中）

代码通过 `load_mols_dataset()` 加载，该方法：
1. 创建 `AffinityMolDataset` 包装器
2. 移除氢原子
3. 归一化坐标
4. 对原子类型进行 token 化
5. 计算距离矩阵和边类型

#### 口袋 LMDB（例如 `tnks2.lmdb`）
每个条目包含：
- `pocket_atoms`：残基-原子类型列表（例如 `["CA", "CB", "N", ...]`）
- `pocket_coordinates`：3D 坐标，numpy 数组 `(N_atoms, 3)`
- `pocket`：口袋标识符
- `pocket_residue_name`：残基名称（可选）

代码通过 `load_pockets_dataset()` 加载，该方法：
1. 创建 `AffinityPocketDataset` 包装器
2. 移除氢原子
3. 将口袋裁剪到 `max_pocket_atoms`（默认 511）
4. 归一化坐标
5. 对残基-原子类型进行 token 化
6. 计算距离矩阵和边类型

### 如何查看数据集格式

#### 方法 1：直接查看 JSON 标签
```python
import json

# 加载 FEP 标签
with open("test_datasets/FEP/fep_labels.json") as f:
    fep_labels = json.load(f)

# 打印第一个目标的信息
target = fep_labels[0]
print(f"目标: {target['pockets']}")
print(f"UniProt: {target['uniprot']}")
print(f"序列长度: {len(target['sequence'])}")
print(f"配体数量: {len(target['ligands'])}")
print(f"第一个配体: {target['ligands'][0]}")
```

#### 方法 2：查看 LMDB 数据
```python
from unicore.data import LMDBDataset

# 加载配体数据集
lig_dataset = LMDBDataset("test_datasets/FEP/lmdbs/tnks2_lig.lmdb")
print(f"配体数量: {len(lig_dataset)}")
print(f"第一个配体的键: {lig_dataset[0].keys()}")
print(f"第一个配体的 SMILES: {lig_dataset[0]['smi']}")

# 加载口袋数据集
pocket_dataset = LMDBDataset("test_datasets/FEP/lmdbs/tnks2.lmdb")
print(f"口袋数量: {len(pocket_dataset)}")
print(f"第一个口袋的键: {pocket_dataset[0].keys()}")
```

## 4. 代码流程总结

```
test.sh（FEP 任务）
    └── unimol/test.py::main()
            └── test_task.test_fep(model)
                    ├── 加载标签：test_datasets/FEP/fep_labels.json
                    └── 对于每个目标：
                            ├── test_fep_target(target, model, label_info)
                            │       ├── load_mols_dataset(FEP/lmdbs/{target}_lig.lmdb)
                            │       ├── load_pockets_dataset(FEP/lmdbs/{target}.lmdb)
                            │       ├── 计算分子嵌入
                            │       ├── 使用序列计算口袋嵌入
                            │       └── 计算相关性（R²）
                            └── 报告所有目标的平均/中位数 R²
```

## 5. 结果处理

运行 zero-shot FEP 测试后，结果保存到：
- `{results_path}/FEP/{target}/saved_mols_embed.npy` - 分子嵌入
- `{results_path}/FEP/{target}/saved_target_embed.npy` - 口袋嵌入
- `{results_path}/FEP/{target}/saved_labels.npy` - 真实活性值
- `{results_path}/FEP/{target}/saved_smis.json` - SMILES 字符串

最终集成结果使用 `ensemble_result.py` 计算：
```bash
python ensemble_result.py zeroshot FEP
```

这会将 pocket ranking 和 protein ranking 模型的预测结果组合以产生最终指标。
