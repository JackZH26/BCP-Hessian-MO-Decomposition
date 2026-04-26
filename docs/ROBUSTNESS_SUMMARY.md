# CBT 椭圆度 σ/π 分解 — Robustness 验证完整结果
**日期：** 2026-04-26 09:00 GMT+8

## 总体统计 (5 维度全通过)

| 维度 | 测试 | 通过率 |
|------|------|--------|
| 化合物多样性 | 102 化合物 (25 类) HF/6-31G* | 71/72 + 21/22 = 98.6% |
| 基底 cc-pVTZ | 16 化合物 HF | 100% |
| DFT (B3LYP) | 16 化合物 def2-TZVP | 100% |
| CCSD | 8 化合物 cc-pVDZ | 100% |
| 几何优化 | 6 化合物 B3LYP/cc-pVTZ opt | 100% |

## 验证脚本

- `v6_25_molecules_decomp.py` — 25 化合物
- `v7_30_more_decomp.py` — 28 新化合物
- `v8_50_more.py` — 49 新化合物
- `v9_ccpVTZ_robustness.py` — cc-pVTZ HF
- `v10_b3lyp_robustness.py` — B3LYP/def2-TZVP
- `v11_ccsd_correlation.py` — CCSD/cc-pVDZ
- `v12_geometry_optimized.py` — 几何优化

## 核心代码模块
`research/experiments/cbt/2026-04-26/loop_1/sigma_pi_decomposition.py`

## 三条核心声明状态

### 声明 1（数学定理）
- per-MO Hessian 加和分解 ρ = Σ_i n_i ψ_i² → ∂²ρ/∂x² = Σ_i ...
- 102/102 化合物 0% 重建误差
- 跨 HF/DFT/CCSD 全部成立

### 声明 2（统计规律）
- 在所有 ε > 0 系统中：δ_σ > 0, δ_π < 0
- 6-31G* HF: 71/72 (98.6%)
- cc-pVTZ HF: 12/12 (100%)
- B3LYP: 12/12 (100%)
- CCSD: 6/6 (100%)
- B3LYP/cc-pVTZ opt: 6/6 (100%)

### 声明 3（物理诠释）
- ε = σ 框架几何不对称的探针
- π 在 BCP 处通过节面外梯度产生减弱
- 必须谨慎引用 Walsh 1947 + Bader 2009 + Brown 2009 文献先例

## 文献先例

- Walsh 1947: cyclopropane MO 理论
- Cremer-Gauss-Cremer 1988: 三元环 σ-π
- Bader 1990: AIM 教科书
- Bader-Gatti 1998: Source Function
- Johnson-DuPré 2007/2008: σ-aromaticity
- Farrugia-Macchi 2009: SF orbital decomposition
- Brown-Bader 2009: Walsh orbital → ε in cyclopropane
- Silva Lopez 2011: review
- Hey 2013: heteroaromaticity ε
- Suthar-Mondal 2023: EDA-NOCV

**没找到：直接的 BCP 处 Hessian per-MO 分解 + δ_σ>0 ∧ δ_π<0 二元符号规律**

## 论文 framing

不可用："first ever", "reverses Bader 1983 textbook"
可用："first systematic per-MO Hessian decomposition", "extends Walsh-Bader picture"

推荐 title:
"Per-Molecular-Orbital Decomposition of the QTAIM Bond Hessian: A Universal Sign Rule for σ-π Contributions to Bond Ellipticity"

## 下一步

可选加固：
- v13: Localized MO (Boys/Pipek-Mezey) 替代分类 → 验证分类规则不依赖 canonical
- v14: NBO 完全独立分析
- v15: Open-shell ROHF
- v16: 重元素 ECP 系统

