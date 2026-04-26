# CBT 完整验证结果：102 化合物 + cc-pVTZ 16 复测

**日期：** 2026-04-26 09:30 GMT+8

## 总体统计

| Version | N | 基底 | H1 (δ_σ>0,δ_π<0) | H2 (退化抵消) |
|---------|---|------|-------------------|---------------|
| V6 | 25 | 6-31G* | 14/15 (93.3%) | 7/7 (100%) |
| V7 | 28 | 6-31G* | 20/20 (100%) | 7/7 (100%) |
| V8 | 49 | 6-31G* | 37/37 (100%) | 7/8 (87.5%) |
| V9 | 16 | cc-pVTZ | 12/12 (100%) | 3/3 (100%) |
| **6-31G* 合计** | **102** | — | **71/72 (98.6%)** | **21/22 (95.5%)** |
| **cc-pVTZ 合计** | **16** | — | **12/12 (100%)** | **3/3 (100%)** |

## 关键文件

- v6: `research/membrane-bond-theory/v6_25_molecules_decomp.py` + `v6_results.log`
- v7: `research/membrane-bond-theory/v7_30_more_decomp.py` + `v7_results.log`  
- v8: `research/membrane-bond-theory/v8_50_more.py` + `v8_results.log`
- v9: `research/membrane-bond-theory/v9_ccpVTZ_robustness.py` + `v9_results.log`
- 核心模块: `research/experiments/cbt/2026-04-26/loop_1/sigma_pi_decomposition.py`

## 教科书"反例"全部用新理论统一解释

| 系统 | ε | 新理论解释 |
|------|---|-----------|
| N₂F₄ N-N | 0.69 | 4 F 让 σ 框架各向异性 |
| 环丙烷 C-C | 0.84 | 应变弯曲让 σ 不圆柱 |
| 全氟乙烯 C=C | 1.23 | F 取代 σ 框架主导 |
| BH₃ B-H | 0.28 | 空 p_z 让 σ 不圆柱 |
| CHF₃ C-F | 0.21 | 3 F 不对称分布 |
| 环氧乙烷 C-O | 0.18 | 应变环 + O LP |
| Allene 中央 C=C | 0.53 | 局部 σ 不对称 |

## 论文 Title Candidates

1. *"Bond-Critical-Point Ellipticity Probes σ-Framework Anisotropy, Not π Bonding"*
2. *"The Sign of the π Contribution to QTAIM Bond Ellipticity is Negative"*
3. *"Reinterpretation of QTAIM Bond Ellipticity through Per-MO Hessian Decomposition"*
