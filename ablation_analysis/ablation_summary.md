# Ablation Study Results Summary

| Configuration    |   Dice |   Δ Dice |   Jaccard |   Δ Jaccard | Transformer   | Complexity   | Entropy   |
|:-----------------|-------:|---------:|----------:|------------:|:--------------|:-------------|:----------|
| all              |      0 |        0 |         0 |           0 | ✓             | ✓            | ✓         |
| no_transformer   |      0 |        0 |         0 |           0 | ✗             | ✓            | ✓         |
| no_complexity    |      0 |        0 |         0 |           0 | ✓             | ✗            | ✓         |
| no_entropy       |      0 |        0 |         0 |           0 | ✓             | ✓            | ✗         |
| only_transformer |      0 |        0 |         0 |           0 | ✓             | ✗            | ✗         |
| only_complexity  |      0 |        0 |         0 |           0 | ✗             | ✓            | ✗         |
| only_entropy     |      0 |        0 |         0 |           0 | ✗             | ✗            | ✓         |
| none             |      0 |        0 |         0 |           0 | ✗             | ✗            | ✗         |

## Key Findings:

- **Best Configuration**: `all` with Dice=0.0000

### Component Importance:

- **Transformer**: 0.0000 (with) vs 0.0000 (without) = +0.0000
- **Complexity**: 0.0000 (with) vs 0.0000 (without) = +0.0000
- **Entropy**: 0.0000 (with) vs 0.0000 (without) = +0.0000
