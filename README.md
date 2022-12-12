## Hierarchical and Contrastive Representation Learning for Knowledge-aware Recommendation

### HiCON

The code of paper "Hierarchical and Contrastive Representation Learning for Knowledge-aware Recommendation".

### Environment

Requirments

```python
pytorch == 1.5.0
numpy == 1.15.4
scipy == 1.1.0
sklearn == 0.20.0
torch_scatter == 2.0.5
torch_sparse == 0.6.10
networkx == 2.5
```

### Getting Started

 Run experiments

```python
# Train and test on Book-Crossing
python main_HiCON.py --dataset book --lr 1e-4

# Train and test on MovieLens-1M
python main_HiCON.py --dataset movie --lr 1e-3

# Train and test on LastFM
python main_HiCON.py --dataset music --lr 3e-3
```