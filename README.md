# Fast Matrix Factorization for Implicit Feedback (eALS, RDD-based)

This project includes `fast_mf_pipeline.ipynb`, which implements the eALS method introduced in the original paper:

```bibtex
@inproceedings{he2016fast,
  title={Fast matrix factorization for online recommendation with implicit feedback},
  author={He, Xiangnan and Zhang, Hanwang and Kan, Min-Yen and Chua, Tat-Seng},
  booktitle={SIGIR},
  pages={549--558},
  year={2016},
  publisher={ACM},
  address={Pisa, Italy},
  doi={10.1145/2911451.2911489}
}
```

## Dataset

- Same as the paper: `AmazonMoviesDataset.txt`
- Format: one review record per block (10 lines), records separated by blank lines.
- Fields parsed: `product/productId`, `review/userId`, `review/time`.

### Download

1. Go to the SNAP Amazon links page: https://snap.stanford.edu/data/web-Amazon-links.html
2. Download the Amazon Movies dataset archive (`movies.txt.gz`) from that page.
3. Place it in `data/` and extract it as `data/AmazonMoviesDataset.txt`.

Example:

```bash
mkdir -p data
curl -L https://snap.stanford.edu/data/movies.txt.gz -o data/movies.txt.gz
gunzip -c data/movies.txt.gz > data/AmazonMoviesDataset.txt
rm -f data/movies.txt.gz
```

## Implementation

- eALS objective with paper's Amazon settings:
  - `K=128`, `lambda=0.01`, `c0=64`, `alpha=0.5`, `w_ui=1`, `r_ui=1`.
- Iterative 10-core filtering (`>=10` interactions for both users and items).
- Chronological leave-one-out split (latest item per user used as test).
- Map-and-broadcast training design:
  - Interaction histories are RDDs.
  - `P` and `Q` are NumPy arrays on the driver.
  - `S^q` and `S^p` are computed on the driver and broadcast each phase.
- Offline ranking evaluation:
  - score all items, mask training items, compute `HR@100` and `NDCG@100`.

## Project layout

- `src/data_ingest.py` parsing, filtering, indexing, split, history building, and data preparation.
- `src/train_eals.py` Eq.12/Eq.13 coordinate updates and training loop.
- `src/evaluate.py` HR/NDCG evaluation.
- `fast_mf_pipeline.ipynb` notebook with visible step-by-step execution.

## Run

Open and execute `fast_mf_pipeline.ipynb`.

## Output artifacts

Each run writes:

- `P.npy`, `Q.npy` (trained factor matrices)
- `metrics.json` (`hr`, `ndcg`, `evaluated_users`)
- `config.json` (eALS + runtime settings)
- `prepare_stats.json` (stage-level preprocessing counts and timings)
- `train_log.json` (iteration timing)
- `id_maps.json` (`user_ids`, `item_ids` ordered by internal index)
- `run_summary.md`
