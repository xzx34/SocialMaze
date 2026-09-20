---
annotations_creators:
- synthetic
language:
- en
license: cc-by-4.0
multilinguality:
- monolingual
pretty_name: SocialMaze Corrected Expanded Hidden Role Deduction
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- question-answering
tags:
- social-reasoning
- large-language-models
- benchmark
- hidden-role-deduction
configs:
- config_name: default
  data_files:
  - split: easy
    path: data/easy-*.parquet
  - split: hard
    path: data/hard-*.parquet
---

# SocialMaze: Corrected Expanded Hidden Role Deduction

This is the authoritative maintained data release for **SocialMaze: A
Benchmark for Evaluating and Enhancing Social Reasoning in Large Language
Models in Complex Social Environments**, published in Findings of EMNLP 2026.

Version **2.0.0** contains 200,000 corrected Hidden Role Deduction (HRD)
instances generated with the exhaustive solver in
[`xzx34/SocialMaze`](https://github.com/xzx34/SocialMaze):

| Split | Players | Rows | Player 1 role distribution | Parquet shards |
|---|---:|---:|---|---:|
| `easy` | 6 | 100,000 | 25,000 each: Investigator, Criminal, Rumormonger, Lunatic | 2 |
| `hard` | 10 | 100,000 | 25,000 each: Investigator, Criminal, Rumormonger, Lunatic | 5 |

Every row has been checked for unique solvability, answer agreement, schema
validity, round-trip conversion, a terminal answer in its reasoning trace,
unique stable ID, and unique observable-content fingerprint within its split.
A second complete generation with seed `20260920` produced the same content
hashes. See `validation_report.json`, the two generation metadata files and
`checksums.sha256` for auditable release details.

## Scope

This is a **corrected expanded HRD release**. It is not a complete downloadable
mirror of the paper's six-task, 70,000-instance evaluation collection. The
maintained GitHub repository provides HRD generation, solving and evaluation;
the other five tasks have archived scripts and small demonstration samples.
The workflow and SFT/DPO experiments are not fully released.

The separate [`MBZUAI/SocialMaze`](https://huggingface.co/datasets/MBZUAI/SocialMaze)
repository is an unmaintained 2025 legacy mirror made with the old generator.
It contains some ambiguous rows and a skewed Player 1 role mix, cannot be
updated by the current maintainer, and should not be used for new experiments.
For reproducibility, the former contents of this personal repository remain
available with `revision="legacy-v1"`.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("xzx34/SocialMaze")
easy = dataset["easy"]
hard = dataset["hard"]
```

Legacy access, when reproducing an older experiment:

```python
legacy = load_dataset("xzx34/SocialMaze", revision="legacy-v1")
# Or explicitly load the obsolete organizational mirror:
legacy_mirror = load_dataset("MBZUAI/SocialMaze")
```

## Schema

The existing flat schema is preserved: `task`, `system_prompt`, `prompt`,
`answer`, `reasoning_process`, and `round 1` through `round 3`. Version 2 adds
a stable `id` field.

## Authors

Zixiang Xu, Yanbo Wang, Yue Huang, Haomin Zhuang, Yujun Zhou, Jiayi Ye,
Sixian Li, Zirui Song, Lang Gao, Chenxi Wang, Zhaorun Chen, Wang Pan,
Yue Zhao, Jieyu Zhao, Xiangliang Zhang, and Xiuying Chen.

- Paper: https://arxiv.org/abs/2505.23713
- Project homepage: https://xzx34.github.io/socialmaze/
- Code: https://github.com/xzx34/SocialMaze
- Venue: Findings of EMNLP 2026

## License

The v2 data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@inproceedings{xu2026socialmaze,
  title={{SocialMaze}: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments},
  author={Xu, Zixiang and Wang, Yanbo and Huang, Yue and Zhuang, Haomin and Zhou, Yujun and Ye, Jiayi and Li, Sixian and Song, Zirui and Gao, Lang and Wang, Chenxi and Chen, Zhaorun and Pan, Wang and Zhao, Yue and Zhao, Jieyu and Zhang, Xiangliang and Chen, Xiuying},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  month={October},
  year={2026},
  address={Budapest, Hungary},
  publisher={Association for Computational Linguistics},
  note={To appear},
  eprint={2505.23713},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.23713}
}
```
