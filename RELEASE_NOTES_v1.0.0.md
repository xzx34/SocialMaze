# SocialMaze v1.0.0

This is the first formal release of the maintained SocialMaze Hidden Role
Deduction implementation.

Highlights:

- exhaustive HRD solver and corrected generator with observable-content
  fingerprint deduplication;
- corrected expanded dataset v2.0.0 at
  [`xzx34/SocialMaze`](https://huggingface.co/datasets/xzx34/SocialMaze)
  (100,000 easy and 100,000 hard examples, exact four-role balance, stable IDs
  and checksums);
- OpenAI-compatible evaluation and reporting pipeline with offline mocks;
- installable `models.yaml`, verified wheel/sdist, and Python 3.10–3.13 CI;
- archived scripts and synthetic demonstrations for the other five paper
  tasks, with real Amazon/OpenReview text removed;
- Apache-2.0 source-code license and CC BY 4.0 data license.

The release assets include the wheel, source distribution, and a SHA-256
manifest. The full six-task paper data, workflow implementation, and SFT/DPO
training pipeline are not included in this release.

Canonical resources:

- Project Page: https://xzx34.github.io/socialmaze/
- Paper: https://arxiv.org/abs/2505.23713
- Code: https://github.com/xzx34/SocialMaze
- Dataset: https://huggingface.co/datasets/xzx34/SocialMaze
