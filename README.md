# MPAIL2

![MPAIL2 Demo](docs/media/mpail2-teaser.gif)
<p align="center">
  <a href="https://www.youtube.com/watch?v=yQw0JmvOVwM">🎥 Demo Video</a> •
  <a href="https://uwrobotlearning.github.io/mpail2/">🌐 Website</a> •
  <a href="https://arxiv.org/pdf/2602.24121">📄 Paper</a>
</p>


## Quick Start

### Installation

> Tested on **Ubuntu 22.04** with **Python 3.10**

```
conda create -n mpail2 python=3.10
conda activate mpail2
pip install --upgrade pip
pip install -e .
```

This is the default install for `mpail2`, including Gym / MuJoCo and the Python packages used by the real-robot stacks. IsaacLab and the real-robot environments still require their own system-level setup first, so install Isaac Sim / IsaacLab, ROS 2 / `ros2_kortex`, or the Franka control stack before using those backends. See [`docs/INSTALL.md`](docs/INSTALL.md) for environment-specific setup.

### Training (Isaac or Gymnasium)

One entry point: [`python -m mpail2.train.train`](mpail2/train/train.py) or `mpail2-train`. If ``--env`` matches an Isaac task (default is ``push`` when omitted), Hydra + Isaac Lab run; otherwise the Gymnasium / MuJoCo script runs (e.g. ``Ant-v5``).

```bash
python mpail2/train/train.py --env push --headless log.no_wandb=True   # Isaac
python mpail2/train/train.py --env Ant-v5                              # Gym (MuJoCo)
python mpail2/train/train.py --env Humanoid-v5 log.video=True
```

See [`mpail2/train/README.md`](mpail2/train/README.md).

## About the Files

![MPAIL2 Overview](docs/media/mpail2.drawio.png)

### Agent
1. `runner.py` : Outer-most loop. Steps environment and calls `act()` on the learner.
2. `learner.py` : Stores interactions, calls planner, and updates component models.
3. `planner.py` : Performs online planning (MPPI) using component models.

**All loss computations and gradient updates are performed within `learner.py`**. For those interested in reading the implementation, `learner.py` is the file to begin with.

### Component Models
Composed by the planner are the component models in the above figure and discussed in Section 3 of the paper.

1. `encoder.py` : expects `Dict[str,tensor]` observations
2. `dynamics.py` : $f:\mathcal{Z} \times \mathcal{A}^{H} \rightarrow \mathcal{Z}^{H+1}$
3. `reward.py`: $r:\mathcal{Z}\times\mathcal{Z}\rightarrow \mathbb{R}$
4. `value.py`: Ensembled, $Q:\mathcal{Z}\times\mathcal{A}\rightarrow \mathbb{R}$
5. `sampling.py` (policy): composes the policy $\pi(\mathbf{a}_{t:t+H}|z)$. Uses policy and previous plan to sample plans from policy and fitted gaussian.

Except for `sampling.py`, these files are primarily `torch.nn.Modules` with `forward` definitions as you expect their mathematical representations to be.

# BibTex

If you found this work useful in your efforts, please consider citing:

```
@article{han2026planning,
  title   = {Planning from Observation and Interaction},
  author  = {Han, Tyler and Shen, Siyang and Baijal, Rohan and Ravichandran, Harine and Nemekhbold, Bat and Huang, Kevin and Jung, Sanghun and Boots, Byron},
  journal = {arXiv preprint arXiv:2602.24121},
  year    = {2026}
}
```
