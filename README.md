# MPAIL2

![MPAIL2 Demo](media/mpail2-teaser.gif)
https://www.youtube.com/watch?v=yQw0JmvOVwM

[Website](https://uwrobotlearning.github.io/mpail2/) | [Paper](https://arxiv.org/pdf/2602.24121)

**3/2/2026**: This repository contains the MPAIL2 algorithm implementation. Examples of IsaacLab or real-world workflows will be included later or in a separate repository.

## Quick Start

### Installation

```
git clone git@github.com:UWRobotLearning/mpail2.git
cd mpail2
conda create -n mp2 # OPTIONAL
pip install -e .
```

### (Gymnasium) Training

Ant:
```
python train/train_gym_mpail.py --video # Defaults to Ant-v5
```

Humanoid:
```
python train/train_gym_mpail.py --env Humanoid-v5 --video
```

## About the Files

![MPAIL2 Overview](media/mpail2.drawio.png)


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

- `layers.py`
- `storage.py`