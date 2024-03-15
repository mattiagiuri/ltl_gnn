Since the DeepMind control suite uses a different version of Mujoco from safety_gymnasium, two different conda environments are required.

Python 3.10 is currently the maximum version supported by safety-gymnasium.

```bash
conda create -n deepltl python=3.10
cd envs/zones/safety-gymnasium
pip install -e .
```

`conda create -n deepltl-dmc python=3.10`

