# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

我们使用pixi进行包管理。首先安装pixi:

Linux & macOS:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Windows:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

安装完成后,在项目根目录运行:
```bash
pixi install
```

这将自动安装所有必需的依赖项。

### Example Usages

要激活pixi环境,在终端中运行:
```bash
pixi shell
```

### Simulated experiments

我们使用 ``sim_transfer_cube_scripted`` 任务作为示例。另一个可选任务是 ``sim_insertion_scripted``。

数据集的保存路径和数量配置在 `constants.py` 中的 `SIM_TASK_CONFIGS` 字典中。默认配置如下:
```python
DATA_DIR = "data"  # 数据保存的默认路径
SIM_TASK_CONFIGS = {
    "sim_transfer_cube_scripted": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
    ...
}
```

你可以根据需要修改这些配置。要生成示教数据，运行:

```bash
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted
```

可以添加 ``--onscreen_render`` 参数来查看实时渲染。
要查看记录的数据集，运行:

```bash
python3 visualize_episodes.py --dataset_dir data/sim_transfer_cube_scripted --episode_idx 0
```

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

