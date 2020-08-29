## Introduction
This repo consists of code accompanying the submission "AutoFuzz: Grammar-Based Fuzzing forSelf-Driving Car Controller". AutoFuzz is a grammar-based input fuzzing tool for self-driving car, which analyzes CARLA specification to generate semantically and temporally valid test scenario.


## Setup
### Requirements
* OS: Ubuntu 18.04
* CPU: at least 8 cores
* GPU: at least 8GB memory
* Carla 0.9.9 (for installation details, see below)
* Python 3.7 with packages listed in `requirements.yml`

### Cloning this Repository

Clone this repo

```
git clone https://github.com/autofuzz2020/AutoFuzz
```

All python packages used are specified in `requirements.yml`.

This code uses CARLA 0.9.9.

You will also need to install CARLA 0.9.9, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.



### Installation of Carla 0.9.9
The following commands can be used to install carla 0.9.9

Create a new conda environment:
```
conda create --name carla99 python=3.7
conda activate carla99
```
Download CARLA_0.9.9.4.tar.gz and AdditionalMaps_0.9.9.4.tar.gz from [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) and run
```
mkdir carla_0994_no_rss
tar -xvzf CARLA_0.9.9.4.tar.gz -C carla_0994_no_rss
```
move `AdditionalMaps_0.9.9.4.tar.gz` to `carla_0994_no_rss/Import/` and in the folder `carla_0994_no_rss/` run:
```
./ImportAssets.sh
```
Then, run
```
cd carla_0994_no_rss/PythonAPI/carla/dist
easy_install carla-0.9.9-py3.7-linux-x86_64.egg
```
Test the installation by running
```
cd ../../..
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```
A window should pop up.

### Download a LBC pretrained model
Download the checkpoint from [Wandb project](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc).

Navigate to one of the runs, like https://app.wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files

Go to the "files" tab, and download the model weights, named "epoch=24.ckpt", and pass in the file path as the `TEAM_CONFIG` in `run_agent.sh`. Move this model's checkpoint to the `models` folder.



## Run Fuzzing
```
python ga_fuzzing.py
```
For more API information, checkout `ga_fuzzing.py`.





## Check out maps and find coordinates
Check out the map details by spinning up a CARLA server

```
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```
and running
```
python inspect_routes.py
```
Also see the corresponding birdview layout [here](https://carla.readthedocs.io/en/latest/core_map/) for direction and traffic lights information.

Note to switch town map, one can change the corresponding variable inside this script.



## Retrain model from scratch
Note: the retraining code only supports single-GPU training.
Download dataset [here](https://drive.google.com/file/d/1dwt9_EvXB1a6ihlMVMyYx0Bw0mN27SLy/view). Add the extra data got from fuzzing into the folder of the dataset and then run stage 1 and stage 2.

Stage 1 (~24 hrs on 2080Ti):
```
CUDA_VISIBLE_DEVICES=0 python carla_project/src/map_model.py --dataset_dir path/to/data
```

Stage 2 (~36 hrs on 2080Ti):
```
CUDA_VISIBLE_DEVICES=0 python carla_project/src/image_model --dataset_dir path/to/data --teacher_path path/to/model/from/stage1
```




# Reference
This repo is built on top of [here](https://github.com/bradyz/2020_CARLA_challenge) and [here](https://github.com/msu-coinlab/pymoo)
