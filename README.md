# PHC-text2motion
This repository serves as a simple starting point for the text-to-motion component of [PHC](https://github.com/ZhengyiLuo/PHC). The majority of the code is sourced from [PHC](https://github.com/ZhengyiLuo/PHC).

## Installation

### PHC
To setup PHC, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda activate isaac
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/ZDDWLIG/PHC-text2motion.git
cd PHC-text2motion
pip install -r requirement.txt
```

2. Setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 

```
wget -O isaac-gym.tar.gz https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym.tar.gz
cd isaac-gym/python
conda activate isaac
pip install -e .

```


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). 

```
wget -O SMPL_python_v.1.1.0.zip https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip
wget -O models_smplx_v1_1.zip https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip
unzip SMPL_python_v.1.1.0.zip
unzip models_smplx_v1_1.zip

```
PS. Register first!

Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```


Make sure you have the SMPL paramters properly setup by running the following scripts:
```
python scripts/vis/vis_motion_mj.py
python scripts/joint_monkey_smpl.py
```

4. Use the following script to download trained models and sample data.

```
cd PHC-text2motion
bash download_data.sh
```
### MDM
To setup MDM, follow the following instructions: 

1. Create new conda environment:


```
sudo apt update
sudo apt install ffmpeg
git clone https://github.com/GuyTevet/motion-diffusion-model.git
cd motion-diffusion-model
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

2. Get data and model:

Download dependencies:

```
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```
If you have trouble downloading those dependencies, check [here](https://drive.google.com/drive/folders/1L6SrlxvxDh2GTfF-fu_yoKGdzMVCIz5T?dmr=1&ec=wgc-drive-globalnav-goto).

Download text data:

```
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
cd motion-diffusion-model
```

Download pretrained models:

```
wget -O humanml_trans_enc_512.zip https://drive.google.com/file/d/1DXRBHTb7XgCKcxUvJR3wyYkw_GJr2rvX/view?usp=drive_link
unzip humanml_trans_enc_512.zip

```


## Running

Go to MDM directory and run:

```
cd motion-diffusion-model
python language_to_pose_server.py --model_path ./humanml_trans_enc_512/model000475000.pt
```

Then, you can start typing the language commands in the server script after the "Type MDM Prompt:" message. Press "return" to send more commands. 

Open a new terminal and run the following command to launch the simulation.

```
cd PHC-text2motion
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv env=env_im_getup_mcp env.task=HumanoidImMCPDemo robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1 test=True no_virtual_display=True
```