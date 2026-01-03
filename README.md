# Synt.Eye: Empowering Vision-Based Industrial AI with Synthetic Data

<p align="center">
  <img src=https://github.com/rparak/Synth_Eye/blob/main/images/Logo_White.png width="800" height="400">
</p>

## Project Overview

**Synth Eye** is a modular platform for generating **high-quality, photorealistic synthetic data** that faithfully replicates real-world environments, objects, and operating conditions. The generated datasets are intended to improve the efficiency, robustness, and overall performance of neural network training, with a primary focus on **computer vision applications in industrial manufacturing**.

The platform is specifically designed to support **visual inspection and automated quality control tasks**, enabling the development and validation of machine vision systems in scenarios where real data acquisition is costly, time-consuming, or limited.

The project was developed as part of **internal research activities at the Research and Innovation Center INTEMAC**.

### What Is Synthetic Data?

**Synthetic data** refers to high-quality, artificially generated data that replicates real-world environments, objects, and operating conditions. It is used to enhance the efficiency, robustness, and performance of neural network learning, particularly in cases where collecting, labeling, or scaling real-world data is challenging or impractical.

### Features

- High-quality, photorealistic synthetic data generation.
- Synthetic data tailored for **visual inspection and quality control** applications.
- Targeted support for industrial and manufacturing use cases.
- Modular and easily extensible system architecture.
- Compatibility with object detection and vision-based machine learning pipelines.
- Optimized for efficient dataset generation and experimental workflows.


## Installation

This project relies on a dedicated **Conda environment** to ensure a reproducible, isolated, and stable setup across platforms. Automated installation scripts are provided for both **Linux/macOS** and **Windows**.

### Prerequisites

Before proceeding, ensure the following requirements are met:

- Operating System:
  - **Linux / macOS**
  - **Windows**
- **Miniconda** installed and available in the system PATH  
  https://docs.conda.io/en/latest/miniconda.html
- **Blender** installed (required for synthetic data generation)  
  https://www.blender.org/download/
- Git

> ⚠️ **Note:** Ensure that the installed Blender version is compatible with the project scripts and is accessible from the command line if required.

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/rparak/Synth_Eye.git
cd Synth_Eye
```

#### 2. Platform-Specific Installation
#### 2.1 Linux / macOS

Make the installation script executable:

```bash
chmod +x install.sh
```

Run the installation script:

```bash
./install.sh
```

#### 2.2 Windows

Open Command Prompt and navigate to the project directory:

```bash
cd path\to\Synth_Eye
```

Run the installation script:

```bash
install.bat
```

#### 3. Verification

If the installation completes successfully, the Conda environment is fully configured and verified.

Activate the environment manually:

```bash
conda activate env_synth_eye
```

The system is now ready for synthetic data generation, experimentation, and model training.

## TODO

- Improve the light configuration to better support rectangular objects.
- Add a standardized dataset template structure to the root of the repository.
- Remove the `time.sleep(10)` delay from `gen_synthetic_data.py`.
- Ensure proper handling of label `.txt` files:
  - If a label file already exists for a given index, delete or overwrite it instead of appending new bounding boxes to the existing file.

## Contributors

<table> <tr> <td align="center"> <a href="https://github.com/rparak"> <img src="https://avatars.githubusercontent.com/rparak" width="120px;" alt="Roman Parak"/><br /> <strong>Roman Parak</strong> </a><br /> </td> <td align="center"> <a href="https://github.com/LukasMoravansky"> <img src="https://avatars.githubusercontent.com/LukasMoravansky" width="120px;" alt="Lukas Moravansky"/><br /> <strong>Lukas Moravansky</strong> </a><br /> </td> </tr> </table>

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

You are free to use, modify, distribute, and sublicense this software, provided that the original copyright notice and permission notice are included in all copies or substantial portions of the software.

## Acknowledgements

This project was developed as part of internal research activities at the **Research and Innovation Center INTEMAC**, with a focus on advancing synthetic data generation for **industrial artificial intelligence and machine vision applications**.


