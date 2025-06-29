# Aligning Agents Like LLMs
This repository contains supplmentary videos for the [paper](https://arxiv.org/abs/2406.04208) "Aligning Agents like LLMs".

The videos show sample behaviours of our trained models in the Bleeding Edge environment. Our agents receive the *same* images shown as input (which are resized and fed into CNN layers), and output a gamepad action at each timestep.

If the videos do not play in your browser, please use a different browser (such as Chrome) or check out the corresponding videos in the Supplementary Material.

The videos are organised as follows:

- **Base Model**

  - This folder contains trajectories demonstrating some undesirable behaviour of the 'Base Model' when rolled out (corresponding to Section 3.1).

- **Fine-Tuned Model**
  - This folder contains 1 example of the 'Fine-Tuned Model' going to each of the Left/Middle/Right jumppads (corresponding to Section 3.2). It also contains a video of the agent missing a jumppad, but then turning around to hit it - a behaviour that was not present in the fine-tuning dataset, but was present in the pre-training dataset, demonstrating an unexpected benefit of pre-training.

- **Pre-Training Ablation Model**

  - This folder contains an example of a Fine-Tuned Only Model (without pre-training, corresponding to Section 3.2 and Appendix E.1). In this example the agent misses a jumppad and gets stuck, similar to the trajectory above but where the Fine-Tuned Model (with pre-training) was able to turn around and reach the jumppad.

- **Aligned towards Left Jumppad**

  - Contains representative trajectories of the 'Preference FT + 500k Reward Model' agent that has been successfully aligned with navigating to the left jumppad (corresponding to Section 3.5.1).

- **Aligned towards Right Jumppad**

  - Equivalent to the above for an agent instead aligned with navigating to the right jumppad (corresponding to Section 3.5.2).
