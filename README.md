# Aligning Agents Like LLMs
This repository contains supplmentary videos for the ICLR submission "Aligning Agents like LLMs".

The videos show sample behaviours of our trained models in the Bleeding Edge environment. Our agent receives the *same* images as input (they are then resized before being fed into the CNN layers), and outputs a gamepad action at each timestep.

The videos are organised as follows:

- **Base Model**
  
  - This folder contains rollouts demonstrating some undesirable behaviour of the 'Base Model' when rolled out (corresponding to Section 4.1).

- **Fine-Tuned Model**
- This folder contains 1 example each of the 'Fine-Tuned Model' going to the Left/Middle/Right jumppads, as well as it missing a jumppad, but then turning around to hit it - interestingly a behaviour that was not present in the fine-tuning dataset, but is present in the pre-training dataset (corresponding to Section 4.2).

- **Aligned towards Left Jumppad**
  
  - Contains representative rollouts of the 'Preference FT + 500k Reward Model' agent that has been aligned towards successfully hitting the left jumppad (corresponding to Section 4.5.1).

- **Aligned towards Right Jumppad**
  
  - Similarly to above for the right jumppad (corresponding to Section 4.5.2).



