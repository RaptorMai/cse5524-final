
# CSE 5524 Final Project for Team 20

## Team member
Zheda Ma, Jnsu Yoo

---
Env setup

    python -m venv cse5524
    pip install -r requirements.txt

**Download fundation model checkpoints**
https://huggingface.co/docs/timm/en/index

We are using

 - BioCLIP
 - DINOv2-B
 - SigLIP
 - SigLIP2
 - CLIP-B

**Training details are in `./PEFT/`**

**Eval and prepare for submissoin for Kaggle**

    python eval-fungi.py --classifier_ckpt (trained ckpt path) --output_path submission/two_step.csv
