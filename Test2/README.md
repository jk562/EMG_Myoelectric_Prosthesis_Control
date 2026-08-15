# Test 2 — Self-Supervised Pretraining: Contrastive vs. Masked-Reconstruction

**Part 2 — Contrastive and Masked-Reconstruction Pretraining, then Fine-tuning.**

`Test1` trained a CNN from scratch, end-to-end, on one subject's labelled EMG windows. This stage asks: can the same CNN encoder be pretrained on unlabelled EMG from many subjects, then fine-tuned on the small labelled set, to get better/more robust/more label-efficient kinematic regression than training from scratch?

## What's here

`EMG_Project2.ipynb` — pretrains the encoder two competing ways (contrastive/SimCLR-style, and masked-reconstruction), then fine-tunes and compares both against a from-scratch baseline on:

- **Fine-tune comparison** (`fig3_finetune_comparison.png`) — scratch vs. contrastive vs. masked.
- **Label efficiency** (`fig4_label_efficiency.png`) — accuracy vs. amount of labelled fine-tuning data.
- **Robustness** (`fig5_robustness_comparison.png`) — accuracy under noise.
- **Pretraining losses** — `fig1_contrastive_loss.png`, `fig2_masked_loss.png`.
- `fig6_prosthetic_hand_demo.gif` — a qualitative demo driven by the resulting model.

## Outcome

Contrastive (SimCLR-style) pretraining came out ahead as the best all-round encoder — this is the choice `Test3` builds its streamlined pipeline on. (Masked-reconstruction SSL was revisited much later, independently, in `Test9`'s final system, where it was re-tested and kept for a real, verified accuracy gain on the weakest joints.)

Note trained model weights from this notebook were not saved to disk (the run's checkpoint only existed in the notebook's kernel); see `Test4/train_masked_checkpoint.py` for a standalone reproduction of the masked-SSL half of this pipeline.
