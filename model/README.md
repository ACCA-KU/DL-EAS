# Pretrained Model Weights

The pretrained DL-EAS weights are split into multiple parts due to GitHub’s file size limit (25 MB).

To reconstruct the full model checkpoint, download all parts in this folder and run the following command in your terminal:

```bash
cat pretrained_weight.vol*.egg > pretrained_weight.ckpt
