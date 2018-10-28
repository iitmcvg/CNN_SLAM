# Download Monodepth Checkpoints

We presently use [monodepth](g3docs/monodepth.md) as our depth estimation engine.

To download the checkpoints, run:

```
bash utils/download_monodepth_ckpt.sh
```

Ensure you have [`aria2`](https://aria2.github.io) installed for the same. If you prefer to use `wget`, uncomment line 13 in `monodepth/utils/get_model.sh`.

These are downloaded to `checkpoints/`.

The following are included:
* `model_kitti_resnet50`: main model trained on the **kitti** split
* `model_eigen_resnet50`: main model trained on the **eigen** split
* `model_cityscapes_resnet50`:  main model trained on **cityscapes**
* `model_city2kitti_resnet50`: `model_cityscapes` fine-tuned on **kitti**
* `model_city2eigen_resnet50`: `model_cityscapes` fine-tuned on **eigen**