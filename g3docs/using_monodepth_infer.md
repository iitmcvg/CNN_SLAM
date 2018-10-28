# Using Monodepth Infer

Included in `monodepth_infer`:
* `monodepth_single`
* `monodepth_batch`

## Using Batch mode

Runs with a batch size of 1 (cant go higher), on a directory:

Directory must have structure:
```

```

Run as: (from `monodepth_infer`)

```

usage: monodepth_batch.py [-h] [--path PATH]
                          [--checkpoint_path CHECKPOINT_PATH]
                          [--image_format IMAGE_FORMAT]

Monodepth Batch Inference.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to files. Needs to have a rgb folder containing
                        images.
  --checkpoint_path CHECKPOINT_PATH
                        path to checkpoint
  --image_format IMAGE_FORMAT
                        path to files. Needs to have a rgb folder containing
                        images.
```