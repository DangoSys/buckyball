# pebble regression eval

`models.toml` lists models for `--eval-performance` and per-model accuracy specs.

## Accuracy

Each model hardcodes its dataset relative path under `/root`. Kernel only packs
that directory from the host tree. No CLI flags.

| model | metric | relative path |
|-------|--------|---------------|
| lenet | top1 | `./datasets/MNIST` |
| mobilenet | top1 | `./datasets/ImageNet` |
| resnet | top1 | `./datasets/ImageNet` |
| yolo | map | `./datasets/COCO` |

Host source (copied to `/root/datasets/...`):

```text
bb-tests/workloads/src/ModelTest/e2e/datasets/
  MNIST/raw/t10k-*-ubyte
  ImageNet/labels.csv          # relpath,class_id
  ImageNet/images/...
  COCO/boxes.csv               # file,class_id,x1,y1,x2,y2
  COCO/images/...
```

Sample count = files/entries in that directory. Put a subset there for faster
eval; put the full set for full numbers.

UART:

- top1: `top1=<correct>/<total>`
- map: `map=<0..1>`

Top-level `accuracy` = mean of per-model accuracies.
