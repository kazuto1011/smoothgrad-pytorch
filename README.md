# SmoothGrad with PyTorch
WIP, not tested on GPU

## Dependencies

* Python 2.7
* PyTorch
* torchvision
* tqdm

## Examples

```bash
python main.py --image samples/cat_dog.png [--no-cuda] [--guided]
```

![](samples/cat_dog.png)

ResNet-152<br>
bull mastiff 54.2% @1<br>
#samples: 50

|Noise level (σ)|10%|15%|20%|
|:-|:-:|:-:|:-:|
|SmoothGrad [1]|![](samples/bull_mastiff_10.gif)|![](samples/bull_mastiff_15.gif)|![](samples/bull_mastiff_20.gif)|
|Guided Backprop + SmoothGrad|![](samples/bull_mastiff_10_guided.gif)|![](samples/bull_mastiff_15_guided.gif)|![](samples/bull_mastiff_20_guided.gif)|
## References
\[1\] D. Smikov, N. Thorat, B. Kim, F. Viégas, M. Wattenberg. "SmoothGrad: removing noise by adding noise". arXiv, 2017<br>
