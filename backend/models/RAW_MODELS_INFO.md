# RAW Models Info

Ovi modeli su pretrained na ImageNet i NISU fine-tuned za AI/Real detekciju.

## Modeli:
- EfficientNet-B0 (RAW): Pretrained na ImageNet, classifier head za 2 klase
- ViT-B/16 (RAW): Pretrained na ImageNet, classifier head za 2 klase

## Kako koristiti:
1. Backbone je FROZEN (ImageNet features)
2. Classifier head je RANDOM inicijaliziran
3. Za dobre rezultate, treba LINEAR PROBE na AI/Real podacima

## Datoteke:
- models/cnn_raw/model.pth
- models/vit_raw/model.pth
