## Inference with Detectron
To run inference using pretrained Detectron models on 
[Deep Learning AMI Ubuntu Linux - 2.5_Jan2018 (ami-1aa7c063)](https://aws.amazon.com/marketplace/pp/B06VSPXKDX)
Instance Type: p2.xlarge:  

1. Run [`setup.sh`](Detectron/setup.sh) which will install:  
- Caffe2 with [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron) (takes about 1.5 hour!)
- Python dependencies
- [COCO API](https://github.com/cocodataset/cocoapi)
- [Detectron](https://github.com/facebookresearch/Detectron)
2. Add [`infer_simple_new.py`](Detectron/infer_simple_new.py) to the folder `/home/ubuntu/src/detectron/tools/`  

3. Create a folder with images you want to use (`/home/ubuntu/image_originals/` in our case). All the images must be 
of the same extension: JPG or PNG. We will use `JPG`.  
4. Decide where and with which extension (JPG, PNG or default PDF) you want to save the output. 
In our case `JPG` and `/home/ubuntu/image_results/`.  
5. Choose a model from the [model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). In this example, we're using an end-to-end trained Mask 
R-CNN model with a ResNet-101-FPN backbone.
4. Run the following command:
```
python2 /home/ubuntu/src/detectron/tools/infer_simple_new.py \
    --cfg /home/ubuntu/src/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /home/ubuntu/image_results \
    --image-ext jpg \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    --output-ext jpg \
	/home/ubuntu/image_originals
```
