# TTA_utils
## TTA implementation 
* fork lib from git : https://github.com/qubvel/ttach
  tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(),  merge_mode='mean')

* parameter: 
    model: input model
   
    tta.aliases.five_crop_transform(): transformation type
    (using it from aloases.py)
    
     OR self-define transforms here!
     e.g:
      
      @define a 2 * 2 * 3 * 3 = 36 augmentations!
      
      transforms = tta.Compose(
        [
          tta.HorizontalFlip(),
          tta.Rotate90(angles=[0, 180]),
          tta.Scale(scales=[1, 2, 4]),
          tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
     )
   
   merge_mode: the way to merge data, for image classification: using 'mean, geo-mean'

