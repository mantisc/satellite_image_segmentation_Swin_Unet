This project aims to train a Swin-Unet based model to complete the segmentation task on satellite images provided on Sentinel-Hub and corresponding label masks.



Comparison between ground truth label map and predicted label map.
![image](https://github.com/mantisc/satellite_image_segmentation_Swin_Unet/blob/main/image_versus_mask.png)



Metric Monitor in training process
![image](https://github.com/mantisc/satellite_image_segmentation_Swin_Unet/blob/main/metric_monitor_wandb.jpg)


Subsequent work will focus on improve model's performance on small scale categories such as road and water. Deformable patch merging layer will be a feasible option.
