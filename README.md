# AdaMOT
Codes for the TIP paper "A closer look at the joint training of object detection and Re-identification in multi-object tracking"  

![Alt text](https://raw.githubusercontent.com/DemoGit4LIANG/AdaMOT/main/Screenshots/main1.png)

Our main contributions, __Identity-aware Label Assignment__ and __Discriminative Focal Loss__ are implemented in __ada_matcher.py__ and __ada_loss.py__.  

The entire project is largely based on FairMOT (FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking), most of the upgraded ccomponments in AdaMOT are named as "ada_xxx.py".  

Our results:  
![Alt text](https://raw.githubusercontent.com/DemoGit4LIANG/AdaMOT/main/Screenshots/results.png)  

##### Usage:
        (To train) python ada_train.py --exp_id adamot_17 --load_model detector_pretrained_weights.pth --data_cfg src/lib/cfg/mot17.json --batch_size 24 --lr_step 20 --pos 7 --id_aware
        (To test) python ada_track.py --load_model trained_weights.pth --conf_thres 0.4 --test_mot17 True
        Other usage is identitcal to FairMOT
##### Coming soon:

The pre-trained weights.  

The codes and of pre-trained weights of our improved CenterNet detector inside AdaMOT.

![Alt text](https://raw.githubusercontent.com/DemoGit4LIANG/AdaMOT/main/Screenshots/main2.png)