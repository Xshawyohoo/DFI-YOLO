import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    #创新
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy24.yaml') #yolo11-P2+Dynamiconv+DAT+AIFI train 129 涨但不够多、
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy23.yaml') #yolo11-P2+SPDConv+SCSA+AIFI train 128 最高！
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy22.yaml') #dysample+RepGFPN-1 train126 0.6580
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy21.yaml') #Dynamiconv+LSKA+AIFI train125 0.6646
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy20.yaml') #dysample+FocalModulation train123？跑不出来东西
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy19.yaml') #dysample+CGA train122
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy18.yaml') #dysample+EMA train121
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy17.yaml') #dysample+MLCA train120
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-HetConv-1.yaml') #train119 降pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy16.yaml') #Foggy7基础上改进
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy15.yaml') #ScConv+SCSA+AIFI pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy14.yaml') #SPDConv+SCSA+AIFI+AFPN train95，降
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy13.yaml') #SPDConv+SCSA+AIFI+AFPN 降
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy12.yaml') #SPDConv+LSKA+AIFI train87 pass涨幅不大
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy11.yaml') #SPDConv+ECA+AIFI train86涨，map95不如foggy7，其他高些
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy10.yaml') #SPDConv+SCSA+SDI train82小涨，不多
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy9.yaml') #SPDConv+CGA+AIFI train80
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy8.yaml') #SPDConv+CAA+AIFI train79
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy7.yaml') #SPDConv+SCSA+AIFI train78 涨0.03多啊啊啊啊
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy6.yaml') #SPDConv+EMA+AIFI train76
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy5.yaml') #SPDConv+MSDA+AIFI train70 涨0.02！
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy4.yaml') #SPDConv+DAT+AIFI+c3k2Dynamiconv train66 涨！
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy3.yaml') #SPDConv+DAT+AIFI 涨0.017
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy2.yaml') #train49效果最好，Dynamiconv+DAT+AIFI
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-DynamicConv.yaml') #train49效果最好
    #检测头
    # model = YOLO('ultralytics/cfg/models/11/yolo11-ASFFHead.yaml') #train69 pass降
    #特殊场景检测
    # model = YOLO('ultralytics/cfg/models/11/yolo11-CPAEnhancer.yaml') #内存不够...
    # model = YOLO('ultralytics/cfg/models/11/yolo11-IAT.yaml') #
    #细节涨点
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Dysample.yaml') #train118 啊啊啊啊啊啊采纳！
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-iAFF.yaml') #train64 不涨 pass
    #neck
    # model = YOLO('ultralytics/cfg/models/11/yolo11-FocalModulation.yaml') #train47 大踩雷
    # model = YOLO('ultralytics/cfg/models/11/yolo11-AIFI.yaml') train46涨，采纳
    # model = YOLO('ultralytics/cfg/models/11/yolo11-HSFPN-1.yaml') #train45 降，pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-ASFYOLO-1.yaml') #train44，涨，但计算量大
    # model = YOLO('ultralytics/cfg/models/11/yolo11-RepGFPN-1.yaml') train37 涨，采纳，可以再调一下参数
    # model = YOLO('ultralytics/cfg/models/11/yolo11-SlimNeck-1.yaml') train36 p涨
    # model = YOLO('ultralytics/cfg/models/11/yolo11-CCFM.yaml') train35 
    # model = YOLO('ultralytics/cfg/models/11/yolo11-BiFPN.yaml') train34 涨0.0002的map50-95，p值涨
    #注意力机制
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-SENetV2.yaml') #train124
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-SENetV2.yaml')
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-DAT.yaml') train29 可考虑，待定
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-CAA.yaml') train28
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-iRMB.yaml') train27降pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-SENetV1.yaml') train26 持平待定p涨
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-TripleAtt.yaml') train25 持平待定p涨
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-CGA.yaml') train24 pass持平
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-MSDA.yaml') train21 pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-iEMA.yaml') train20 out
    # model = YOLO('ultralytics/cfg/models/11/yolo11-SCSA.yaml') train18 out...
    # model = YOLO('ultralytics/cfg/models/11/yolo11-MLCA.yaml') train15 out
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C2PSA-LSKA.yaml') train14 持平
    #卷积
    # model = YOLO('ultralytics/cfg/models/11/yolo11-AKConv.yaml') #
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-AKConv.yaml') #
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-ScConv.yaml') #
    # model = YOLO('ultralytics/cfg/models/11/yolo11-DynamicConv.yaml') #train65 降 pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-SPDConv.yaml') #train58 涨，半采纳
    # model = YOLO('ultralytics/cfg/models/11/yolo11-LAE.yaml') #train57 out
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-LDConv-1.yaml') #报错
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-MAB1.yaml') train12 out
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-MSCB1.yaml') pass train10
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-WTConv.yaml')  # train9、train12 pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-OREPA.yaml') HazeData/train8 pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-ContextGuided.yaml')  pass HazeData/train6
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-RFAConv.yaml') # pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-GhostModule.yaml')  exp20 pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-Foggy.yaml') #TwoBackbone+C3k2-DynamicConv  HazeData/train3可采纳0.5160
    # model = YOLO('ultralytics/cfg/models/11/yolo11-SAConv.yaml')  exp18 无增强不采纳
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-RepVGG.yaml') exp14 pass
    # model = YOLO('ultralytics/cfg/models/11/yolo11-TwoBackbone.yaml') exp13 可采纳0.5082
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-DynamicConv.yaml') #HazeData/train39 可采纳0.5058
    # model = YOLO('ultralytics/cfg/models/11/yolo11-ODConv.yaml')  exp10
    # model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-ODConv.yaml')  exp9 可以理会
    # model = YOLO('ultralytics/cfg/models/11/yolo11-FasterNet.yaml') exp8可以考虑结合
    # model = YOLO('ultralytics/cfg/models/11/yolo11-HGNetV2-l.yaml') exp7
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml') # HazeData/train41、train132低
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'datasets/Haze_data/data.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                # cache=False,
                imgsz=640,
                epochs=130,
                lr0=0.001,              # 初始学习率
                # lr0=0.01,              # 初始学习率
                iou=0.5,
                weight_decay=0.0005,   # 权重衰减防止过拟合
                cos_lr=True,
                warmup_bias_lr=0.01,  # 通常设置为 0.01 到 0.1 之间
                # single_cls=False,  # 是否是单类别检测
                batch=64,
                # close_mosaic=0,
                # workers=0,
                momentum=0.9,           # 控制梯度更新的动量项，一般用于加速收敛，减少振荡
                patience=50,            # 如果在10个epochs内验证集mAP没有提升，则停止训练
                device='0',
                optimizer='Adam', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                # amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/HazeData',
                name='train',
                )