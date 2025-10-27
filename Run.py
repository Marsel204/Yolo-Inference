from ObjectDetectionModule import ObjectDetection as ODModule


odm = ODModule(model_path='best.pt', conf_threshold=0.25)
odm.camera_stream_detection(camera_index=0)
