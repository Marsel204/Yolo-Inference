from ultralytics import YOLO
import cv2

class ObjectDetectionModule:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

    def detect_objects(self, image):
        results = self.model(image)[0]
        detections = []
        for result in results.boxes:
            if result.conf[0] >= self.conf_threshold:
                bbox = result.xyxy[0].cpu().numpy().tolist()
                conf = result.conf[0].cpu().numpy().item()
                cls = int(result.cls[0].cpu().numpy().item())
                class_name = self.class_names[cls] if cls in self.class_names else str(cls)
                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class': cls,
                    'class_name': class_name
                })
        return detections
    
    def draw_detections(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            label = f'{class_name} {conf:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return image
    
    def camera_stream_detection(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            detections = self.detect_objects(frame)
            frame_with_detections = self.draw_detections(frame, detections)
            
            cv2.imshow('Object Detection', frame_with_detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    odm = ObjectDetectionModule(model_path='best.onnx', conf_threshold=0.25)
    odm.camera_stream_detection(camera_index=0)
