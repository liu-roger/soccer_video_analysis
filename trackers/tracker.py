from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            break
        return detections


    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            
            class_names = detection.names
            class_names_inverse = {v:k for k,v in class_names.items()} 
            print(class_names)

            # Converting to supervision formatting
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalie to player
            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_index] = class_names_inverse['player']

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            print(detection_with_tracks)

