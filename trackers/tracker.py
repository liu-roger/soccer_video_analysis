from ultralytics import YOLO
import supervision as sv
import pickle
import os

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
        return detections


    def get_object_tracks(self, frames, read_from_stub = False, stub_path=None):
        
        if read_from_stub and (stub_path is not None) and (os.path.exists(stub_path)):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            'players':[],
            'referees':[],
            'ball':[]
        }
        for frame_num, detection in enumerate(detections):
            
            class_names = detection.names
            class_names_inverse = {v:k for k,v in class_names.items()} 

            # Converting to supervision formatting
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalie to player
            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_index] = class_names_inverse['player']

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inverse['player']:
                    tracks['players'][frame_num][track_id] = {'bbox' :bounding_box}
                
                if class_id == class_names_inverse['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox' :bounding_box}
                
            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inverse['ball']:
                    tracks['ball'][frame_num][1] = {'bbox' :bounding_box}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        return(tracks)
    

