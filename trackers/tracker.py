from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        # interpolate ball positions
        df_ball_positions = df_ball_positions.interpolate()
        # if first few frames are not detected we find nearest detecting and backfill it
        df_ball_positions = df_ball_positions.bfill()
        # converting back to original format
        ball_positions = [{1: {'bbox' : x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

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
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - (rectangle_width//2)
        x2_rect = x_center + (rectangle_width//2)
        y1_rect = (y2 - rectangle_height//2)
        y2_rect = (y2 + rectangle_height//2)
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED
                          )
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, 
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2
                        )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10, y-20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0),2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # count times each team has ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Possession: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Possession: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self,video_frames, tracks, team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'],(0,255,255))

            # draw ball
            for track_id, ball in ball_dict.items():
                frane = self.draw_triangle(frame, ball['bbox'], (0,255,0))

            # draw team possession percentage
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames

