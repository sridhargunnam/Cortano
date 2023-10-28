'''
Mapping:
    Gets the timestamp, tag/landmark pose, and spit out the robot's current x,y, theta
    Get the timestamp, object's x,y,z and maximum likelyhood grid based on all the previous observations.   
    voxel grid update based on depth maps. This is done only for the frames where landmark pose was detected.   
'''


from collections import namedtuple

# Define a named tuple with three fields: two integers and a float
InputFrameLandMark = namedtuple('InputFrame', ['frame_id', 'timestamp', 'color', 'depth', 'landmark_pose' ])
InputFrameDetection = namedtuple('Detectionframe', ['frame_id', 'timestamp', 'color', 'depth', 'detections' ])

class Mapping:
    def __init__(self):
        self.landmark_pose = None
        self.voxel_grid = None
    
    def update(self, input_frame):
        if input_frame is InputFrameLandMark:
            self.landmark_pose = input_frame.landmark_pose
        elif input_frame is InputFrameDetection:
            self.voxel_grid = input_frame.voxel_grid
        else:
            print("Invalid input frame type")
            return
        
