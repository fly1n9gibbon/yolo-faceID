
#=======================================
#        Model & Version Settings 
#========================================
# A descriptive name for the recognition model, used for naming output folders and logging.
FACE_RECOGNITION_MODEL_NAME = 'yolo__recognizer_final_augmentations'

# Path to YOLO Detection Model
YOLO_DETECTION_MODEL_PATH = 'models/face_detector_model.pt'

# Path to YOLO Face Recognition Model (CLS - Classification model)
YOLO_RECOGNITION_MODEL_PATH = 'models/yolo11l_recognizer_logged_final_augmentations/weights/yolo_recognizer_logged_final_augmentations.pt'

#===================================================
#        Folder Processing & Output Settings 
#==================================================
# The name of the main parent folder to store all individual run results.
TOP_LEVEL_RESULTS_FOLDER = "recognition_runs"

# The suffix to append to the each run results folder (the ones inside TOP_LEVEL_RESULTS_FOLDER).
OUTPUT_FOLDER_SUFFIX = "-results"

# The standard name for the log file created inside each results folder.
LOG_FILENAME = "recognition_log.csv"

# The date format used for naming processed image files.
FILENAME_DATE_FORMAT = "%d.%m.%Y"

# List of image file extensions to look for when processing a folder.
SUPPORTED_IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.JPG", "*.JPEG", "*.PNG"]

#=========================================
#        Live Recognition Settings 
#=========================================
# Source for the camera feed (0 for default webcam (can be 1 or 2), path to a video file or livestream).
VIDEO_SOURCE = 0

# Confidence threshold for displaying a recognition result.
# Note: The recognition script's -t argument will override this.
YOLO_RECOGNITION_CONFIDENCE_THRESHOLD = 0.70