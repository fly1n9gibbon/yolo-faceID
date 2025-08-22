import cv2
import os
import glob
import argparse
import sys
import csv
from ultralytics import YOLO
from datetime import datetime
import yolo_config 

class FaceRecognizer:
    def __init__(self, detection_path, recognition_path):
        print("Initializing YOLO FaceID...")
        self.detection_model = YOLO(detection_path)
        self.recognition_model = YOLO(recognition_path)
        self.class_names = self.recognition_model.names
        print("✅ YOLO FaceID ready.")

    def process_frame(self, frame, threshold):
        primary_name = "unknown"
        primary_confidence = 0.0
        first_detection = True
        face_count = 0
        detection_results = self.detection_model(frame, verbose=False)
        if detection_results and detection_results[0].boxes:
            face_count = len(detection_results[0].boxes)
        for result in detection_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 < x2 and y1 < y2:
                    face_crop = frame[y1:y2, x1:x2]
                    rec_results = self.recognition_model(face_crop, verbose=False)
                    pred_index = rec_results[0].probs.top1
                    confidence = rec_results[0].probs.top1conf.item()
                    label = "Unknown"
                    pred_name = "unknown"
                    if confidence > threshold:
                        pred_name = self.class_names[pred_index]
                        label = f"{pred_name}: {confidence:.2f}"
                        if first_detection:
                            primary_name = pred_name
                            primary_confidence = confidence
                            first_detection = False
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame, primary_name, primary_confidence, face_count

# ==============================================================================
# Function to process a folder of images
# ==============================================================================
def process_folder(recognizer, input_path, output_path, threshold, log_file, model_name):
    print(f"Processing folder: {input_path}")
    print(f"Using model: {model_name}")
    print(f"Results will be saved in: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    print(f"Analysis log will be saved to: {log_file}")
    
    current_date = datetime.now().strftime(yolo_config.FILENAME_DATE_FORMAT)
    name_counters = {}
    multiple_faces_counter = 0

    image_paths = [p for ext in yolo_config.SUPPORTED_IMAGE_EXTENSIONS for p in glob.glob(os.path.join(input_path, ext))]
    if not image_paths:
        print(f"❌ Error: No images found in '{input_path}'.")
        return

    log_header = ['timestamp', 'face_recognition_model_name', 'original_filename', 'status', 'recognized_person', 'confidence', 'face_count', 'threshold_used', 'output_path']
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        log_writer = csv.writer(f)
        log_writer.writerow(log_header)
        print(f"Found {len(image_paths)} images to process...")
        for image_path in image_paths:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"⚠️ Warning: Could not read image {image_path}. Skipping.")
                continue
            
            annotated_frame, primary_name, primary_confidence, face_count = recognizer.process_frame(frame, threshold)
            original_filename = os.path.basename(image_path)
            _, extension = os.path.splitext(original_filename)
            full_save_path = ""
            status = ""

            if face_count > 1:
                multiple_faces_counter += 1
                target_folder = os.path.join(output_path, "multiple_faces")
                status = "Multiple Faces"
                os.makedirs(target_folder, exist_ok=True)
                new_filename = f"multiple-{multiple_faces_counter}({current_date}){extension}"
                full_save_path = os.path.join(target_folder, new_filename)
            else:
                current_count = name_counters.get(primary_name, 0) + 1
                name_counters[primary_name] = current_count
                target_folder = os.path.join(output_path, primary_name)
                status = "Recognized" if primary_name != "unknown" else "Unknown"
                os.makedirs(target_folder, exist_ok=True)
                confidence_percent = int(primary_confidence * 100)
                new_filename = f"{primary_name}-{current_count}({confidence_percent}%)({current_date}){extension}"
                full_save_path = os.path.join(target_folder, new_filename)
            
            cv2.imwrite(full_save_path, annotated_frame)
            print(f"  -> Processed '{original_filename}'. Saved to '{full_save_path}'")
            log_row = [
                datetime.now().isoformat(), model_name, original_filename, status,
                primary_name, f"{primary_confidence:.4f}", face_count,
                threshold, full_save_path
            ]
            log_writer.writerow(log_row)
    print("\n✅ Folder processing complete.")

# ==============================================================================
# Function to process a live camera feed
# ==============================================================================
def process_live_feed(recognizer, video_source, threshold):
    print(f"Starting live feed from camera index: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{video_source}'.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame from camera. Exiting.")
            break
        annotated_frame, _, _, _ = recognizer.process_frame(frame, threshold)
        cv2.imshow("Live Face Recognition (Press 'q' to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

# ==============================================================================
# MAIN SCRIPT LOGIC
# ==============================================================================
def main(args):
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Using model paths from yolo_config file
    recognizer = FaceRecognizer(
        detection_path=yolo_config.YOLO_DETECTION_MODEL_PATH, 
        recognition_path=yolo_config.YOLO_RECOGNITION_MODEL_PATH 
    )
    
    if args.source == 'folder':
        if not args.input:
            print("❌ Error: --input argument is required for 'folder' mode.")
            sys.exit(1)
        
        top_level_dir = yolo_config.TOP_LEVEL_RESULTS_FOLDER 
        os.makedirs(top_level_dir, exist_ok=True)
        run_subfolder_name = f"{yolo_config.FACE_RECOGNITION_MODEL_NAME}_{run_timestamp}{yolo_config.OUTPUT_FOLDER_SUFFIX}" 
        final_run_path = os.path.join(top_level_dir, run_subfolder_name)
        log_file_path = os.path.join(final_run_path, yolo_config.LOG_FILENAME) 
        
        process_folder(
            recognizer, args.input, final_run_path, args.threshold,
            log_file_path, yolo_config.FACE_RECOGNITION_MODEL_NAME 
        )
        
    elif args.source == 'live':
        process_live_feed(recognizer, args.video_source, args.threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recognize faces from a folder or live feed using YOLO.")
    parser.add_argument("-s", "--source", required=True, choices=['folder', 'live'], help="The source of media: a 'folder' of images or a 'live' camera feed.")
    parser.add_argument("-i", "--input", help="Path to the input folder of photos (REQUIRED for 'folder' source).")
    
    parser.add_argument("-vs", "--video-source", type=int, default=yolo_config.VIDEO_SOURCE, help=f"Camera index for 'live' source. Default: {yolo_config.VIDEO_SOURCE}")
    parser.add_argument("-t", "--threshold", type=float, default=yolo_config.YOLO_RECOGNITION_CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold for face recognition. Overrides the config file default ({yolo_config.YOLO_RECOGNITION_CONFIDENCE_THRESHOLD}).")
    
    args = parser.parse_args()
    main(args)