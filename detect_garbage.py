#!/usr/bin/env python3
"""
Garbage Detection System for FPV Drone Video
Uses YOLO11n with ByteTrack for real-time object detection and tracking
"""

import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import argparse
from pathlib import Path


def detect_garbage_in_video(
    video_path: str,
    output_path: str,
    model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    device: str = None
):
    """
    Process video to detect and track garbage items.

    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model_name: YOLO model to use (default: yolo11n.pt)
        conf_threshold: Confidence threshold for detections (default: 0.25)
        device: Device to run inference on (default: auto-detect GPU)
    """

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == '0':
        device = 'cuda'

    print(f"🚀 Initializing Garbage Detection System")
    print(f"📹 Video: {video_path}")
    print(f"🎯 Model: {model_name}")
    print(f"💻 Device: {'GPU' if 'cuda' in device else 'CPU'}")
    print(f"🎚️  Confidence threshold: {conf_threshold}")
    print("-" * 60)

    # Load YOLO model (auto-downloads on first use)
    print("Loading YOLO model...")
    model = YOLO(model_name)
    model.to(device)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # COCO class IDs for garbage detection
    # Class 39: bottle (water bottles, tea bottles, etc.)
    # Class 73: book (can detect some paper-like flat objects)
    TARGET_CLASSES = [39, 73]
    CLASS_NAMES = {39: 'Bottle', 73: 'Paper'}

    # Tracking statistics
    unique_objects = set()  # Set of unique track IDs
    frame_count = 0
    detection_count = 0

    print("🎬 Processing video...")
    print("-" * 60)

    # Process video with tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO with tracking (ByteTrack)
        # persist=True maintains track IDs across frames
        results = model.track(
            frame,
            conf=conf_threshold,
            classes=TARGET_CLASSES,
            device=device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Process detections
        current_frame_detections = 0
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # Track unique objects
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
                unique_objects.update(track_ids)
                current_frame_detections = len(boxes)
                detection_count += current_frame_detections

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Add running counter overlay
        total_unique = len(unique_objects)
        counter_text = f"Total Garbage Detected: {total_unique}"
        frame_info = f"Frame: {frame_count}/{total_frames}"

        # Draw semi-transparent background for text
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 90), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

        # Draw counter text
        cv2.putText(
            annotated_frame,
            counter_text,
            (20, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame,
            frame_info,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Write frame
        out.write(annotated_frame)

        # Progress update every 30 frames (~1 second)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Unique objects: {total_unique}")

    # Cleanup
    cap.release()
    out.release()

    # Final statistics
    print("-" * 60)
    print("✅ Processing Complete!")
    print("-" * 60)
    print(f"📊 Statistics:")
    print(f"   • Total frames processed: {frame_count}")
    print(f"   • Unique garbage items detected: {len(unique_objects)}")
    print(f"   • Total detections (all frames): {detection_count}")
    print(f"   • Average detections per frame: {detection_count/frame_count if frame_count > 0 else 0:.2f}")
    print(f"   • Output video saved to: {output_path}")
    print("-" * 60)

    return {
        'total_frames': frame_count,
        'unique_objects': len(unique_objects),
        'total_detections': detection_count,
        'output_path': output_path
    }


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Detect and track garbage in drone video footage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output video file (default: input_detected.mp4)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='yolo11n.pt',
        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolov8n.pt', 'yolov8s.pt'],
        help='YOLO model to use'
    )
    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (0.0-1.0)'
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default=None,
        help='Device to use (0 for GPU, cpu for CPU, default: auto-detect)'
    )

    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.video_path)
        args.output = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")

    # Run detection
    try:
        detect_garbage_in_video(
            video_path=args.video_path,
            output_path=args.output,
            model_name=args.model,
            conf_threshold=args.confidence,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == '__main__':
    main()
