"""WebRTC Real-Time Video Processor for Browser/Phone Camera Box Scanning.

Wraps CameraBoxTracker into a streamlit-webrtc VideoProcessorBase so that
each incoming video frame is processed in real-time: detection, EPnP pose
estimation, Kalman-filtered tracking, and annotated HUD overlay.

Usage in Streamlit:
    from streamlit_webrtc import webrtc_streamer
    from box_dimensioning.app.webrtc_processor import BoxVideoProcessor

    webrtc_streamer(
        key="box-scanner",
        video_processor_factory=BoxVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
"""

from typing import Dict, Optional
import av
import cv2
import numpy as np

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    # Fallback for environments without streamlit-webrtc
    class VideoProcessorBase:
        pass

from box_dimensioning.pipeline.camera_box_tracker import CameraBoxTracker, TrackingState
from box_dimensioning.dataset.box_geometry import compute_volumetric_weight


class BoxVideoProcessor(VideoProcessorBase):
    """Real-time WebRTC video frame processor for 3D box dimensioning."""

    def __init__(self):
        self.tracker = CameraBoxTracker(
            default_distance_cm=60.0,
            detection_interval=12,
            tracker_type="mil",
            enable_smoothing=True,
        )
        self.latest_result: Dict = {
            "state": TrackingState.INITIALIZING,
            "dimensions_cm": {"length": 0.0, "width": 0.0, "height": 0.0},
            "volumetric_weight_kg": 0.0,
            "fps": 0.0,
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single video frame from the browser camera.

        Called automatically by streamlit-webrtc for each incoming frame.
        Returns the annotated frame with cuboid wireframe and HUD overlay.
        """
        img = frame.to_ndarray(format="bgr24")

        # Run the full tracking pipeline (detect → EPnP → Kalman → overlay)
        result = self.tracker.process_frame(img)

        # Store latest result for external metric display
        self.latest_result = {
            "state": result.get("state", TrackingState.INITIALIZING),
            "dimensions_cm": result.get("dimensions_cm", {"length": 0.0, "width": 0.0, "height": 0.0}),
            "volumetric_weight_kg": result.get("volumetric_weight_kg", 0.0),
            "fps": result.get("fps", 0.0),
            "corners_2d": result.get("corners_2d"),
            "corners_3d": result.get("corners_3d"),
        }

        annotated = result["annotated_frame"]

        # Add scanning guidance overlay
        h, w = annotated.shape[:2]
        state = result.get("state", TrackingState.INITIALIZING)

        if state == TrackingState.INITIALIZING:
            self._draw_scan_guide(annotated, w, h, "Point camera at a box or parcel", (0, 165, 255))
        elif state == TrackingState.TRACKING:
            d = result.get("dimensions_cm", {})
            if d.get("length", 0) > 2.0:
                self._draw_scan_guide(annotated, w, h, "Measuring... Hold steady", (0, 255, 0))
        elif state == TrackingState.LOST:
            self._draw_scan_guide(annotated, w, h, "Box lost — re-center the parcel", (0, 0, 255))

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _draw_scan_guide(self, img: np.ndarray, w: int, h: int, text: str, color: tuple):
        """Draws a scanning guidance bar at the bottom of the frame."""
        bar_h = 44
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.65, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - bar_h // 2 + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.65, color, 2, cv2.LINE_AA)
