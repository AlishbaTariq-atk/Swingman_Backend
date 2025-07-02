"""
Swingman Core Module - Enhanced with YOLO and Pose Integration
"""

# Import existing core components
from .bat_tracker import BatTracker
from .bat_grid import BatGrid
from .impact_detector import ImpactDetector
from .swing_analyzer import SwingAnalyzer
from .bat_visualizer import BatVisualizer
from .heatmap_generator import HeatmapGenerator
from .swing_data_manager import SwingDataManager

# NEW: Import YOLO and Pose integration
from .yolo_detector import YoloDetector
from .pose_analyzer import PoseAnalyzer
from .enhanced_swing_tracker import EnhancedSwingTracker

# Version info
__version__ = "0.3.0"

# Convenience function
def create_enhanced_tracker(custom_bat_model_path=None, enable_pose=True):
    """
    Create an enhanced swing tracker with YOLO and pose analysis
    
    Args:
        custom_bat_model_path: Path to your trained bat detection model
        enable_pose: Whether to enable pose analysis
    
    Returns:
        EnhancedSwingTracker instance
    """
    return EnhancedSwingTracker(custom_bat_model_path, enable_pose)