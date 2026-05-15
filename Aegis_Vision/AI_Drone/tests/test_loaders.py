"""Tests for data loaders module.

Tests VisDrone and DOTA loader functionality including format conversion,
filtering, and validation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from data.loaders.dota_loader import (
    convert_obb_to_yolo,
    filter_difficult_annotations,
    parse_dota_annotation,
)
from data.loaders.visdrone_loader import (
    convert_to_yolo_format,
    filter_invalid_boxes,
    parse_annotation_file,
    VISDRONE_CLASS_MAP,
)
from utils.bbox_utils import obb_to_aabb


class TestVisDroneLoader:
    """Test cases for VisDrone loader."""
    
    def test_class_mapping_skips_ignored(self):
        """Test that classes 0 and 11 are mapped to None (skipped)."""
        assert VISDRONE_CLASS_MAP[0] is None  # ignored_region
        assert VISDRONE_CLASS_MAP[11] is None  # others
    
    def test_class_mapping_valid(self):
        """Test that valid classes are mapped correctly."""
        assert VISDRONE_CLASS_MAP[1] == 0   # pedestrian
        assert VISDRONE_CLASS_MAP[4] == 2   # car
        assert VISDRONE_CLASS_MAP[9] == 5   # bus
    
    def test_parse_annotation_file(self, tmp_path):
        """Test parsing a VisDrone annotation file."""
        # Create test annotation file
        ann_file = tmp_path / "test.txt"
        ann_content = "100,200,50,80,1,4,0,0\n150,250,60,90,1,5,0,1\n"
        ann_file.write_text(ann_content)
        
        annotations = parse_annotation_file(ann_file)
        
        assert len(annotations) == 2
        assert annotations[0]["bbox_xywh"] == [100, 200, 50, 80]
        assert annotations[0]["class_id"] == 2  # car
        assert annotations[1]["class_id"] == 3  # van
    
    def test_convert_to_yolo_format(self):
        """Test YOLO format conversion."""
        annotation = {
            "bbox_xywh": [100, 200, 50, 80],
            "class_id": 2,
        }
        
        yolo_str = convert_to_yolo_format(annotation, img_w=640, img_h=480)
        
        # Parse result
        parts = [float(x) for x in yolo_str.split()]
        assert len(parts) == 5
        assert parts[0] == 2.0  # class_id
        # cx = (100 + 25) / 640 = 0.1953
        # cy = (200 + 40) / 480 = 0.5
        assert 0 < parts[1] < 1  # cx normalized
        assert 0 < parts[2] < 1  # cy normalized
        assert 0 < parts[3] < 1  # w normalized
        assert 0 < parts[4] < 1  # h normalized
    
    def test_filter_invalid_boxes(self):
        """Test that invalid boxes are filtered out."""
        annotations = [
            {"bbox_xywh": [100, 200, 50, 80], "class_id": 2, "score": 1, "truncation": 0, "occlusion": 0},
            {"bbox_xywh": [0, 0, 0, 0], "class_id": 2, "score": 1, "truncation": 0, "occlusion": 0},  # area = 0
            {"bbox_xywh": [100, 200, 2, 2], "class_id": 2, "score": 1, "truncation": 0, "occlusion": 0},  # area = 4 < 16
            {"bbox_xywh": [100, 200, 50, 80], "class_id": 2, "score": 0, "truncation": 0, "occlusion": 0},  # score = 0
            {"bbox_xywh": [100, 200, 50, 80], "class_id": 2, "score": 1, "truncation": 3, "occlusion": 0},  # truncation > 2
        ]
        
        filtered = filter_invalid_boxes(annotations, img_w=640, img_h=480)
        
        # Only first annotation should remain
        assert len(filtered) == 1
        assert filtered[0]["bbox_xywh"] == [100, 200, 50, 80]
    
    def test_yolo_format_5_values(self):
        """Test that YOLO label format has exactly 5 values, all 0-1."""
        annotation = {
            "bbox_xywh": [320, 240, 100, 100],
            "class_id": 2,
        }
        
        yolo_str = convert_to_yolo_format(annotation, img_w=640, img_h=480)
        parts = yolo_str.split()
        
        assert len(parts) == 5
        
        # All values should be in [0, 1]
        values = [float(x) for x in parts]
        assert all(0 <= v <= 1 for v in values)


class TestDOTALoader:
    """Test cases for DOTA loader."""
    
    def test_parse_dota_annotation(self, tmp_path):
        """Test parsing a DOTA annotation file."""
        ann_file = tmp_path / "test.txt"
        ann_content = """imagesource: GoogleEarth
gsd: 0.5
100 100 200 100 200 200 100 200 car 0
150 150 250 150 250 250 150 250 ship 0
300 300 400 300 400 400 300 400 baseball-diamond 0
"""
        ann_file.write_text(ann_content)
        
        annotations = parse_dota_annotation(ann_file)
        
        # baseball-diamond should be skipped (mapped to None)
        assert len(annotations) == 2
        assert annotations[0]["class_id"] == 2  # car -> merged with VisDrone car
        assert annotations[1]["class_id"] == 7  # ship
    
    def test_obb_to_aabb_conversion(self):
        """Test OBB to AABB conversion."""
        # Square OBB at 45 degrees
        points = [
            (100, 100),
            (200, 100),
            (200, 200),
            (100, 200),
        ]
        
        xmin, ymin, xmax, ymax = obb_to_aabb(points)
        
        assert xmin == 100
        assert ymin == 100
        assert xmax == 200
        assert ymax == 200
    
    def test_convert_obb_to_yolo(self):
        """Test OBB to YOLO format conversion."""
        points = [
            (100, 100),
            (200, 100),
            (200, 200),
            (100, 200),
        ]
        
        yolo_str = convert_obb_to_yolo(points, img_w=640, img_h=480, class_id=6)
        
        parts = [float(x) for x in yolo_str.split()]
        assert len(parts) == 5
        assert parts[0] == 6.0  # class_id
        assert all(0 <= v <= 1 for v in parts[1:])  # All normalized
    
    def test_filter_difficult_annotations(self):
        """Test filtering of difficult annotations."""
        annotations = [
            {"points": [(0, 0), (10, 0), (10, 10), (0, 10)], "class_id": 6, "difficulty": 0},
            {"points": [(0, 0), (10, 0), (10, 10), (0, 10)], "class_id": 6, "difficulty": 1},
            {"points": [(0, 0), (10, 0), (10, 10), (0, 10)], "class_id": 6, "difficulty": 2},  # Filtered
        ]
        
        filtered = filter_difficult_annotations(annotations)
        
        assert len(filtered) == 2
        assert all(a["difficulty"] <= 1 for a in filtered)


# ── Phase 2: Dataset YAML Update Tests ───────────────────────────────────────

class TestDatasetYAMLUpdate:
    """Test cases for dataset.yaml update functionality."""

    def test_visdrone_update_dataset_yaml(self, tmp_path):
        """Test VisDrone _update_dataset_yaml creates correct YAML."""
        from data.loaders.visdrone_loader import _update_dataset_yaml

        processed_path = tmp_path / "processed"
        processed_path.mkdir()

        # Create the config directory and dataset.yaml
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        dataset_yaml = config_dir / "dataset.yaml"

        # Write initial YAML with placeholder path
        initial_content = f"""
train: placeholder/path/train/images
val: placeholder/path/val/images
nc: 12
names:
  0: pedestrian
  1: bicycle
"""
        dataset_yaml.write_text(initial_content)

        # Update the YAML
        _update_dataset_yaml(processed_path, dataset_yaml)

        # Read updated YAML
        updated_content = dataset_yaml.read_text()

        # Verify paths updated
        assert str(processed_path) in updated_content
        assert "placeholder/path" not in updated_content
        assert "train:" in updated_content
        assert "val:" in updated_content

    def test_dota_update_dataset_yaml(self, tmp_path):
        """Test DOTA _update_dataset_yaml creates correct YAML."""
        from data.loaders.dota_loader import _update_dataset_yaml

        processed_path = tmp_path / "processed"
        processed_path.mkdir()

        # Create the config directory and dataset.yaml
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        dataset_yaml = config_dir / "dataset.yaml"

        # Write initial YAML
        initial_content = f"""
train: old/path/train/images
val: old/path/val/images
nc: 12
names:
  0: pedestrian
"""
        dataset_yaml.write_text(initial_content)

        # Update the YAML
        _update_dataset_yaml(processed_path, dataset_yaml)

        # Read updated YAML
        updated_content = dataset_yaml.read_text()

        # Verify paths updated
        assert str(processed_path) in updated_content
        assert "old/path" not in updated_content

    def test_update_dataset_yaml_creates_file_if_missing(self, tmp_path):
        """Test that _update_dataset_yaml creates file if it doesn't exist."""
        from data.loaders.visdrone_loader import _update_dataset_yaml

        processed_path = tmp_path / "processed"
        processed_path.mkdir()

        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        dataset_yaml = config_dir / "dataset.yaml"

        # File doesn't exist
        assert not dataset_yaml.exists()

        # Update should create it
        _update_dataset_yaml(processed_path, dataset_yaml)

        # Verify file created with correct content
        assert dataset_yaml.exists()
        content = dataset_yaml.read_text()
        assert str(processed_path) in content
        assert "nc: 12" in content
        assert "train:" in content
        assert "val:" in content
