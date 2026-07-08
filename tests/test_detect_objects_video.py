import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Mock out external libraries that might not be installed
sys.modules["openfilter"] = MagicMock()
sys.modules["openfilter.filter_runtime"] = MagicMock()
sys.modules["openfilter.filter_runtime.filter"] = MagicMock()
sys.modules["openfilter.filter_runtime.filters.video_in"] = MagicMock()
sys.modules["filter_sam3_detector"] = MagicMock()

# Add root directory to sys.path to allow importing from examples
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from examples.detect_objects_video import main  # noqa: E402


class TestDetectObjectsVideo(unittest.TestCase):
    def setUp(self):
        # Automatically mock file-system checks to return True for video and exemplar inputs
        self.is_file_patcher = patch(
            "examples.detect_objects_video.Path.is_file", return_value=True
        )
        self.exists_patcher = patch(
            "examples.detect_objects_video.Path.exists", return_value=True
        )
        self.mock_is_file = self.is_file_patcher.start()
        self.mock_exists = self.exists_patcher.start()

    def tearDown(self):
        self.is_file_patcher.stop()
        self.exists_patcher.stop()

    @patch("examples.detect_objects_video.Filter.run_multi")
    @patch("examples.detect_objects_video.Path.mkdir")
    def test_single_prompt(self, mock_mkdir, mock_run_multi):
        """Test with a single prompt value."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--prompt",
            "cup",
            "--output-dir",
            "./results",
            "--confidence",
            "0.2",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that Filter.run_multi was called
        mock_run_multi.assert_called_once()
        filters = mock_run_multi.call_args[0][0]

        # We expect a VideoIn filter and a FilterSAM3Detector filter
        self.assertEqual(len(filters), 2)

        video_in_class, video_in_config = filters[0]
        self.assertEqual(video_in_config["outputs"], ["tcp://127.0.0.1:5555"])

        detector_class, detector_config = filters[1]
        self.assertEqual(detector_config["text_prompts"], ["cup"])
        self.assertEqual(detector_config["confidence_threshold"], 0.2)
        self.assertEqual(
            detector_config["output_path"], str(Path("./results/detections.jsonl"))
        )

    @patch("examples.detect_objects_video.Filter.run_multi")
    @patch("examples.detect_objects_video.Path.mkdir")
    def test_multiple_prompts_extend(self, mock_mkdir, mock_run_multi):
        """Test with multiple prompt values using space separator."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--prompt",
            "cup",
            "bowl",
            "--output-dir",
            "./results",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_multi.assert_called_once()
        filters = mock_run_multi.call_args[0][0]
        detector_class, detector_config = filters[1]
        self.assertEqual(detector_config["text_prompts"], ["cup", "bowl"])

    @patch("examples.detect_objects_video.Filter.run_multi")
    @patch("examples.detect_objects_video.Path.mkdir")
    def test_multiple_prompts_repeated(self, mock_mkdir, mock_run_multi):
        """Test with multiple prompt values using repeated --prompt flags."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--prompt",
            "cup",
            "--prompt",
            "bowl",
            "--output-dir",
            "./results",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_multi.assert_called_once()
        filters = mock_run_multi.call_args[0][0]
        detector_class, detector_config = filters[1]
        self.assertEqual(detector_config["text_prompts"], ["cup", "bowl"])

    @patch("examples.detect_objects_video.Filter.run_multi")
    @patch("examples.detect_objects_video.Path.mkdir")
    def test_duplicate_prompts_deduplicated(self, mock_mkdir, mock_run_multi):
        """Test that duplicate prompt values are successfully deduplicated."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--prompt",
            "cup",
            "bowl",
            "cup",
            "--output-dir",
            "./results",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_multi.assert_called_once()
        filters = mock_run_multi.call_args[0][0]
        detector_class, detector_config = filters[1]
        self.assertEqual(detector_config["text_prompts"], ["cup", "bowl"])

    @patch("examples.detect_objects_video.Filter.run_multi")
    @patch("examples.detect_objects_video.Path.mkdir")
    def test_exemplars_only(self, mock_mkdir, mock_run_multi):
        """Test configured with exemplars instead of prompts."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--exemplars",
            "./cup_examples/",
            "--output-dir",
            "./results",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_multi.assert_called_once()
        filters = mock_run_multi.call_args[0][0]
        detector_class, detector_config = filters[1]
        self.assertNotIn("text_prompts", detector_config)
        self.assertEqual(detector_config["exemplars_path"], "./cup_examples/")

    @patch("argparse.ArgumentParser.error", side_effect=SystemExit)
    def test_no_prompts_or_exemplars(self, mock_error):
        """Ensure script fails when neither --prompt nor --exemplars is provided."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--output-dir",
            "./results",
        ]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                main()

        mock_error.assert_called_once_with(
            "one of the arguments --prompt --exemplars is required"
        )

    @patch("examples.detect_objects_video.Filter.run_multi")
    @patch("examples.detect_objects_video.Path.mkdir")
    def test_visualize_flag(self, mock_mkdir, mock_run_multi):
        """Test with --visualize flag sets save_annotated_frames to True and annotated_frames_output_dir."""
        test_args = [
            "detect_objects_video.py",
            "--video",
            "input.mp4",
            "--prompt",
            "cup",
            "--visualize",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_multi.assert_called_once()
        filters = mock_run_multi.call_args[0][0]
        detector_class, detector_config = filters[1]
        self.assertTrue(detector_config["save_annotated_frames"])
        self.assertEqual(
            detector_config["annotated_frames_output_dir"], str(Path("./output/frames"))
        )


if __name__ == "__main__":
    unittest.main()
