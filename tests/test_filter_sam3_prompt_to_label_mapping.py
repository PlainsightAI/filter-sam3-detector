import unittest

from filter_sam3_detector.filter import FilterSAM3Detector


def make_base_config(**overrides):
    config = {
        "name": "sam3_test",
        "type": "filter_sam3_detector",
    }
    config.update(overrides)
    return config

def make_detector():
    return FilterSAM3Detector(make_base_config())


class TestFilterSAM3PromptToLabelMapping(unittest.TestCase):

    def setUp(self):
        self.detector = make_detector()

    def test_text_prompts_none(self):
        config = self.detector.normalize_config(
            make_base_config(text_prompts=None)
        )

        self.assertIsNone(config["text_prompts"])
        self.assertEqual(config["prompt_label_map"], {})

    def test_unmapped_prompts(self):
        config = self.detector.normalize_config(
            make_base_config(text_prompts="car###truck###dog")
        )

        self.assertEqual(config["text_prompts"], ["car", "truck", "dog"])
        self.assertEqual(config["prompt_label_map"], {
            "car": "car",
            "truck": "truck",
            "dog": "dog",
        })

    def test_class_prompt_mapping(self):
        config = self.detector.normalize_config(
            make_base_config(text_prompts="vehicle|||car###vehicle|||truck###animal|||dog")
        )

        self.assertEqual(config["text_prompts"], ["car", "truck", "dog"])
        self.assertEqual(config["prompt_label_map"], {
            "car": "vehicle",
            "truck": "vehicle",
            "dog": "animal",
        })

    def test_mixed_prompts(self):
        config = self.detector.normalize_config(
            make_base_config(text_prompts="vehicle|||car###truck###animal|||dog")
        )

        self.assertEqual(config["text_prompts"], ["car", "truck", "dog"])
        self.assertEqual(config["prompt_label_map"], {
            "car": "vehicle",
            "truck": "truck",
            "dog": "animal",
        })

    def test_whitespace_handling(self):
        config = self.detector.normalize_config(
            make_base_config(text_prompts="  vehicle|||car  ###  animal|||dog ### truck  ")
        )

        self.assertEqual(config["text_prompts"], ["car", "dog", "truck"])
        self.assertEqual(config["prompt_label_map"], {
            "car": "vehicle",
            "dog": "animal",
            "truck": "truck",
        })

    def test_custom_delimiters(self):
        config = self.detector.normalize_config(
            make_base_config(
                text_prompts="vehicle=car|animal=dog|truck",
                class_delimiter="=",
                prompt_delimiter="|",
            )
        )

        self.assertEqual(config["text_prompts"], ["car", "dog", "truck"])
        self.assertEqual(config["prompt_label_map"], {
            "car": "vehicle",
            "dog": "animal",
            "truck": "truck",
        })

    def test_backward_compatibility(self):
        config = self.detector.normalize_config(
            make_base_config(
                text_prompts="car,dog,truck",
                prompt_delimiter=",",
            )
        )
        self.assertEqual(config["text_prompts"], ["car", "dog", "truck"])
        self.assertEqual(config["prompt_label_map"], {
            "car": "car",
            "dog": "dog",
            "truck": "truck",
        })
    def test_empty_delimiters(self):
        with self.assertRaises(ValueError):
            self.detector.normalize_config(
                make_base_config(
                    text_prompts="vehicle|||car###animal|||dog",
                    class_delimiter="",
                    prompt_delimiter="###",
                )
            )

        with self.assertRaises(ValueError):
            self.detector.normalize_config(
                make_base_config(
                    text_prompts="vehicle|||car###animal|||dog",
                    class_delimiter="|||",
                    prompt_delimiter="",
                )
            )

    def test_same_delimiters(self):
        with self.assertRaises(ValueError):
            self.detector.normalize_config(
                make_base_config(
                    text_prompts="vehicle|||car###animal|||dog",
                    class_delimiter="|||",
                    prompt_delimiter="|||"
                )
            )

    def test_duplicate_prompt_mapping(self):
        with self.assertRaises(ValueError):
            self.detector.normalize_config(
                make_base_config(
                    text_prompts="vehicle|||car###automobile|||car"
                )
            )

    def test_list_input_no_mapping(self):
        config = self.detector.normalize_config(
            make_base_config(text_prompts=["car", "truck", "dog"])
        )

        self.assertEqual(config["text_prompts"], ["car", "truck", "dog"])
        self.assertEqual(config["prompt_label_map"], {})

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            self.detector.normalize_config(
                    make_base_config(text_prompts={"car": "vehicle"})
            )


if __name__ == "__main__":
    unittest.main()
