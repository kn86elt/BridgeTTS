import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import bridge_server_api as api


class ReferenceAudioTests(unittest.TestCase):
    def generate(self, schema, *, embedding=False, version="v3", config=None):
        file_data = {"path": "reference.wav", "meta": {"_type": "gradio.FileData"}}
        client = Mock()
        client._info = {"named_endpoints": {"/_run_generation": {"parameters": [
            {"parameter_name": "uploaded_audio", "type": schema},
            *[{"parameter_name": name} for name in ("checkpoint", "num_steps", "num_candidates", "seed_raw", "seconds_raw", "duration_scale")],
        ]}}}
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "reference.wav"
            speaker = Path(directory) / "voice.speaker.safetensors"
            output = Path(directory) / "output.wav"
            audio.write_bytes(b"reference")
            speaker.write_bytes(b"embedding")
            output.write_bytes(b"generated audio")
            client.predict.return_value = [str(output)]
            with patch.object(api, "get_tts_client", return_value=client), \
                 patch.object(api, "load_settings", return_value={"prefer_speaker_embedding": embedding}), \
                 patch.object(api, "tts_supports_embedding", True), \
                 patch.object(api, "tts_api_version", version), \
                 patch.object(api, "TTS_CONFIG", {version: config} if config is not None else api.TTS_CONFIG), \
                 patch.object(api, "_handle_file_with_retry", return_value=file_data):
                encoded = api.generate_and_encode_tts("テスト", str(audio), str(speaker) if embedding else None)
            self.assertEqual(base64.b64decode(encoded), b"generated audio")
        return client.predict.call_args.kwargs, file_data

    def test_multiple_file_api_receives_list(self):
        kwargs, file_data = self.generate({"type": "array", "items": {"type": "object"}})
        self.assertEqual(kwargs["uploaded_audio"], [file_data])

    def test_legacy_single_file_api_keeps_dict(self):
        kwargs, file_data = self.generate({"type": "object"})
        self.assertEqual(kwargs["uploaded_audio"], file_data)

    def test_embedding_stays_single_and_reference_is_empty(self):
        kwargs, file_data = self.generate({"type": "array"}, embedding=True)
        self.assertEqual(kwargs["uploaded_audio"], [])
        self.assertEqual(kwargs["uploaded_speaker_embedding"], file_data)

    def test_nullable_and_referenced_array_schemas(self):
        for schema in (
            {"anyOf": [{"type": "null"}, {"type": "array"}]},
            {"$ref": "#/$defs/ListFiles", "$defs": {"ListFiles": {"type": "array"}}},
        ):
            with self.subTest(schema=schema):
                kwargs, file_data = self.generate(schema)
                self.assertEqual(kwargs["uploaded_audio"], [file_data])

    def test_api_profile_detection_keeps_legacy_and_modern_separate(self):
        for parameters, expected in (
            ([], "v2"),
            ([{"parameter_name": "t_schedule_mode"}], "v3"),
            ([{"parameter_name": "t_schedule_mode"}, {"parameter_name": "uploaded_audio", "type": {"type": "array"}}], "v4"),
        ):
            client = Mock()
            client._info = {"named_endpoints": {"/_run_generation": {"parameters": parameters}}}
            self.assertEqual(api.detect_tts_version(client), expected)

    def test_v4_keeps_twenty_steps_and_does_not_override_server_checkpoint(self):
        kwargs, _ = self.generate({"type": "array"}, version="v4", config=api._DEFAULT_TTS_V4)
        self.assertEqual(kwargs["num_steps"], "20")
        self.assertNotIn("checkpoint", kwargs)

    def test_meanflow_checkpoint_and_four_or_automatic_steps_are_sent(self):
        for steps in (4, ""):
            with self.subTest(steps=steps):
                config = {**api._DEFAULT_TTS_V4, "checkpoint_v4": "Aratako/Irodori-TTS-v4.1-Small-MF", "num_steps": steps,
                          "unsupported_option": 100}
                kwargs, _ = self.generate({"type": "array"}, version="v4", config=config)
                self.assertEqual(kwargs["checkpoint"], "Aratako/Irodori-TTS-v4.1-Small-MF")
                self.assertEqual(kwargs["num_steps"], str(steps))
                self.assertNotIn("unsupported_option", kwargs)

    def test_loading_v4_profile_preserves_legacy_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "tts_config.json"
            filename.write_text(json.dumps({"v2": {"num_steps": 17}, "v3": {"num_steps": 20},
                                            "v4": {"checkpoint_v4": "Aratako/Irodori-TTS-v4.1-Small-MF", "num_steps": 4}}), encoding="utf-8")
            with patch.object(api, "TTS_CONFIG_FILE", str(filename)):
                config = api.load_tts_config()
            self.assertEqual(config["v2"]["num_steps"], 17)
            self.assertEqual(config["v3"]["num_steps"], 20)
            self.assertEqual(config["v4"]["num_steps"], 4)
