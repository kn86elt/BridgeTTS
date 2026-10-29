import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import bridge_server_api as api


class TtsSettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.filename = Path(self.directory.name) / "tts_config.json"
        self.original = copy.deepcopy(api.DEFAULT_TTS_CONFIG)
        self.original["v3"]["cfg_scale_text"] = 6.5
        self.filename.write_text(json.dumps(self.original), encoding="utf-8")
        for name, value in (("TTS_CONFIG_FILE", str(self.filename)), ("TTS_CONFIG", copy.deepcopy(self.original))):
            patcher = patch.object(api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(api.app)

    def test_ui_settings_persist_and_preserve_unedited_parameters(self):
        response = self.client.post("/tts_config", json={
            "v4": {"checkpoint_v4": "Aratako/Irodori-TTS-v4.1-Small-MF", "num_steps": 4,
                   "t_schedule_mode": "sway", "sway_coeff": -0.5},
            "v3": {"num_steps": 20, "t_schedule_mode": "sway", "sway_coeff": -1},
        })
        self.assertEqual(response.status_code, 200)
        saved = self.client.get("/tts_config").json()
        self.assertEqual(saved["v4"]["num_steps"], 4)
        self.assertEqual(saved["v4"]["checkpoint_v4"], "Aratako/Irodori-TTS-v4.1-Small-MF")
        self.assertEqual(saved["v3"]["cfg_scale_text"], 6.5)
        self.assertEqual(saved["v2"], self.original["v2"])
        self.assertEqual(api.TTS_CONFIG["v4"]["sway_coeff"], -0.5)

    def test_invalid_settings_are_rejected_without_partial_saves(self):
        for invalid in ({"num_steps": 0}, {"num_steps": 1.5}, {"num_steps": True},
                        {"sway_coeff": 2}, {"t_schedule_mode": "unknown"}):
            with self.subTest(invalid=invalid):
                response = self.client.post("/tts_config", json={"v2": {"num_steps": 10}, "v4": invalid})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(json.loads(self.filename.read_text(encoding="utf-8")), self.original)
                self.assertEqual(api.TTS_CONFIG, self.original)

    def test_v4_auto_steps_and_current_api_header(self):
        self.assertEqual(self.client.post("/tts_config", json={"version": "v4", "num_steps": ""}).status_code, 200)
        self.assertEqual(self.client.get("/tts_config?version=v4").json()["num_steps"], "")
        with patch.object(api, "tts_client", object()), patch.object(api, "tts_api_version", "v4"):
            self.assertEqual(self.client.get("/tts_config").headers["X-TTS-API-Version"], "v4")

    def test_user_model_list_is_persisted_and_validated(self):
        presets = [{"name": "好みのモデル", "checkpoint": "example/model"}]
        self.assertEqual(self.client.post("/tts_config", json={"v4": {"model_presets": presets}}).status_code, 200)
        self.assertEqual(self.client.get("/tts_config?version=v4").json()["model_presets"], presets)
        for invalid in ({"name": ""}, [presets[0], presets[0]], [{"name": "空", "checkpoint": " "}]):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.client.post("/tts_config", json={"v4": {"model_presets": invalid}}).status_code, 422)
                self.assertEqual(self.client.get("/tts_config?version=v4").json()["model_presets"], presets)
