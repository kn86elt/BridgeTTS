import base64
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import bridge_server_api as api


class ReferenceAudioTests(unittest.TestCase):
    def generate(self, schema, *, embedding=False, version="v3", config=None, additional=False, failed_reference=None):
        file_data = {"path": "reference.wav", "meta": {"_type": "gradio.FileData"}}
        client = Mock()
        client._info = {"named_endpoints": {"/_run_generation": {"parameters": [
            {"parameter_name": "uploaded_audio", "type": schema},
            *[{"parameter_name": name} for name in ("checkpoint", "num_steps", "num_candidates", "seed_raw", "seconds_raw", "duration_scale")],
        ]}}}
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "reference.wav"
            speaker = Path(directory) / "voice.speaker.safetensors"
            output_dir = Path(directory) / "generated"
            output_dir.mkdir()
            output = output_dir / "output.wav"
            audio.write_bytes(b"reference")
            if additional:
                references = Path(directory)
                for name in ("02.wav", "01.WAV", "ignored.mp3"):
                    (references / name).write_bytes(b"reference")
                (references / "nested.wav").mkdir()
            def upload(path, _label):
                if Path(path).name == failed_reference:
                    return None
                return {**file_data, "path": Path(path).name} if additional else file_data
            speaker.write_bytes(b"embedding")
            output.write_bytes(b"generated audio")
            client.predict.return_value = [str(output)]
            with patch.object(api, "get_tts_client", return_value=client), \
                 patch.object(api, "load_settings", return_value={"prefer_speaker_embedding": embedding}), \
                 patch.object(api, "tts_supports_embedding", True), \
                 patch.object(api, "tts_api_version", version), \
                 patch.object(api, "TTS_CONFIG", {version: config} if config is not None else api.TTS_CONFIG), \
                 patch.object(api, "_handle_file_with_retry", side_effect=upload):
                reference = api._character_voice_reference({
                    "voice_path": str(audio), "prompt": str(Path(directory) / "reference.txt"),
                    "char_format": "subfolder" if additional else "flat",
                })
                encoded = api.generate_and_encode_tts("テスト", reference, str(speaker) if embedding else None)
            self.assertEqual(base64.b64decode(encoded), b"generated audio")
        return client.predict.call_args.kwargs, file_data

    def test_multiple_file_api_receives_list(self):
        kwargs, file_data = self.generate({"type": "array", "items": {"type": "object"}})
        self.assertEqual(kwargs["uploaded_audio"], [file_data])

    def test_multiple_wavs_are_ordered_and_scoped_to_character_folder(self):
        kwargs, _ = self.generate({"type": "array"}, version="v4", additional=True)
        self.assertEqual([item["path"] for item in kwargs["uploaded_audio"]], ["reference.wav", "01.WAV", "02.wav"])

    def test_only_folder_and_zip_characters_capture_multiple_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.wav").write_bytes(b"voice")
            (root / "extra.wav").write_bytes(b"extra")
            for format in ("flat", "subfolder", "zip"):
                with self.subTest(format=format):
                    primary = str(root / "voice.wav")
                    result = api._character_voice_reference({
                        "voice_path": primary, "prompt": str(root / "voice.txt"), "char_format": format,
                    })
                    self.assertEqual(result, primary if format == "flat" else [primary, str(root / "extra.wav")])

    def test_character_discovery_supports_folder_zip_and_unchanged_flat_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "characters"
            root.mkdir()
            (root / "Flat.txt").write_text("character", encoding="utf-8")
            (root / "Flat.wav").write_bytes(b"primary")
            (root / "Other.wav").write_bytes(b"not a Flat reference")
            folder = root / "Folder"
            folder.mkdir()
            for name, data in (("Folder.txt", b"character"), ("Folder.wav", b"primary"), ("extra.wav", b"extra")):
                (folder / name).write_bytes(data)
            with zipfile.ZipFile(root / "Zip.zip", "w") as archive:
                archive.writestr("Zip/Zip.txt", "character")
                archive.writestr("Zip/Zip.wav", b"primary")
                archive.writestr("Zip/extra.wav", b"extra")
            characters = []
            with patch.object(api, "ZIP_CACHE_DIR", str(Path(directory) / "cache")):
                api._scan_char_dir(str(root), characters, set())
            by_name = {character["name"]: character for character in characters}
            self.assertEqual(api._character_voice_reference(by_name["Flat"]), str(root / "Flat.wav"))
            for name in ("Folder", "Zip"):
                character = by_name[name]
                references = api._character_voice_reference(character)
                self.assertEqual([Path(path).name for path in references], [name + ".wav", "extra.wav"])
                self.assertEqual(api._tts_reference_paths(references, multiple=False), [character["voice_path"]])

    def test_legacy_versions_keep_original_reference_with_extra_wavs_present(self):
        for version in ("v2", "v3"):
            with self.subTest(version=version):
                kwargs, file_data = self.generate({"type": "object"}, version=version, additional=True)
                self.assertEqual(kwargs["uploaded_audio"], file_data)

    def test_failed_extra_reference_does_not_discard_other_references(self):
        kwargs, _ = self.generate({"type": "array"}, additional=True, failed_reference="01.WAV")
        self.assertEqual([item["path"] for item in kwargs["uploaded_audio"]], ["reference.wav", "02.wav"])

    def test_embedding_preference_ignores_extra_wavs(self):
        kwargs, _ = self.generate({"type": "array"}, embedding=True, additional=True)
        self.assertEqual(kwargs["uploaded_audio"], [])
        self.assertEqual(kwargs["uploaded_speaker_embedding"]["path"], "voice.speaker.safetensors")

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
