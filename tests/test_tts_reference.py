import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import bridge_server_api as api


class ReferenceAudioTests(unittest.TestCase):
    def generate(self, schema, *, embedding=False):
        file_data = {"path": "reference.wav", "meta": {"_type": "gradio.FileData"}}
        client = Mock()
        client._info = {"named_endpoints": {"/_run_generation": {"parameters": [
            {"parameter_name": "uploaded_audio", "type": schema},
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
