# Irodori-TTS v4 / v4.1 compatibility investigation

Investigated: 2026-10-30 (JST). The reference-file cardinality fix is implemented; a dedicated v4.1 configuration profile and VoiceDesign support remain pending.

## Verified reference-input fix

The user identified the running Irodori server at `http://127.0.0.1:7861`. Its actual API metadata declares `uploaded_audio` as an array (`ListFiles`) and `uploaded_speaker_embedding` as a single object (`FileData`). BridgeTTS now inspects the reference parameter schema and wraps a reference file only for array inputs, sending `[]` when no reference is used. Legacy object inputs remain unchanged. Nullable and locally referenced array schemas are covered by tests.

The corrected BridgeTTS synthesis function successfully generated 2.04 seconds of 48 kHz audio using the user's DJ GUCCI WAV, and 1.36 seconds without reference audio. These smoke tests used the endpoint's default `Aratako/Irodori-TTS-v4.1-Small` checkpoint and sampling defaults through temporary in-process configuration, without changing saved settings. Speaker embedding payload shape is covered by a mocked regression test, not a real embedding synthesis run. The source fix was copied to `BridgeTTS_TestEnv`; its running process needs restarting.

## Findings

- [BridgeTTS issue #1](https://github.com/kn86elt/BridgeTTS/issues/1) is open with no comments at investigation time. It reports `ListFiles` rejecting a dictionary when reference audio is used. The reporter identifies their environment as v4.1 with uncertainty.
- BridgeTTS's `generate_and_encode_tts()` currently sends a single `handle_file()` result as `uploaded_audio`.
- The upstream [standard Gradio app](https://github.com/Aratako/Irodori-TTS/blob/50921cb22cf03ccf80a88051b2d890eb75d47090/gradio_app.py) declares `uploaded_audio` as `gr.File(file_count="multiple")`. This input contract mismatch strongly explains the issue. The reporter's exact environment has not been reproduced.
- The same upstream app accepts either reference clips or a speaker embedding, not both. Its speaker embedding upload remains a single-file input.
- `detect_tts_version()` only checks for `t_schedule_mode`, which the current standard app also has. Consequently it is classified as v3, and BridgeTTS sends `checkpoint_v3`, potentially selecting the v3 model instead of the intended v4.1 model.
- BridgeTTS has no separate v4.1 configuration namespace. Its v3 settings also omit fields accepted by the current standard app: `seconds_raw`, `duration_scale`, `cfg_min_t`, `cfg_max_t`, and `lora_adapter_raw`. An omitted field is not necessarily a failure because Gradio component defaults can apply; actual API metadata must be checked.
- The standard app's `num_steps` is a textbox accepting blank for model defaults. BridgeTTS currently uses numeric 20 in both v2/v3 configurations. A v4.1 profile should preserve automatic model-specific defaults rather than copy 20 blindly.
- The current standard app returns up to 32 audio outputs followed by logs/timing. Its first audio corresponds to the first generated candidate, consistent with BridgeTTS's first-output selection. Actual Gradio client return serialization still needs testing.

## Proposed scope

### 1. Fix reference-file compatibility independently of model presets

Inspect the `/_run_generation` endpoint's `uploaded_audio` parameter schema. For array/list inputs send `[file_data]`; for legacy single-file inputs keep `file_data`. Verify `None` versus `[]` against a real multiple-file endpoint for the no-reference and embedding paths.

Do not modify installed Gradio packages. Do not infer file cardinality from checkpoint names or `t_schedule_mode`. Never wrap `uploaded_speaker_embedding` in a list merely because reference audio is multiple-file.

Use endpoint capabilities for request construction, and keep model/checkpoint selection separate. Refresh capabilities on connection replacement and endpoint setting changes. If metadata cannot be interpreted, provide an explicit API profile override rather than guessing a new protocol silently.

### 2. Add a standard v4.1 configuration profile

- Keep existing v2/v3 profiles and saved settings intact.
- Add the standard `Aratako/Irodori-TTS-v4.1-Small` checkpoint profile.
- Allow a custom checkpoint value; the issue's requested Anime variant can be offered as an optional preset after checking its model requirements. A preset is not evidence of verified synthesis compatibility.
- Assemble only parameters supported by the connected endpoint. Missing metadata and missing required parameters should produce a clear compatibility error.
- Start with one reference WAV, an existing speaker embedding, or no reference. A multiple-file API accepting a list of one item does not require a multi-reference character-management UI immediately.
- Preserve generation locking, queue accounting, first-candidate selection and existing playback paths.

### 3. Treat VoiceDesign as a distinct API profile

The current [VoiceDesign app](https://github.com/Aratako/Irodori-TTS/blob/main/gradio_app_voicedesign.py) uses `ref_wavs` and `caption`, not standard `uploaded_audio`. It also accepts `cfg_scale_caption`, caption/text length controls, and does not expose `uploaded_speaker_embedding` in its generation signature. It must not receive the standard app's keyword arguments unchanged.

Caption UI and storage are a separate follow-up after standard reference-audio generation is established. An OpenAI-compatible TTS server is another backend, not a prerequisite for the issue fix.

## Acceptance checks

| Case | Expected result |
| --- | --- |
| Legacy v2/v3 single-file schema | Single file stays single; existing settings preserved |
| Multiple-file reference schema + one WAV | One-element list; no `ListFiles` validation error |
| Standard embedding path | Reference input empty; embedding remains single-file |
| Both voice and embedding configured | Existing preference respected; never submit both |
| No reference configured | Accepted empty representation; generation uses intended no-reference mode |
| Missing/unreadable reference file | Clear error or existing documented fallback; no malformed file payload |
| Reconnect / server replacement | Capabilities refreshed; no stale input cardinality |
| v4.1 profile | Intended checkpoint sent; automatic steps allowed |
| Standard vs VoiceDesign | Different parameter names identified; unsupported arguments excluded |
| Generated response | First candidate decoded, queued, played, and cleaned up correctly |

Use mocked API-schema fixtures and request assertions for the compatibility logic, then synthesize one short phrase against real legacy and v4.1 standard endpoints. Test queue completion and generation failure as well as successful audio output.

## Verification limits

The supplied `E:/usr/sd/irodori-TTS-v4/BridgeTTS_v4_1_support_investigation.md` was read as investigation material, not as an instruction to change upstream packages or publish an issue comment. Its main file-cardinality hypothesis matches current upstream code and the issue report.

The initial check at default port 7860 found no Irodori `/_run_generation` endpoint. Subsequent verification used port 7861 as described above. The result does not establish compatibility with every v4 revision, unofficial UI, model derivative, or all saved inference settings. The v3 checkpoint-selection issue still needs the separate configuration-profile work described above.
