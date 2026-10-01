import json
from typing import Final

import httpx

from litellm.llms.elevenlabs.audio_transcription.transformation import ElevenLabsAudioTranscriptionConfig


def test_native_diarization_and_segments_survive_proxy_serialization() -> None:
    payload: Final = {
        "language_code": "en",
        "language_probability": 0.97,
        "text": "Hello there.",
        "words": [
            {"text": "Hello", "type": "word", "start": 0, "end": 0.4, "speaker_id": "speaker_0", "logprob": -0.1},
            {"text": " ", "type": "spacing", "start": 0.4, "end": 0.4},
            {"text": "there.", "type": "word", "start": 0.5, "end": 1, "speaker_id": "speaker_1"},
        ],
        "additional_formats": [{"requested_format": "segmented_json", "content": '{"segments":[]}'}],
    }
    config: Final = ElevenLabsAudioTranscriptionConfig()
    response: Final = config.transform_audio_transcription_response(httpx.Response(200, json=payload))
    response._hidden_params["api_key"] = "private-provider-key"
    serialized: Final = json.loads(response.model_dump_json())

    assert serialized["text"] == payload["text"]
    assert serialized["words"] == [
        {"word": "Hello", "start": 0, "end": 0.4},
        {"word": "there.", "start": 0.5, "end": 1},
    ]
    assert serialized["provider_specific_fields"]["elevenlabs"] == payload
