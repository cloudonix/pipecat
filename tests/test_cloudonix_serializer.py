#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.cloudonix import CloudonixFrameSerializer


class TestCloudonixFrameSerializer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = CloudonixFrameSerializer(
            stream_sid="test_stream_sid",
            call_sid="test_call_sid",
            domain_id="test_domain_id",
            bearer_token="test_bearer_token",
            session_token="test_session_token",
        )

    async def test_setup_initializes_sample_rate(self):
        """Test that setup method initializes sample rate from StartFrame."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await self.serializer.setup(start_frame)
        self.assertEqual(self.serializer._sample_rate, 16000)

    async def test_setup_uses_pipeline_sample_rate(self):
        """Test that setup method uses pipeline sample rate."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await self.serializer.setup(start_frame)
        self.assertEqual(self.serializer._sample_rate, 16000)

    async def test_serialize_audio_frame(self):
        """Test serialization of audio frames to Cloudonix media messages."""
        # Setup sample rate
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await self.serializer.setup(start_frame)

        # Create test audio frame
        audio_data = b"\x00\x01\x02\x03" * 100  # Simple test data
        audio_frame = AudioRawFrame(
            audio=audio_data,
            num_channels=1,
            sample_rate=16000
        )

        # Mock the audio conversion to return predictable data
        with patch("pipecat.serializers.cloudonix.pcm_to_ulaw") as mock_convert:
            mock_convert.return_value = b"\x7F\x80\x81\x82" * 25  # Mock μ-law data

            result = await self.serializer.serialize(audio_frame)

            # Verify the result is valid JSON
            self.assertIsNotNone(result)
            message = json.loads(result)

            # Verify message structure
            self.assertEqual(message["event"], "media")
            self.assertEqual(message["streamSid"], "test_stream_sid")
            self.assertIn("media", message)
            self.assertIn("payload", message["media"])

            # Verify audio conversion was called
            mock_convert.assert_called_once()

    async def test_serialize_audio_frame_empty_result(self):
        """Test that empty audio data returns None."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await self.serializer.setup(start_frame)

        audio_frame = AudioRawFrame(
            audio=b"",
            num_channels=1,
            sample_rate=16000
        )

        with patch("pipecat.serializers.cloudonix.pcm_to_ulaw") as mock_convert:
            mock_convert.return_value = None
            result = await self.serializer.serialize(audio_frame)
            self.assertIsNone(result)

    async def test_serialize_interruption_frame(self):
        """Test serialization of interruption frames."""
        interruption_frame = InterruptionFrame()
        result = await self.serializer.serialize(interruption_frame)

        self.assertIsNotNone(result)
        message = json.loads(result)

        self.assertEqual(message["event"], "clear")
        self.assertEqual(message["streamSid"], "test_stream_sid")

    async def test_serialize_transport_message_frames(self):
        """Test serialization of transport message frames."""
        message_data = {"type": "test", "data": "value"}

        # Test regular transport message
        transport_frame = OutputTransportMessageFrame(message=message_data)
        result = await self.serializer.serialize(transport_frame)
        self.assertEqual(result, json.dumps(message_data))

        # Test urgent transport message
        urgent_frame = OutputTransportMessageUrgentFrame(message=message_data)
        result = await self.serializer.serialize(urgent_frame)
        self.assertEqual(result, json.dumps(message_data))

    async def test_serialize_end_frame_triggers_hangup(self):
        """Test that EndFrame triggers hangup when auto_hang_up is enabled."""
        end_frame = EndFrame()

        with patch.object(self.serializer, "_hang_up_call", new_callable=AsyncMock) as mock_hangup:
            result = await self.serializer.serialize(end_frame)

            # Should trigger hangup and return None
            mock_hangup.assert_called_once()
            self.assertIsNone(result)

            # Second call should not trigger hangup again
            result = await self.serializer.serialize(end_frame)
            mock_hangup.assert_called_once()  # Still only called once
            self.assertIsNone(result)

    async def test_serialize_cancel_frame_triggers_hangup(self):
        """Test that CancelFrame triggers hangup when auto_hang_up is enabled."""
        cancel_frame = CancelFrame()

        with patch.object(self.serializer, "_hang_up_call", new_callable=AsyncMock) as mock_hangup:
            result = await self.serializer.serialize(cancel_frame)
            mock_hangup.assert_called_once()
            self.assertIsNone(result)

    async def test_serialize_unhandled_frame(self):
        """Test that unhandled frames return None."""
        # Create a mock frame that's not handled
        class MockFrame:
            pass

        mock_frame = MockFrame()
        result = await self.serializer.serialize(mock_frame)
        self.assertIsNone(result)

    async def test_deserialize_connected_event(self):
        """Test deserialization of connected event."""
        message = {
            "event": "connected",
            "protocol": "Call",
            "version": "1.0.0"
        }
        data = json.dumps(message)

        result = await self.serializer.deserialize(data)
        self.assertIsNone(result)  # Connected events don't produce frames

    async def test_deserialize_start_event(self):
        """Test deserialization of start event."""
        message = {
            "event": "start",
            "streamSid": "test_stream_sid",
            "start": {
                "session": "test_session",
                "callSid": "test_call_sid",
                "tracks": ["inbound"]
            }
        }
        data = json.dumps(message)

        result = await self.serializer.deserialize(data)
        self.assertIsNone(result)  # Start events don't produce frames

    async def test_deserialize_stop_event(self):
        """Test deserialization of stop event."""
        message = {
            "event": "stop",
            "streamSid": "test_stream_sid",
            "stop": {
                "session": "test_session",
                "callSid": "test_call_sid"
            }
        }
        data = json.dumps(message)

        result = await self.serializer.deserialize(data)
        self.assertIsNone(result)  # Stop events don't produce frames

    async def test_deserialize_media_event(self):
        """Test deserialization of media events to audio frames."""
        # Setup sample rate
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await self.serializer.setup(start_frame)

        # Create mock μ-law data and encode it properly as base64
        mock_ulaw_data = b"\x7F\x80\x81\x82" * 25
        payload_b64 = base64.b64encode(mock_ulaw_data).decode("utf-8")
        payload = json.dumps({"event": "media", "media": {"payload": payload_b64}})

        # Mock the audio conversion
        with patch("pipecat.serializers.cloudonix.ulaw_to_pcm") as mock_convert:
            mock_convert.return_value = b"\x00\x01\x02\x03" * 100  # Mock PCM data

            result = await self.serializer.deserialize(payload)

            self.assertIsInstance(result, InputAudioRawFrame)
            self.assertEqual(result.audio, b"\x00\x01\x02\x03" * 100)
            self.assertEqual(result.num_channels, 1)
            self.assertEqual(result.sample_rate, 16000)

            # Verify audio conversion was called
            mock_convert.assert_called_once()

    async def test_deserialize_media_event_empty_result(self):
        """Test that empty media data returns None."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await self.serializer.setup(start_frame)

        payload = json.dumps({"event": "media", "media": {"payload": ""}})

        with patch("pipecat.serializers.cloudonix.ulaw_to_pcm") as mock_convert:
            mock_convert.return_value = None
            result = await self.serializer.deserialize(payload)
            self.assertIsNone(result)

    async def test_deserialize_dtmf_event(self):
        """Test deserialization of DTMF events."""
        message = {
            "event": "dtmf",
            "dtmf": {"digit": "5"}
        }
        data = json.dumps(message)

        result = await self.serializer.deserialize(data)

        self.assertIsInstance(result, InputDTMFFrame)
        self.assertEqual(result.button, KeypadEntry.FIVE)

    async def test_deserialize_dtmf_event_invalid_digit(self):
        """Test deserialization of invalid DTMF digits."""
        message = {
            "event": "dtmf",
            "dtmf": {"digit": "invalid"}
        }
        data = json.dumps(message)

        result = await self.serializer.deserialize(data)
        self.assertIsNone(result)  # Invalid digits return None

    async def test_deserialize_unknown_event(self):
        """Test deserialization of unknown events."""
        message = {"event": "unknown", "data": "test"}
        data = json.dumps(message)

        result = await self.serializer.deserialize(data)
        self.assertIsNone(result)

    async def test_deserialize_malformed_json(self):
        """Test deserialization of malformed JSON."""
        data = "invalid json"

        # Should not raise exception, just return None
        result = await self.serializer.deserialize(data)
        self.assertIsNone(result)

    async def test_parameter_validation_auto_hangup_enabled(self):
        """Test parameter validation when auto_hang_up is enabled."""
        with self.assertRaises(ValueError) as context:
            CloudonixFrameSerializer(
                stream_sid="test_stream_sid",
                # Missing required parameters for auto_hang_up
            )

        self.assertIn("auto_hang_up is enabled but missing required parameters", str(context.exception))
        self.assertIn("call_sid", str(context.exception))
        self.assertIn("domain_id", str(context.exception))
        self.assertIn("bearer_token", str(context.exception))

    async def test_parameter_validation_auto_hangup_disabled(self):
        """Test that validation is skipped when auto_hang_up is disabled."""
        params = CloudonixFrameSerializer.InputParams(auto_hang_up=False)

        # Should not raise exception even with missing credentials
        serializer = CloudonixFrameSerializer(
            stream_sid="test_stream_sid",
            params=params,
        )
        self.assertIsNotNone(serializer)

    async def test_default_sample_rate(self):
        """Test that default sample rate is 8000 Hz."""
        params = CloudonixFrameSerializer.InputParams()
        self.assertEqual(params.sample_rate, 8000)

    async def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters provided."""
        params = CloudonixFrameSerializer.InputParams(
            sample_rate=8000,
            auto_hang_up=True
        )

        serializer = CloudonixFrameSerializer(
            stream_sid="test_stream_sid",
            call_sid="test_call_sid",
            domain_id="test_domain_id",
            bearer_token="test_bearer_token",
            session_token="test_session_token",
            region="us-east-1",
            edge="ashburn",
            params=params,
        )

        self.assertEqual(serializer._stream_sid, "test_stream_sid")
        self.assertEqual(serializer._call_sid, "test_call_sid")
        self.assertEqual(serializer._domain_id, "test_domain_id")
        self.assertEqual(serializer._bearer_token, "test_bearer_token")
        self.assertEqual(serializer._session_token, "test_session_token")
        self.assertEqual(serializer._cloudonix_sample_rate, 8000)
        self.assertTrue(serializer._params.auto_hang_up)


if __name__ == "__main__":
    unittest.main()