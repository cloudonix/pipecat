"""Cloudonix Media Streams WebSocket protocol serializer for Pipecat."""

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class CloudonixFrameSerializer(FrameSerializer):
    """Serializer for Cloudonix Media Streams WebSocket protocol.

    This serializer provides independent implementation of Cloudonix WebSocket protocol
    handling, including audio encoding/decoding, protocol events, and call termination.
    Fully decoupled from Twilio implementation for maximum compatibility and maintainability.
    """

    class InputParams(BaseModel):
        """Configuration parameters for CloudonixFrameSerializer.

        Parameters:
            sample_rate: Sample rate for audio processing, defaults to 8000 Hz (μ-law standard).
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """

        sample_rate: int = 8000
        auto_hang_up: bool = True

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        domain_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
        session_token: Optional[str] = None,
        region: Optional[str] = None,
        edge: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        """Initialize the CloudonixFrameSerializer.

        Args:
            stream_sid: The WebSocket Stream SID (Twilio-compatible).
            call_sid: The associated Cloudonix Call SID (optional, but required for auto hang-up).
            domain_id: Cloudonix domain ID (required for auto hang-up).
            bearer_token: Cloudonix bearer token (required for auto hang-up).
            session_token: Cloudonix session token from call initiation (optional, available for hangup).
            region: Optional region parameter (legacy compatibility).
            edge: Optional edge parameter (legacy compatibility).
            params: Configuration parameters.
        """
        self._params = params or CloudonixFrameSerializer.InputParams()

        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_sid:
                missing_credentials.append("call_sid")
            if not domain_id:
                missing_credentials.append("domain_id")
            if not bearer_token:
                missing_credentials.append("bearer_token")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )

        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._domain_id = domain_id
        self._bearer_token = bearer_token
        self._session_token = session_token

        self._sample_rate = 0  # Pipeline input rate (set during setup)
        self._cloudonix_sample_rate = self._params.sample_rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._hangup_attempted = False

        logger.info(f"Cloudonix serializer initialized with session_token: {session_token}")
        logger.info(f"Cloudonix serializer params are now {self.__dict__}")

    async def _hang_up_call(self):
        """Terminate the Cloudonix call by issuing a DELETE request to the session endpoint."""
        logger.debug(f"Attempting hangup for call {self._call_sid}")

        # If session_token is not available, fall back to WebSocket close behavior
        if not self._session_token:
            logger.warning(
                f"No session_token available for call {self._call_sid}. "
                f"Relying on WebSocket close for hangup."
            )
            return

        # Validate required parameters for API call
        if not self._domain_id or not self._bearer_token:
            logger.warning(
                f"Missing domain_id or bearer_token for call {self._call_sid}. "
                f"Cannot perform explicit hangup via API."
            )
            return

        try:
            import aiohttp

            # Construct the DELETE session endpoint
            # Using "self" as customer-id as per Cloudonix documentation
            base_url = "https://api.cloudonix.io"
            endpoint = f"{base_url}/customers/self/domains/{self._domain_id}/sessions/{self._session_token}"

            # Prepare headers with Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self._bearer_token}",
                "Content-Type": "application/json",
            }

            logger.info(f"Terminating Cloudonix call {self._call_sid} via DELETE {endpoint}")

            # Make the DELETE request to terminate the session
            async with aiohttp.ClientSession() as session:
                async with session.delete(endpoint, headers=headers) as response:
                    status = response.status
                    response_text = await response.text()

                    if status in (200, 204, 404):
                        # 200/204: Success
                        # 404: Session already terminated (acceptable)
                        logger.info(
                            f"Successfully terminated Cloudonix session {self._session_token} "
                            f"(HTTP {status}), Response: {response_text}"
                        )
                    else:
                        logger.warning(
                            f"Unexpected response terminating Cloudonix session {self._session_token}: "
                            f"HTTP {status}, Response: {response_text}"
                        )

        except Exception as e:
            logger.error(f"Error terminating Cloudonix call {self._call_sid}: {e}", exc_info=True)

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Cloudonix WebSocket format.

        Handles conversion of various frame types to Cloudonix WebSocket messages.
        For EndFrames, initiates call termination if auto_hang_up is enabled.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            await self._hang_up_call()
            return None
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Cloudonix
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._cloudonix_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Cloudonix WebSocket data to Pipecat frames.

        Handles Cloudonix WebSocket protocol events independently:
        - connected: Sent when WebSocket connection is established
        - start: Sent when stream starts with metadata
        - stop: Sent when stream stops
        - media: Audio data in μ-law format (8kHz, mono)
        - dtmf: DTMF key presses

        Args:
            data: The raw WebSocket data from Cloudonix.

        Returns:
            A Pipecat frame corresponding to the Cloudonix event, or None if unhandled.
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return None
        event_type = message.get("event")

        # Handle Cloudonix-specific events
        if event_type == "connected":
            protocol = message.get("protocol")
            version = message.get("version")
            logger.debug(f"Cloudonix stream connected: protocol={protocol}, version={version}")
            return None
        elif event_type == "start":
            stream_sid = message.get("streamSid")
            start_data = message.get("start", {})
            session = start_data.get("session")
            call_sid = start_data.get("callSid")
            tracks = start_data.get("tracks", [])
            logger.debug(f"Cloudonix stream started: streamSid={stream_sid}, callSid={call_sid}, session={session}, tracks={tracks}")
            return None
        elif event_type == "stop":
            stream_sid = message.get("streamSid")
            stop_data = message.get("stop", {})
            session = stop_data.get("session")
            call_sid = stop_data.get("callSid")
            logger.debug(f"Cloudonix stream stopped: streamSid={stream_sid}, callSid={call_sid}, session={session}")
            return None

        # Handle standard WebSocket events independently
        elif event_type == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Cloudonix's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._cloudonix_sample_rate, self._sample_rate, self._input_resampler
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif event_type == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError as e:
                # Handle case where string doesn't match any enum value
                logger.warning(f"Invalid DTMF digit received: {digit}")
                return None
        else:
            return None