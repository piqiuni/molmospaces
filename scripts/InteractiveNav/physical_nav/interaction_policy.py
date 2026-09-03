"""Composable interaction policies for simulator and physical platforms.

Navigation owns the stable ``InteractionRequest``/``InteractionResult``
contract.  An actuator performs the requested operation and a verifier owns
the postcondition.  This keeps force, human-assist and VLA policies swappable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import time
from typing import Any, Callable, Iterable, Protocol


TARGET_KINDS = {"door", "fridge", "drawer"}


@dataclass(frozen=True)
class InteractionRequest:
    command_id: str
    decision_id: str = ""
    candidate_id: str = ""
    target_id: str = ""
    target_kind: str = "door"
    action: str = "open"
    expected_state: str = "open"
    target_name: str = ""
    bbox: tuple[float, float, float, float] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InteractionRequest":
        interaction = payload.get("interaction_command") or payload
        kind = str(
            interaction.get("target_kind")
            or interaction.get("container_kind")
            or payload.get("target_kind")
            or ("door" if str(payload.get("node_type", "")).casefold() == "portal" else "fridge")
        ).casefold()
        if kind in {"refrigerator", "refrigerator_door", "fridge_door"}:
            kind = "fridge"
        if kind in {"drawer_container", "drawer_cabinet", "cabinet"}:
            kind = "drawer"
        raw_bbox = interaction.get("bbox") or interaction.get("drawer_container_bbox_2d")
        bbox = None
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            try:
                bbox = tuple(float(v) for v in raw_bbox[:4])
            except (TypeError, ValueError):
                bbox = None
        action = str(interaction.get("action") or payload.get("action") or "open").casefold()
        expected = str(interaction.get("expected_state") or payload.get("expected_state") or ("closed" if action == "close" else "open"))
        return cls(
            command_id=str(payload.get("command_id") or ""),
            decision_id=str(payload.get("decision_id") or ""),
            candidate_id=str(payload.get("candidate_id") or ""),
            target_id=str(interaction.get("object_id") or interaction.get("node_id") or payload.get("target_id") or ""),
            target_kind=kind,
            action=action,
            expected_state=expected,
            target_name=str(payload.get("target_name") or interaction.get("target_name") or ""),
            bbox=bbox,
            context=dict(payload),
        )


@dataclass(frozen=True)
class InteractionResult:
    command_id: str
    success: bool
    status: str
    target_id: str
    target_kind: str
    pre_state: str = "unknown"
    post_state: str = "unknown"
    verification: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InteractionActuator(Protocol):
    name: str

    def supports(self, request: InteractionRequest) -> bool: ...

    def execute(self, request: InteractionRequest, cancel: Callable[[], bool]) -> dict[str, Any]: ...


class InteractionVerifier(Protocol):
    name: str

    def verify(self, request: InteractionRequest, cancel: Callable[[], bool]) -> dict[str, Any]: ...


class CallbackActuator:
    """Adapter for simulator force code or a future VLA client."""

    def __init__(self, name: str, callback: Callable[[InteractionRequest], dict[str, Any]], supported: Iterable[str] = TARGET_KINDS):
        self.name = str(name)
        self._callback = callback
        self._supported = {str(value).casefold() for value in supported}

    def supports(self, request: InteractionRequest) -> bool:
        return request.target_kind in self._supported

    def execute(self, request: InteractionRequest, cancel: Callable[[], bool]) -> dict[str, Any]:
        if cancel():
            return {"success": False, "status": "CANCELLED", "reason": "cancelled_before_actuation"}
        return dict(self._callback(request) or {})


class HumanAssistActuator:
    name = "human_assist"

    def __init__(self, speak: Callable[[str, bool], dict[str, Any]], retry_count: int = 2, retry_interval_s: float = 12.0):
        self._speak = speak
        self.retry_count = max(0, int(retry_count))
        self.retry_interval_s = max(0.0, float(retry_interval_s))

    def supports(self, request: InteractionRequest) -> bool:
        return request.target_kind in TARGET_KINDS and request.action in {"open", "close", "scan"}

    def execute(self, request: InteractionRequest, cancel: Callable[[], bool]) -> dict[str, Any]:
        if request.action not in {"open", "scan"}:
            return {"success": False, "status": "FAILED", "reason": "human_policy_only_opens"}
        noun = {"door": "门", "fridge": "冰箱", "drawer": "抽屉"}.get(request.target_kind, request.target_kind)
        text = f"您好，请帮我把前面的{noun}打开，谢谢。"
        attempts = []
        for attempt in range(self.retry_count + 1):
            if cancel():
                return {"success": False, "status": "CANCELLED", "reason": "cancelled_during_request", "attempts": attempts}
            try:
                response = dict(self._speak(text, True) or {})
                attempts.append({"attempt": attempt + 1, "text": text, "response": response})
                if response.get("accepted", True) is not False:
                    return {"success": True, "status": "REQUESTED", "speech": response, "attempts": attempts, "text": text}
            except Exception as exc:
                attempts.append({"attempt": attempt + 1, "error": str(exc)[:160]})
            if attempt < self.retry_count:
                deadline = time.monotonic() + self.retry_interval_s
                while time.monotonic() < deadline:
                    if cancel():
                        return {"success": False, "status": "CANCELLED", "reason": "cancelled_between_requests", "attempts": attempts}
                    time.sleep(min(0.10, deadline - time.monotonic()))
        return {"success": False, "status": "FAILED", "reason": "speech_request_failed", "attempts": attempts}


class VisualMLLMVerifier:
    """M3 verifier requiring the target to remain open for a stable interval."""

    name = "physical_visual_mllm"

    def __init__(self, image_provider: Callable[[], str | None], request_json: Callable[..., dict[str, Any]], emit: Callable[[dict[str, Any]], None] | None = None, *, stable_open_s: float = 3.0, sample_period_s: float = 1.0, timeout_s: float = 45.0, min_confidence: float = 0.60, temporary_skip_s: float = 30.0):
        self._image_provider = image_provider
        self._request_json = request_json
        self._emit = emit or (lambda _event: None)
        self.stable_open_s = max(0.5, float(stable_open_s))
        self.sample_period_s = max(0.2, float(sample_period_s))
        self.timeout_s = max(self.stable_open_s + 1.0, float(timeout_s))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.temporary_skip_s = max(0.0, float(temporary_skip_s))

    @staticmethod
    def _extract(payload: dict[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices") or []
        content = ((choices[0] if choices else {}).get("message") or {}).get("content", payload)
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        if isinstance(content, dict):
            return content
        text = str(content)
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        return {"state": "unknown", "confidence": 0.0, "reason": text[:160]}

    def verify(self, request: InteractionRequest, cancel: Callable[[], bool]) -> dict[str, Any]:
        started = time.monotonic()
        started_at = time.time()
        deadline_at = started_at + self.timeout_s
        open_since: float | None = None
        last_sample = 0.0
        evidence: list[dict[str, Any]] = []
        target_name = request.target_name or request.target_kind
        base_event = {
            "module": "M3",
            "stage": "M3",
            "role": "physical_visual_mllm",
            "command_id": request.command_id,
            "target_id": request.target_id,
            "target_name": target_name,
            "target_kind": request.target_kind,
            "sample_period_s": self.sample_period_s,
            "deadline_at": deadline_at,
        }
        # Publish the active target before the first (potentially slow) model
        # request, so the dashboard follows policy state rather than model
        # response latency.
        self._emit({
            **base_event,
            "phase": "SCHEDULED",
            "result": {"status": "WAITING", "reason": "interaction_policy_started"},
            "next_call_at": started_at,
            "timestamp": started_at,
        })
        while time.monotonic() - started < self.timeout_s:
            if cancel():
                cancelled_at = time.time()
                self._emit({
                    **base_event,
                    "phase": "CANCELLED",
                    "result": {
                        "status": "CANCELLED",
                        "reason": "cancelled_during_m3",
                    },
                    "timestamp": cancelled_at,
                })
                return {"success": False, "status": "CANCELLED", "reason": "cancelled_during_m3", "evidence": evidence}
            now = time.monotonic()
            if now - last_sample < self.sample_period_s:
                time.sleep(0.05)
                continue
            last_sample = now
            evaluating_at = time.time()
            self._emit({
                **base_event,
                "phase": "EVALUATING",
                "result": {"status": "EVALUATING", "reason": "m3_request_started"},
                "timestamp": evaluating_at,
            })
            image = self._image_provider()
            if not image:
                event_timestamp = time.time()
                sample = {"state": "unknown", "reason": "image_unavailable", "timestamp": event_timestamp}
                evidence.append(sample)
                self._emit({
                    **base_event,
                    "phase": "SCHEDULED",
                    "result": sample,
                    "image_available": False,
                    "next_call_at": event_timestamp + self.sample_period_s,
                    "timestamp": event_timestamp,
                })
                open_since = None
                time.sleep(0.05)
                continue
            target_reference = request.target_id or request.candidate_id
            prompt = (
                f"判断图像中的 {request.target_kind}（{target_name}"
                f"{f'，对象 {target_reference}' if target_reference else ''}）"
                f"是否处于 {request.expected_state} 状态。只依据可见证据，输出 JSON："
                "{state: open|closed|ajar|unknown, confidence: 0..1, visible_change: yes|no|unknown, reason: string}."
            )
            request_started = time.monotonic()
            try:
                response = self._request_json(
                    prompt, image_data_url=image, max_tokens=128
                )
            except Exception as exc:
                event_timestamp = time.time()
                sample = {
                    "state": "unknown",
                    "confidence": 0.0,
                    "reason": f"m3_request_failed: {str(exc)[:120]}",
                    "latency_s": time.monotonic() - request_started,
                    "timestamp": event_timestamp,
                }
                evidence.append(sample)
                self._emit({
                    **base_event,
                    "phase": "SCHEDULED",
                    "result": sample,
                    "image_available": True,
                    "next_call_at": event_timestamp + self.sample_period_s,
                    "timestamp": event_timestamp,
                })
                open_since = None
                continue
            parsed = self._extract(response if isinstance(response, dict) else {})
            state = str(parsed.get("state") or parsed.get("observed_state") or "unknown").casefold()
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            is_open = state in {"open", "ajar"} and confidence >= self.min_confidence
            event_timestamp = time.time()
            # The verifier is sequential: a slow VLM call itself consumes the
            # sampling period. Report the true earliest next start instead of
            # unconditionally adding another full period after completion.
            next_call_delay_s = max(
                0.0, self.sample_period_s - (time.monotonic() - last_sample)
            )
            sample = {"state": state, "confidence": confidence, "reason": str(parsed.get("reason") or "")[:160], "latency_s": time.monotonic() - request_started, "timestamp": event_timestamp}
            evidence.append(sample)
            self._emit({
                **base_event,
                "phase": "SCHEDULED",
                "request": prompt,
                "result": sample,
                "image_available": True,
                "next_call_at": event_timestamp + next_call_delay_s,
                "timestamp": event_timestamp,
            })
            if is_open:
                if open_since is None:
                    open_since = now
                stable_for = now - open_since
                if stable_for >= self.stable_open_s:
                    succeeded_at = time.time()
                    self._emit({
                        **base_event,
                        "phase": "SUCCEEDED",
                        "result": {
                            **sample,
                            "status": "SUCCEEDED",
                            "stable_duration_s": stable_for,
                        },
                        "timestamp": succeeded_at,
                    })
                    return {"success": True, "status": "SUCCEEDED", "post_state": "open", "stable_duration_s": stable_for, "verification_source": self.name, "evidence": evidence}
            else:
                open_since = None
        timed_out_at = time.time()
        self._emit({
            **base_event,
            "phase": "TIMEOUT",
            "result": {"status": "TIMEOUT", "reason": "target_not_open_within_timeout"},
            "timestamp": timed_out_at,
        })
        return {
            "success": False,
            "status": "TIMEOUT",
            "post_state": "unknown",
            "verification_source": self.name,
            "reason": "target_not_open_within_timeout",
            "temporary_skip_s": self.temporary_skip_s,
            "evidence": evidence,
        }


class InteractionPolicy:
    """Orchestrates one actuator and one verifier without owning navigation."""

    def __init__(self, actuator: InteractionActuator, verifier: InteractionVerifier, emit: Callable[[dict[str, Any]], None] | None = None):
        self.actuator = actuator
        self.verifier = verifier
        self._emit = emit or (lambda _event: None)

    def execute(self, request: InteractionRequest, cancel: Callable[[], bool] = lambda: False) -> InteractionResult:
        if not request.command_id:
            return InteractionResult("", False, "REJECTED", request.target_id, request.target_kind, detail={"reason": "missing_command_id"})
        if not self.actuator.supports(request):
            return InteractionResult(request.command_id, False, "REJECTED", request.target_id, request.target_kind, detail={"reason": "actuator_unsupported", "actuator": getattr(self.actuator, "name", "")})
        self._emit({"module": "POLICY", "stage": "STARTED", "command_id": request.command_id, "target_id": request.target_id, "target_kind": request.target_kind, "timestamp": time.time()})
        pre_state = str(request.context.get("pre_state") or "unknown")
        actuation = self.actuator.execute(request, cancel)
        if not actuation.get("success") and actuation.get("status") not in {"REQUESTED"}:
            actuation_reason = str(actuation.get("reason") or "interaction_actuation_failed")
            speech_transport_failure = actuation_reason == "speech_request_failed"
            return InteractionResult(
                request.command_id,
                False,
                str(actuation.get("status") or "FAILED"),
                request.target_id,
                request.target_kind,
                pre_state=pre_state,
                detail={
                    "actuation": actuation,
                    "reason": (
                        "interaction_transport_speech_unavailable"
                        if speech_transport_failure
                        else actuation_reason
                    ),
                    "failure_reason": (
                        "interaction_transport_speech_unavailable"
                        if speech_transport_failure
                        else actuation_reason
                    ),
                    "retryable": speech_transport_failure,
                },
            )
        verification = self.verifier.verify(request, cancel)
        result = InteractionResult(
            request.command_id,
            bool(verification.get("success")),
            str(verification.get("status") or ("SUCCEEDED" if verification.get("success") else "FAILED")),
            request.target_id,
            request.target_kind,
            pre_state=pre_state,
            post_state=str(verification.get("post_state") or "unknown"),
            verification={**verification, "verification_source": getattr(self.verifier, "name", "")},
            detail={"actuation": actuation},
        )
        self._emit({"module": "POLICY", "stage": "FINISHED", "result": result.to_dict()})
        return result


def build_interaction_policy(
    profile: str,
    *,
    speak: Callable[[str, bool], dict[str, Any]] | None = None,
    image_provider: Callable[[], str | None] | None = None,
    request_json: Callable[..., dict[str, Any]] | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    actuator_callback: Callable[[InteractionRequest], dict[str, Any]] | None = None,
    verifier: InteractionVerifier | None = None,
    options: dict[str, Any] | None = None,
) -> InteractionPolicy:
    """Build a named profile without changing the navigation-side contract."""
    options = dict(options or {})
    normalized = str(profile or "physical_human").casefold()
    if normalized == "physical_human":
        if speak is None or image_provider is None or request_json is None:
            raise ValueError("physical_human requires speak, image_provider and request_json")
        actuator = HumanAssistActuator(
            speak,
            retry_count=int(options.get("speech_retry_count", 2)),
            retry_interval_s=float(options.get("speech_retry_interval_s", 12.0)),
        )
        selected_verifier = VisualMLLMVerifier(
            image_provider,
            request_json,
            emit,
            stable_open_s=float(options.get("stable_open_s", 3.0)),
            sample_period_s=float(options.get("m3_sample_period_s", 1.0)),
            timeout_s=float(options.get("interaction_timeout_s", 45.0)),
            min_confidence=float(options.get("m3_min_confidence", 0.60)),
            temporary_skip_s=float(options.get("temporary_skip_s", 30.0)),
        )
        return InteractionPolicy(actuator, selected_verifier, emit)
    if normalized in {"simulation_force", "physical_vla"}:
        if actuator_callback is None:
            raise ValueError(f"{normalized} requires actuator_callback")
        if verifier is None:
            raise ValueError(f"{normalized} requires a verifier")
        return InteractionPolicy(CallbackActuator(normalized, actuator_callback), verifier, emit)
    raise ValueError(f"unknown interaction policy profile: {profile!r}")
