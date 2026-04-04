from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

from inventario_faces.domain.config import DistributedSettings
from inventario_faces.domain.entities import MediaType
from inventario_faces.utils.path_utils import ensure_directory, safe_stem
from inventario_faces.utils.serialization import to_serializable
from inventario_faces.utils.time_utils import utc_now


@dataclass(frozen=True)
class DistributedPlanEntry:
    index: int
    source_path: Path
    media_type: MediaType
    relative_path: str

    @property
    def lock_stem(self) -> str:
        digest = sha1(self.relative_path.encode("utf-8")).hexdigest()[:10]
        return f"{self.index:05d}_{safe_stem(self.source_path.stem)}_{digest}"


@dataclass(frozen=True)
class DistributedClaim:
    entry: DistributedPlanEntry
    lock_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class DistributedClaimResult:
    status: str
    claim: DistributedClaim | None = None
    detail: str | None = None


@dataclass(frozen=True)
class DistributedExecutionSnapshot:
    total_files: int
    completed_files: int
    active_claims: int
    pending_files: int
    processable_files: int = 0
    processable_completed_files: int = 0
    processable_active_claims: int = 0
    processable_pending_files: int = 0

    @property
    def is_complete(self) -> bool:
        return self.total_files > 0 and self.completed_files >= self.total_files


@dataclass(frozen=True)
class DistributedPartialValidation:
    entry: DistributedPlanEntry
    status: str
    detail: str
    partial_path: Path | None
    manifest_item: dict[str, Any]
    payload: dict[str, Any] | None = None

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


@dataclass(frozen=True)
class DistributedNodeStatus:
    node_id: str
    hostname: str
    pid: int | None
    phase: str
    current_relative_path: str | None
    last_heartbeat_utc: str | None
    age_seconds: float | None
    is_stale: bool


@dataclass(frozen=True)
class DistributedHealthSnapshot:
    total_files: int
    completed_files: int
    active_claims: int
    pending_files: int
    healthy_partials: int
    missing_partials: int
    corrupted_partials: int
    stale_claims: int
    active_nodes: int
    stale_nodes: int
    finalize_lock_active: bool
    processable_files: int = 0
    processable_completed_files: int = 0
    processable_active_claims: int = 0
    processable_pending_files: int = 0
    partials: tuple[DistributedPartialValidation, ...] = ()
    nodes: tuple[DistributedNodeStatus, ...] = ()

    @property
    def recovery_needed(self) -> bool:
        return self.missing_partials > 0 or self.corrupted_partials > 0


class DistributedNodeHeartbeat:
    def __init__(
        self,
        coordinator: "DistributedCoordinator",
        total_files: int,
    ) -> None:
        self._coordinator = coordinator
        self._total_files = total_files
        self._phase = "idle"
        self._current_entry: DistributedPlanEntry | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._coordinator.write_node_heartbeat(self._total_files, self._phase, self._current_entry)
        self._thread = threading.Thread(target=self._run, name="inventario_faces_heartbeat", daemon=True)
        self._thread.start()

    def update(self, phase: str, entry: DistributedPlanEntry | None = None) -> None:
        self._phase = phase
        self._current_entry = entry
        self._coordinator.write_node_heartbeat(self._total_files, self._phase, self._current_entry)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._coordinator.settings.heartbeat_interval_seconds * 2))
            self._thread = None
        self._coordinator.remove_node_heartbeat()

    def _run(self) -> None:
        interval_seconds = max(1, self._coordinator.settings.heartbeat_interval_seconds)
        while not self._stop_event.wait(interval_seconds):
            self._coordinator.write_node_heartbeat(self._total_files, self._phase, self._current_entry)


class DistributedCoordinator:
    def __init__(
        self,
        root_directory: Path,
        run_directory: Path,
        settings: DistributedSettings,
    ) -> None:
        self.root_directory = Path(root_directory).resolve()
        self.run_directory = ensure_directory(Path(run_directory))
        self.settings = settings
        self.hostname = settings.node_name.strip() if settings.node_name else socket.gethostname()
        self.pid = os.getpid()
        self.node_id = safe_stem(f"{self.hostname}_{self.pid}")
        self.started_at_utc = utc_now()

        self.coordination_directory = ensure_directory(self.run_directory / "distributed")
        self.claims_directory = ensure_directory(self.coordination_directory / "claims")
        self.partials_directory = ensure_directory(self.coordination_directory / "partials")
        self.nodes_directory = ensure_directory(self.coordination_directory / "nodes")
        self.plan_path = self.coordination_directory / "plan.json"
        self.completed_manifest_path = self.coordination_directory / "completed_manifest.json"
        self.manifest_mutex_path = self.coordination_directory / "completed_manifest.lock"
        self.finalize_lock_path = self.coordination_directory / "finalize.lock"
        self.node_file_path = self.nodes_directory / f"_node_{self.node_id}.json"
        self._write_mutex = threading.Lock()

    def load_or_create_plan(self, planned_files: list[tuple[Path, MediaType]]) -> list[DistributedPlanEntry]:
        if self.plan_path.exists():
            return self._load_plan()
        entries = [
            {
                "index": index,
                "relative_path": str(path.resolve().relative_to(self.root_directory)).replace("\\", "/"),
                "media_type": media_type.value,
            }
            for index, (path, media_type) in enumerate(planned_files, start=1)
        ]
        try:
            self._write_json_atomic(self.plan_path, entries, create_only=True)
        except FileExistsError:
            pass
        return self._load_plan()

    def load_completed_manifest(self) -> dict[str, dict[str, Any]]:
        data = self._load_json(self.completed_manifest_path, default=[])
        if not isinstance(data, list):
            return {}
        manifest: dict[str, dict[str, Any]] = {}
        for item in data:
            if isinstance(item, dict) and "relative_path" in item:
                manifest[str(item["relative_path"])] = item
        return manifest

    def snapshot(self, total_files: int) -> DistributedExecutionSnapshot:
        plan_entries = self._load_plan()
        processable_paths = self._processable_relative_paths(plan_entries)
        manifest = self.load_completed_manifest()
        claim_paths = list(self.claims_directory.glob("*.lock"))
        completed_files = len(manifest)
        active_claims = len(claim_paths)
        pending_files = max(0, total_files - completed_files - active_claims)
        return DistributedExecutionSnapshot(
            total_files=total_files,
            completed_files=completed_files,
            active_claims=active_claims,
            pending_files=pending_files,
            processable_files=len(processable_paths),
            processable_completed_files=sum(
                1 for relative_path in manifest if relative_path in processable_paths
            ),
            processable_active_claims=self._count_processable_claims(claim_paths, processable_paths),
            processable_pending_files=max(
                0,
                len(processable_paths)
                - sum(1 for relative_path in manifest if relative_path in processable_paths)
                - self._count_processable_claims(claim_paths, processable_paths),
            ),
        )

    def try_claim(self, entry: DistributedPlanEntry) -> DistributedClaimResult:
        manifest = self.load_completed_manifest()
        if entry.relative_path in manifest:
            return DistributedClaimResult(status="completed", detail="já concluído no manifesto compartilhado")

        lock_path = self.claims_directory / f"{entry.lock_stem}.lock"
        if lock_path.exists():
            if self._is_stale_lock(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                return DistributedClaimResult(status="busy", detail=self._read_lock_owner(lock_path))

        payload = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "relative_path": entry.relative_path,
            "claimed_at_utc": utc_now().isoformat(),
        }
        try:
            with lock_path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
        except FileExistsError:
            return DistributedClaimResult(status="busy", detail=self._read_lock_owner(lock_path))
        return DistributedClaimResult(
            status="claimed",
            claim=DistributedClaim(entry=entry, lock_path=lock_path, payload=payload),
        )

    def release_claim(self, claim: DistributedClaim) -> None:
        try:
            claim.lock_path.unlink()
        except FileNotFoundError:
            pass

    def write_partial_payload(
        self,
        entry: DistributedPlanEntry,
        payload: dict[str, Any],
        *,
        file_sha512: str,
    ) -> Path:
        output_path = self.partials_directory / f"{entry.lock_stem}.json"
        envelope = {
            "schema_version": 2,
            "node_id": self.node_id,
            "written_at_utc": utc_now().isoformat(),
            "entry": {
                "index": entry.index,
                "relative_path": entry.relative_path,
                "media_type": entry.media_type.value,
                "source_path": str(entry.source_path),
            },
            "file_sha512": file_sha512,
            "payload_sha256": self._compute_payload_digest(payload),
            "payload": payload,
        }
        self._write_json_atomic(output_path, envelope)
        return output_path

    def mark_completed(
        self,
        entry: DistributedPlanEntry,
        *,
        partial_path: Path,
        sha512: str,
        occurrence_count: int,
        track_count: int,
        keyframe_count: int,
        processing_error: str | None,
    ) -> None:
        payload = {
            "index": entry.index,
            "relative_path": entry.relative_path,
            "media_type": entry.media_type.value,
            "source_path": str(entry.source_path),
            "partial_path": str(partial_path),
            "sha512": sha512,
            "occurrence_count": occurrence_count,
            "track_count": track_count,
            "keyframe_count": keyframe_count,
            "processing_error": processing_error,
            "node_id": self.node_id,
            "completed_at_utc": utc_now().isoformat(),
        }

        mutex = self._acquire_mutex(self.manifest_mutex_path)
        try:
            current = list(self.load_completed_manifest().values())
            current = [item for item in current if item.get("relative_path") != entry.relative_path]
            current.append(payload)
            current.sort(key=lambda item: int(item.get("index", 0)))
            self._write_json_atomic(self.completed_manifest_path, current)
        finally:
            self._release_mutex(mutex)

    def try_acquire_finalize_lock(self) -> Path | None:
        if self.finalize_lock_path.exists() and self._is_stale_lock(self.finalize_lock_path):
            try:
                self.finalize_lock_path.unlink()
            except FileNotFoundError:
                pass
        payload = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "claimed_at_utc": utc_now().isoformat(),
        }
        try:
            with self.finalize_lock_path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
        except FileExistsError:
            return None
        return self.finalize_lock_path

    def release_finalize_lock(self) -> None:
        try:
            self.finalize_lock_path.unlink()
        except FileNotFoundError:
            pass

    def write_node_heartbeat(
        self,
        total_files: int,
        phase: str,
        current_entry: DistributedPlanEntry | None,
    ) -> None:
        payload = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "execution_label": self.settings.execution_label,
            "started_at_utc": self.started_at_utc.isoformat(),
            "last_heartbeat": utc_now().isoformat(),
            "total_files": total_files,
            "phase": phase,
            "current_relative_path": current_entry.relative_path if current_entry is not None else None,
        }
        try:
            self._write_json_atomic(self.node_file_path, payload)
        except OSError as exc:
            if not self._is_retryable_file_error(exc):
                raise

    def remove_node_heartbeat(self) -> None:
        try:
            self.node_file_path.unlink()
        except FileNotFoundError:
            pass

    def load_partial_payloads(self) -> list[dict[str, Any]]:
        manifest = self.load_completed_manifest()
        payloads: list[dict[str, Any]] = []
        for item in sorted(manifest.values(), key=lambda entry: int(entry.get("index", 0))):
            partial_path = Path(str(item.get("partial_path", "")))
            if partial_path.exists():
                validation = self.inspect_partial_from_manifest_item(item)
                if validation.is_healthy and validation.payload is not None:
                    payloads.append(validation.payload)
        return payloads

    def inspect_health(self, total_files: int | None = None) -> DistributedHealthSnapshot:
        plan_entries = self._load_plan()
        manifest = self.load_completed_manifest()
        if total_files is None:
            total_files = len(plan_entries)
        processable_paths = self._processable_relative_paths(plan_entries)
        partials = tuple(
            self.inspect_partial_from_manifest_item(item)
            for item in sorted(manifest.values(), key=lambda entry: int(entry.get("index", 0)))
        )
        claim_paths = list(self.claims_directory.glob("*.lock"))
        stale_claims = sum(1 for path in claim_paths if self._is_stale_lock(path))
        nodes = self.load_node_statuses()
        basic_snapshot = self.snapshot(total_files)
        return DistributedHealthSnapshot(
            total_files=basic_snapshot.total_files,
            completed_files=basic_snapshot.completed_files,
            active_claims=basic_snapshot.active_claims,
            pending_files=basic_snapshot.pending_files,
            healthy_partials=sum(1 for item in partials if item.status == "healthy"),
            missing_partials=sum(1 for item in partials if item.status == "missing"),
            corrupted_partials=sum(1 for item in partials if item.status == "corrupt"),
            stale_claims=stale_claims,
            active_nodes=sum(1 for node in nodes if not node.is_stale),
            stale_nodes=sum(1 for node in nodes if node.is_stale),
            finalize_lock_active=self.finalize_lock_path.exists(),
            processable_files=len(processable_paths),
            processable_completed_files=sum(
                1 for relative_path in manifest if relative_path in processable_paths
            ),
            processable_active_claims=self._count_processable_claims(claim_paths, processable_paths),
            processable_pending_files=max(
                0,
                len(processable_paths)
                - sum(1 for relative_path in manifest if relative_path in processable_paths)
                - self._count_processable_claims(claim_paths, processable_paths),
            ),
            partials=partials,
            nodes=nodes,
        )

    def inspect_partial_from_manifest_item(self, manifest_item: dict[str, Any]) -> DistributedPartialValidation:
        relative_path = str(manifest_item.get("relative_path", ""))
        entry = self._entry_from_manifest_item(manifest_item)
        partial_path_raw = manifest_item.get("partial_path")
        partial_path = Path(str(partial_path_raw)) if partial_path_raw else None
        if partial_path is None or not partial_path.exists():
            return DistributedPartialValidation(
                entry=entry,
                status="missing",
                detail="arquivo parcial ausente",
                partial_path=partial_path,
                manifest_item=manifest_item,
            )

        raw_payload, error = self._load_json_strict(partial_path)
        if error is not None or not isinstance(raw_payload, dict):
            return DistributedPartialValidation(
                entry=entry,
                status="corrupt",
                detail=(error or "conteudo JSON invalido"),
                partial_path=partial_path,
                manifest_item=manifest_item,
            )

        payload, detail = self._extract_payload_from_partial(entry, manifest_item, raw_payload)
        if payload is None:
            return DistributedPartialValidation(
                entry=entry,
                status="corrupt",
                detail=detail,
                partial_path=partial_path,
                manifest_item=manifest_item,
            )

        return DistributedPartialValidation(
            entry=entry,
            status="healthy",
            detail="parcial valido",
            partial_path=partial_path,
            manifest_item=manifest_item,
            payload=payload,
        )

    def load_node_statuses(self) -> tuple[DistributedNodeStatus, ...]:
        timeout_seconds = max(60, self.settings.stale_lock_timeout_minutes * 60)
        statuses: list[DistributedNodeStatus] = []
        for node_path in sorted(self.nodes_directory.glob("_node_*.json")):
            payload = self._load_json(node_path, default={})
            if not isinstance(payload, dict):
                continue
            try:
                age_seconds = time.time() - node_path.stat().st_mtime
            except FileNotFoundError:
                age_seconds = None
            statuses.append(
                DistributedNodeStatus(
                    node_id=str(payload.get("node_id", node_path.stem.replace("_node_", ""))),
                    hostname=str(payload.get("hostname", "desconhecido")),
                    pid=(None if payload.get("pid") in (None, "") else int(payload.get("pid"))),
                    phase=str(payload.get("phase", "desconhecido")),
                    current_relative_path=(
                        str(payload.get("current_relative_path"))
                        if payload.get("current_relative_path") not in (None, "")
                        else None
                    ),
                    last_heartbeat_utc=(
                        str(payload.get("last_heartbeat"))
                        if payload.get("last_heartbeat") not in (None, "")
                        else None
                    ),
                    age_seconds=age_seconds,
                    is_stale=(age_seconds is not None and age_seconds > timeout_seconds),
                )
            )
        return tuple(statuses)

    def _entry_from_manifest_item(self, manifest_item: dict[str, Any]) -> DistributedPlanEntry:
        relative_path = str(manifest_item.get("relative_path", ""))
        source_path_value = manifest_item.get("source_path")
        source_path = (
            Path(str(source_path_value)).resolve()
            if source_path_value not in (None, "")
            else (self.root_directory / Path(relative_path)).resolve()
        )
        media_type_value = str(manifest_item.get("media_type", MediaType.OTHER.value))
        try:
            media_type = MediaType(media_type_value)
        except ValueError:
            media_type = MediaType.OTHER
        return DistributedPlanEntry(
            index=int(manifest_item.get("index", 0)),
            relative_path=relative_path,
            source_path=source_path,
            media_type=media_type,
        )

    def _load_json_strict(self, path: Path) -> tuple[Any | None, str | None]:
        try:
            return json.loads(path.read_text(encoding="utf-8")), None
        except FileNotFoundError:
            return None, "arquivo ausente"
        except json.JSONDecodeError as exc:
            return None, f"JSON invalido: linha {exc.lineno}, coluna {exc.colno}"

    def _extract_payload_from_partial(
        self,
        entry: DistributedPlanEntry,
        manifest_item: dict[str, Any],
        raw_payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        if "payload" in raw_payload:
            payload = raw_payload.get("payload")
            if not isinstance(payload, dict):
                return None, "campo payload ausente ou invalido"
            entry_payload = raw_payload.get("entry")
            if isinstance(entry_payload, dict):
                if str(entry_payload.get("relative_path", "")) != entry.relative_path:
                    return None, "relative_path do parcial diverge do manifesto"
            expected_sha = str(manifest_item.get("sha512", ""))
            partial_sha = str(raw_payload.get("file_sha512", ""))
            if expected_sha and partial_sha and expected_sha != partial_sha:
                return None, "hash do arquivo no parcial diverge do manifesto"
            payload_digest = str(raw_payload.get("payload_sha256", ""))
            if payload_digest and payload_digest != self._compute_payload_digest(payload):
                return None, "hash interno do parcial nao confere"
        else:
            payload = raw_payload

        required_keys = {"file_record", "occurrences", "tracks", "keyframes", "raw_face_sizes", "selected_face_sizes"}
        if not required_keys.issubset(payload.keys()):
            missing = ", ".join(sorted(required_keys.difference(payload.keys())))
            return None, f"campos obrigatorios ausentes: {missing}"

        file_record = payload.get("file_record")
        if not isinstance(file_record, dict):
            return None, "file_record ausente ou invalido"

        manifest_sha = str(manifest_item.get("sha512", ""))
        file_sha = str(file_record.get("sha512", ""))
        if manifest_sha and file_sha and manifest_sha != file_sha:
            return None, "hash do file_record diverge do manifesto"

        count_expectations = (
            ("occurrences", "occurrence_count"),
            ("tracks", "track_count"),
            ("keyframes", "keyframe_count"),
        )
        for payload_key, manifest_key in count_expectations:
            items = payload.get(payload_key)
            if not isinstance(items, list):
                return None, f"{payload_key} invalido"
            expected_count = manifest_item.get(manifest_key)
            if expected_count not in (None, "") and int(expected_count) != len(items):
                return None, f"contagem inconsistente para {payload_key}"

        return payload, "parcial valido"

    def _load_plan(self) -> list[DistributedPlanEntry]:
        data = self._load_json(self.plan_path, default=[])
        entries: list[DistributedPlanEntry] = []
        if not isinstance(data, list):
            return entries
        for item in data:
            if not isinstance(item, dict):
                continue
            relative_path = str(item["relative_path"])
            entries.append(
                DistributedPlanEntry(
                    index=int(item["index"]),
                    relative_path=relative_path,
                    source_path=(self.root_directory / Path(relative_path)).resolve(),
                    media_type=MediaType(str(item["media_type"])),
                )
            )
        return entries

    def _processable_relative_paths(self, plan_entries: list[DistributedPlanEntry]) -> set[str]:
        return {
            entry.relative_path
            for entry in plan_entries
            if entry.media_type != MediaType.OTHER
        }

    def _count_processable_claims(self, claim_paths: list[Path], processable_paths: set[str]) -> int:
        total = 0
        for claim_path in claim_paths:
            payload = self._load_json(claim_path, default={})
            if not isinstance(payload, dict):
                continue
            relative_path = str(payload.get("relative_path", ""))
            if relative_path in processable_paths:
                total += 1
        return total

    def _is_stale_lock(self, lock_path: Path) -> bool:
        timeout_seconds = max(60, self.settings.stale_lock_timeout_minutes * 60)
        try:
            lock_age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        if lock_age <= timeout_seconds:
            return False
        payload = self._load_json(lock_path, default={})
        node_id = payload.get("node_id") if isinstance(payload, dict) else None
        if node_id:
            node_file = self.nodes_directory / f"_node_{node_id}.json"
            if node_file.exists():
                try:
                    node_age = time.time() - node_file.stat().st_mtime
                    return node_age > timeout_seconds
                except FileNotFoundError:
                    return True
        return True

    def _read_lock_owner(self, lock_path: Path) -> str:
        payload = self._load_json(lock_path, default={})
        if isinstance(payload, dict):
            hostname = payload.get("hostname") or "outro nó"
            pid = payload.get("pid")
            if pid not in (None, ""):
                return f"{hostname}:{pid}"
            return str(hostname)
        return "outro nó"

    def _acquire_mutex(self, path: Path, timeout_seconds: float = 10.0, retry_seconds: float = 0.1) -> Path:
        deadline = time.time() + timeout_seconds
        payload = {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "claimed_at_utc": utc_now().isoformat(),
        }
        while True:
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, ensure_ascii=False)
                return path
            except FileExistsError:
                if self._is_stale_lock(path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.time() >= deadline:
                    raise TimeoutError(f"Não foi possível obter o mutex compartilhado: {path.name}")
                time.sleep(retry_seconds)

    def _release_mutex(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _load_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            return default

    def _write_json_atomic(self, path: Path, payload: Any, create_only: bool = False) -> None:
        ensure_directory(path.parent)
        serialized = json.dumps(to_serializable(payload), indent=2, ensure_ascii=False)
        if create_only:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            return
        temporary_path: Path | None = None
        with self._write_mutex:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f"{path.name}.{self.node_id}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    stream.write(serialized)
                    stream.flush()
                    temporary_path = Path(stream.name)

                last_error: OSError | None = None
                for attempt in range(12):
                    try:
                        os.replace(temporary_path, path)
                        temporary_path = None
                        return
                    except OSError as exc:
                        if not self._is_retryable_file_error(exc):
                            raise
                        last_error = exc
                        time.sleep(min(0.05 * (attempt + 1), 0.35))

                if last_error is not None:
                    raise last_error
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink()
                    except FileNotFoundError:
                        pass

    def _is_retryable_file_error(self, error: OSError) -> bool:
        winerror = getattr(error, "winerror", None)
        if winerror in {5, 32}:
            return True
        return isinstance(error, PermissionError)

    def _compute_payload_digest(self, payload: Any) -> str:
        serialized = json.dumps(
            to_serializable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(serialized.encode("utf-8")).hexdigest()
