"""
approval_store.py

Session-scoped permission store for Verifier actions.

Approval scopes:
    - once:     approve this single action, immediately consumed
    - duration: approve for N seconds from now
    - session:  approve until the app closes

Grants are keyed by (action, arguments_fingerprint) so that a
grant for kill_process(pid=1234) does NOT silently approve
kill_process(pid=9999).

A separate, time-bounded cache stores a sudo password so that
multiple sudo commands within the grant window do not re-prompt.
The password is stored in memory only and is never written to disk.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

Scope = Literal["once", "duration", "session"]


# ============================================================
# GRANT
# ============================================================

@dataclass
class Grant:
    grant_id: str
    action: str
    scope: Scope
    created_at: float
    expires_at: float | None      # None = never expires (session)
    once_consumed: bool = False
    signature: str = ""

    def is_valid(self, now: float | None = None) -> bool:
        if self.once_consumed:
            return False
        if self.expires_at is None:
            return True
        now = now if now is not None else time.time()
        return now < self.expires_at


# ============================================================
# PENDING
# ============================================================

@dataclass
class PendingApproval:
    approval_id: str
    action: str
    arguments: dict
    risk: str
    description: str
    created_at: float = field(default_factory=time.time)


# ============================================================
# STORE
# ============================================================

class ApprovalStore:
    """
    Thread-safe approval registry.

    All methods are safe to call from any thread.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._grants: dict[str, Grant] = {}
        self._pending: dict[str, PendingApproval] = {}

        # Sudo password cache (in-memory only).
        self._sudo_password: str | None = None
        self._sudo_expires_at: float | None = None

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def fingerprint(arguments: dict) -> str:
        try:
            items = sorted(
                (str(k), repr(v))
                for k, v in (arguments or {}).items()
            )
        except Exception:
            return "<unhashable>"
        return "|".join(f"{k}={v}" for k, v in items)

    # ------------------------------------------------------------------
    # Pending
    # ------------------------------------------------------------------

    def register_pending(
        self,
        action: str,
        arguments: dict,
        risk: str,
        description: str,
    ) -> PendingApproval:
        with self._lock:
            pending = PendingApproval(
                approval_id=str(uuid.uuid4()),
                action=action,
                arguments=arguments,
                risk=risk,
                description=description,
            )
            self._pending[pending.approval_id] = pending
            return pending

    def pop_pending(self, approval_id: str):
        with self._lock:
            return self._pending.pop(approval_id, None)

    # ------------------------------------------------------------------
    # Grants
    # ------------------------------------------------------------------

    def grant(
        self,
        action: str,
        arguments: dict,
        scope: Scope,
        duration_seconds: int | None = None,
    ) -> Grant:
        with self._lock:
            now = time.time()

            if scope == "once":
                expires_at = now + 60
            elif scope == "duration":
                expires_at = now + int(duration_seconds or 3600)
            else:
                expires_at = None

            grant = Grant(
                grant_id=str(uuid.uuid4()),
                action=action,
                scope=scope,
                created_at=now,
                expires_at=expires_at,
                signature=self.fingerprint(arguments),
            )
            self._grants[grant.grant_id] = grant
            return grant

    def check(self, action: str, arguments: dict) -> Grant | None:
        """
        Return a valid grant matching (action, arguments), or None.
        Consumes 'once' grants when matched.
        """
        with self._lock:
            signature = self.fingerprint(arguments)
            now = time.time()

            for grant in self._grants.values():
                if grant.action != action:
                    continue
                if grant.signature != signature:
                    continue
                if not grant.is_valid(now):
                    continue

                if grant.scope == "once":
                    grant.once_consumed = True

                return grant

            return None

    def revoke(self, grant_id: str) -> bool:
        with self._lock:
            return self._grants.pop(grant_id, None) is not None

    def revoke_action(self, action: str) -> int:
        with self._lock:
            removed = 0
            for gid in list(self._grants.keys()):
                if self._grants[gid].action == action:
                    del self._grants[gid]
                    removed += 1
            return removed

    def list_grants(self) -> list[Grant]:
        with self._lock:
            return [g for g in self._grants.values() if g.is_valid()]

    # ------------------------------------------------------------------
    # Sudo password cache
    # ------------------------------------------------------------------

    def grant_sudo(
        self,
        password: str,
        duration_seconds: int | None,
    ):
        """
        Store a sudo password for a bounded duration.

        duration_seconds=None means "until clear_sudo is called"
        (i.e. session-scoped).
        """
        with self._lock:
            self._sudo_password = password
            if duration_seconds is None:
                self._sudo_expires_at = None
            else:
                self._sudo_expires_at = (
                    time.time() + int(duration_seconds)
                )

    def check_sudo(self) -> str | None:
        """
        Return the cached sudo password, or None.

        Automatically invalidates and clears the password if the
        window has expired.
        """
        with self._lock:
            pw = self._sudo_password
            exp = self._sudo_expires_at

            if not pw:
                return None

            if exp is not None and time.time() >= exp:
                self._sudo_password = None
                self._sudo_expires_at = None
                return None

            return pw

    def clear_sudo(self):
        with self._lock:
            self._sudo_password = None
            self._sudo_expires_at = None

    # ------------------------------------------------------------------
    # Bulk clear
    # ------------------------------------------------------------------

    def clear(self):
        with self._lock:
            self._grants.clear()
            self._pending.clear()
            self._sudo_password = None
            self._sudo_expires_at = None


# ============================================================
# SINGLETON
# ============================================================

STORE = ApprovalStore()