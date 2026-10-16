from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.inbox.models import (
    STATUSES, CandidateInput, InboxCandidateORM, InboxStatusEventORM,
    InboxTriageResultORM, TriageOutput, utc_now,
)
from app.inbox.urlnorm import content_hash, normalize_url


class InboxStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def duplicate_reasons(self, candidate: CandidateInput) -> list[str]:
        with self._session_factory() as session:
            return self._duplicate_reasons(session, candidate)

    @staticmethod
    def _duplicate_reasons(session: Session, candidate: CandidateInput) -> list[str]:
        keys = {
            "external_id": candidate.external_id,
            "url_normalized": normalize_url(candidate.url),
            "content_hash": content_hash(candidate.title, candidate.summary),
        }
        rows = session.scalars(select(InboxCandidateORM).where(or_(
            *(getattr(InboxCandidateORM, key) == value for key, value in keys.items())
        ))).all()
        # A candidate can match several keys (even across different stored rows).
        return [key for key, value in keys.items()
                if any(getattr(row, key) == value for row in rows)]

    def add_candidate(self, candidate: CandidateInput) -> tuple[InboxCandidateORM | None, list[str]]:
        with self._session_factory.begin() as session:
            reasons = self._duplicate_reasons(session, candidate)
            if reasons:
                return None, reasons
            values = candidate.model_dump(exclude={"raw"})
            now = utc_now()
            row = InboxCandidateORM(
                **values, url_normalized=normalize_url(candidate.url),
                content_hash=content_hash(candidate.title, candidate.summary),
                raw_json=json.dumps(candidate.raw, ensure_ascii=False), status="new",
                created_at=now, status_updated_at=now,
            )
            session.add(row)
            session.flush()
            session.add(InboxStatusEventORM(
                candidate_id=row.id, from_status=None, to_status="new", created_at=now,
            ))
            return row, []

    def get_candidate(self, candidate_id: int) -> InboxCandidateORM | None:
        with self._session_factory() as session:
            return session.get(InboxCandidateORM, candidate_id)

    def list_candidates(self, *, status: str | None = None, limit: int = 20) -> list[InboxCandidateORM]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        statement = select(InboxCandidateORM).order_by(InboxCandidateORM.id.desc()).limit(limit)
        if status is not None:
            self._validate_status(status)
            statement = statement.where(InboxCandidateORM.status == status)
        with self._session_factory() as session:
            return list(session.scalars(statement))

    def discovery_candidates(
        self, *, limit: int = 200,
    ) -> list[tuple[InboxCandidateORM, InboxTriageResultORM | None]]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        latest = select(
            InboxTriageResultORM.candidate_id,
            func.max(InboxTriageResultORM.id).label("result_id"),
        ).group_by(InboxTriageResultORM.candidate_id).subquery()
        with self._session_factory() as session:
            return list(session.execute(select(InboxCandidateORM, InboxTriageResultORM)
                .outerjoin(latest, latest.c.candidate_id == InboxCandidateORM.id)
                .outerjoin(InboxTriageResultORM, InboxTriageResultORM.id == latest.c.result_id)
                .order_by(InboxCandidateORM.id.desc()).limit(limit)))

    def change_status(
        self, candidate_id: int, status: str, *, user_reason: str | None = None,
        drop_reason: str | None = None, note_source: str | None = None,
    ) -> InboxCandidateORM:
        self._validate_status(status)
        with self._session_factory.begin() as session:
            row = session.get(InboxCandidateORM, candidate_id, with_for_update=True)
            if row is None:
                raise ValueError(f"Candidate {candidate_id} does not exist.")
            if row.status != status:
                now = utc_now()
                session.add(InboxStatusEventORM(
                    candidate_id=row.id, from_status=row.status, to_status=status, created_at=now,
                ))
                row.status = status
                row.status_updated_at = now
            for name, value in (("user_reason", user_reason), ("drop_reason", drop_reason),
                                ("note_source", note_source)):
                if value is not None:
                    setattr(row, name, value)
            return row

    def list_status_events(self, candidate_id: int) -> list[InboxStatusEventORM]:
        with self._session_factory() as session:
            return list(session.scalars(select(InboxStatusEventORM).where(
                InboxStatusEventORM.candidate_id == candidate_id
            ).order_by(InboxStatusEventORM.id)))

    def append_triage_result(
        self, candidate_id: int, *, prompt_version: str, model: str,
        output: TriageOutput | None, raw_output: str | None, error: str | None = None,
        mark_triaged: bool = False, created_at: datetime | None = None,
    ) -> InboxTriageResultORM:
        if (output is None) == (not bool(error)):
            raise ValueError("Provide either a validated output or a non-empty error.")
        with self._session_factory.begin() as session:
            candidate = session.get(InboxCandidateORM, candidate_id, with_for_update=True)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} does not exist.")
            values = output.model_dump(exclude={"prerequisites", "unknowns"}) if output else {}
            row = InboxTriageResultORM(
                candidate_id=candidate_id, prompt_version=prompt_version, model=model,
                **values, prerequisites_json=json.dumps(output.prerequisites if output else [], ensure_ascii=False),
                unknowns_json=json.dumps(output.unknowns if output else [], ensure_ascii=False),
                similar_json="[]", raw_output=raw_output, error=error,
                created_at=created_at or utc_now(),
            )
            session.add(row)
            # A re-triage must preserve an existing human decision.
            if mark_triaged and output is not None and candidate.status == "new":
                now = utc_now()
                session.add(InboxStatusEventORM(
                    candidate_id=candidate_id, from_status="new", to_status="triaged", created_at=now,
                ))
                candidate.status = "triaged"
                candidate.status_updated_at = now
            session.flush()
            return row

    def triage_candidates(
        self, *, limit: int = 20, candidate_id: int | None = None, retriage: bool = False,
    ) -> list[InboxCandidateORM]:
        if limit < 1 or (candidate_id is not None and candidate_id < 1):
            raise ValueError("limit and candidate_id must be positive.")
        failures = select(func.count(InboxTriageResultORM.id)).where(
            InboxTriageResultORM.candidate_id == InboxCandidateORM.id,
            InboxTriageResultORM.error.is_not(None), InboxTriageResultORM.error != "",
        ).correlate(InboxCandidateORM).scalar_subquery()
        statement = select(InboxCandidateORM).order_by(InboxCandidateORM.id).limit(limit)
        if retriage:
            statement = statement.where(select(InboxTriageResultORM.id).where(
                InboxTriageResultORM.candidate_id == InboxCandidateORM.id,
                InboxTriageResultORM.usage.is_not(None), InboxTriageResultORM.error.is_(None),
            ).exists())
        else:
            statement = statement.where(InboxCandidateORM.status == "new", failures < 3)
        if candidate_id is not None:
            statement = statement.where(InboxCandidateORM.id == candidate_id)
        with self._session_factory() as session:
            if candidate_id is not None and session.get(InboxCandidateORM, candidate_id) is None:
                raise ValueError(f"Candidate {candidate_id} does not exist.")
            return list(session.scalars(statement))

    def daily_triage_count(self, now: datetime) -> int:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include a timezone.")
        start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        with self._session_factory() as session:
            return session.scalar(select(func.count(InboxTriageResultORM.id)).where(
                InboxTriageResultORM.created_at >= start,
                InboxTriageResultORM.created_at < start + timedelta(days=1),
            )) or 0

    def recent_triage_results(
        self, *, limit: int = 20,
    ) -> list[tuple[InboxCandidateORM, InboxTriageResultORM]]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        with self._session_factory() as session:
            return list(session.execute(select(InboxCandidateORM, InboxTriageResultORM).join(
                InboxTriageResultORM, InboxTriageResultORM.candidate_id == InboxCandidateORM.id,
            ).order_by(InboxTriageResultORM.created_at.desc(), InboxTriageResultORM.id.desc()).limit(limit)))

    def list_triage_results(self, candidate_id: int) -> list[InboxTriageResultORM]:
        with self._session_factory() as session:
            return list(session.scalars(select(InboxTriageResultORM).where(
                InboxTriageResultORM.candidate_id == candidate_id
            ).order_by(InboxTriageResultORM.id)))

    def delete_candidate(self, candidate_id: int) -> None:
        with self._session_factory.begin() as session:
            for table in (InboxTriageResultORM, InboxStatusEventORM):
                session.execute(delete(table).where(table.candidate_id == candidate_id))
            session.execute(delete(InboxCandidateORM).where(InboxCandidateORM.id == candidate_id))

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Invalid candidate status: {status}")
