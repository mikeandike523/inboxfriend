from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Token(Base):
    __tablename__ = "tokens"

    # Use Google account email as primary key (simple for personal tool)
    user_id: Mapped[str] = mapped_column(String(320), primary_key=True)
    access_token: Mapped[str | None] = mapped_column(Text)
    refresh_token: Mapped[str | None] = mapped_column(Text)
    token_type: Mapped[str | None] = mapped_column(String(32))
    scope: Mapped[str | None] = mapped_column(Text)
    expiry: Mapped[datetime | None] = mapped_column(DateTime)
    # Fix: Use timezone-aware datetime
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))