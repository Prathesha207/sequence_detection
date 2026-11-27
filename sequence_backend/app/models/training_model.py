from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from datetime import datetime
from app.core.database import Base


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    model_name = Column(String(255), nullable=False)
    prompt = Column(Text, default="")
    dataset_video_name = Column(String(255), nullable=True)
    base_detector = Column(String(255), default="")
    epochs = Column(Integer, default=30)
    batch_size = Column(Integer, default=16)
    learning_rate = Column(Float, default=0.001)
    device = Column(String(64), default="cuda:0")
    status = Column(String(32), default="idle")
    current_epoch = Column(Integer, default=0)
    progress_percent = Column(Integer, default=0)
    train_loss = Column(Float, default=0.0)
    val_loss = Column(Float, default=0.0)
    accuracy = Column(Float, default=0.0)
    eta_seconds = Column(Integer, default=0)
    elapsed_seconds = Column(Integer, default=0)
    steps_json = Column(Text, default="[]")
    logs_json = Column(Text, default="[]")
    metrics_json = Column(Text, default="[]")
    output_dir = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class TrainingSetting(Base):
    __tablename__ = "training_settings"

    key = Column(String(64), primary_key=True, index=True)
    value = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
