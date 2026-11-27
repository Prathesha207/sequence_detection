export type TrainingJobStatus =
  | 'idle'
  | 'preparing'
  | 'training'
  | 'evaluating'
  | 'completed'
  | 'failed'
  | 'stopped';

export interface TrainingStepItem {
  id: number;
  action_query: string;
  part: string;
  count: number;
  description?: string;
}

export interface TrainingDetectorOption {
  type?: string;
  name: string;
  path: string;
  conf?: number;
  imgsz?: number;
  enabled?: boolean;
  task?: string;
  classes: string[];
  use_for?: string;
}

export interface TrainingHyperparams {
  model_name: string;
  prompt: string;
  base_detector: string;
  output_dir: string;
}

export interface TrainingVideoMetadata {
  durationSeconds: number;
  durationFormatted: string;
  width: number;
  height: number;
  sizeBytes: number;
  sizeFormatted: string;
}

export interface TrainingEpochMetric {
  epoch: number;
  iter?: number;
  label?: string;
  train_loss: number;
  val_loss?: number | null;
  accuracy?: number | null;
  step_accuracy?: number | null;
  learning_rate?: number;
  timestamp: number;
}

export interface TrainingStageInfo {
  id: number;
  key: string;
  title: string;
  description: string;
  status: 'pending' | 'active' | 'done' | 'failed';
  duration_seconds?: number | null;
  estimated_seconds?: number | null;
  sub_progress?: number | null;
}

export interface TrainingJobState {
  job_id: string;
  status: TrainingJobStatus;
  current_epoch: number;
  total_epochs: number;
  progress_percent: number;
  train_loss: number;
  val_loss?: number | null;
  accuracy?: number | null;
  eta_seconds: number;
  stage_eta_seconds?: number | null;
  elapsed_seconds: number;
  logs: string[];
  metrics_history: TrainingEpochMetric[];
  steps?: TrainingStepItem[];
  stages?: TrainingStageInfo[];
  selected_video: string | null;
  model_name: string;
  model_kind?: string;
  fps?: number;
  output_dir?: string;
  prompt?: string;
  stage_name?: string;
  stage_detail?: string;
  self_check?: string;
  error?: string;
  created_at: string;
  completed_at?: string;
}

export interface TrainingRunSummary {
  job_id: string;
  model_name: string;
  epochs: number;
  final_accuracy: number;
  final_loss: number;
  status: TrainingJobStatus;
  created_at: string;
  duration_seconds: number;
  steps_count: number;
}
