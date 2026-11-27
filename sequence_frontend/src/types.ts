export type ThemeMode = 'pond-light' | 'pond-dark' | 'nature';

export type StreamSourceType = 'sample-pond' | 'oak-camera' | 'uploaded-video' | 'webcam';

export type StepStatus = 'waiting' | 'doing' | 'confirming' | 'warning' | 'done' | 'ooo' | 'taken_back' | 'missing';

export interface SOPStep {
  id: number;
  label: string;
  st: StepStatus;
  at?: number | null;      // Time in seconds when step finished
  took?: number | null;    // Duration in seconds
  det?: number | null;     // Detection confidence %
  taken?: boolean;         // Part taken (step in progress)
  note?: string;           // Out of order, warning, or deviation explanation
}

export interface SOPModel {
  name: string;
  model_type: string;
  num_steps: number;
  labels: string[];
  is_active: boolean;
  at?: string;
  action?: boolean;
}

export interface DetectionBox {
  id: string | number;
  species: string;
  confidence: number;
  isAnomaly?: boolean;
  x: number; // percentage 0 - 100
  y: number; // percentage 0 - 100
  width: number;
  height: number;
  bbox?: [number, number, number, number];
  provisional?: boolean;
  statusEvent?: string;
  handDetected?: boolean;
}

// Detection entity for workpiece objects on canvas
export type DetectionEntity = DetectionBox;

export type AnomalyType = 'NONE' | 'ANOMALY' | 'OUT_OF_ORDER' | 'DEVIATION' | 'WARMING' | 'UNKNOWN';

export interface AnomalyStatus {
  isAnomaly: boolean;
  type: AnomalyType | string;
  message: string;
  subMessage: string;
  doingStep?: number | null;
  doingLabel?: string;
  verdict?: 'NORMAL' | 'ANOMALY' | null;
  detectedCount?: number;
  expectedCount?: number;
  difference?: number;
  foreignSpecies?: string[];
  foreignCount?: number;
}

export interface CameraConfig {
  id?: number;
  sourceName: string;
  resolution: '1920x1080' | '1280x720';
  targetFps: number;
  recordingFormat?: 'AVI' | 'MP4' | 'FFV1';
  rotationAngle?: number;
  controlMode?: 'auto' | 'manual';
  exposure: number; // 0 - 100
  gain?: number;
  focus?: number;
  brightness: number; // -50 to +50
  contrast: number; // 0 to 100
  iso: number;
  autoFocus: boolean;
  autoExposure?: boolean;
  connected: boolean;
  connectionQuality: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  ipAddress: string;
}

export interface ProcessStep {
  id: string;
  label: string;
  status: 'completed' | 'active' | 'pending' | 'error';
  timestamp?: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  level: 'info' | 'success' | 'warn' | 'error' | 'anomaly';
}

export interface DetectionMetrics {
  fps: number;
  inferenceTimeMs: number;
  framesProcessed: number;
  uptimeSeconds: number;
  avgConfidence: number;
  speciesCounts: Record<string, number>;
}
