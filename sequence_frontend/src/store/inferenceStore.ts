import { create } from 'zustand';
import { SOPStep, SOPModel } from '../types';

export interface Detection {
  id: number | string;
  species: string;
  confidence: number;
  isAnomaly: boolean;
  provisional: boolean;
  bbox: [number, number, number, number];
  thumbnail?: string | null;
}

export type SessionLifecycleStatus =
  | 'idle' | 'queued' | 'processing' | 'completed' | 'error' | 'stopped';
export type FrameStatus = 'WARMING' | 'NORMAL' | 'ANOMALY' | 'HAND';

export interface InferenceStats {
  [x: string]: any;
  session_id: string;
  status: SessionLifecycleStatus | FrameStatus | string;
  frames_processed: number;
  total_frames: number;
  progress: number;
  fps: number;
  latency_ms?: number | string | Record<string, any>;
  latency_breakdown?: any;

  // SOP Sequence fields from ML orchestrator
  active_model: string;
  sequence: number[];      // e.g. [1, 3, 2]
  doing: number[] | number | null;
  pending?: Record<string, number>;
  warnings?: Record<string, string>;
  taken?: number[];
  steps: SOPStep[];
  total_steps: number;
  steps_done_count: number;
  steps_progress: number;
  is_anomaly_frame: boolean;
  is_order_warning?: boolean;
  verdict: 'NORMAL' | 'ANOMALY' | 'UNCERTAIN' | null;
  activity_log: Array<{ t: number; text: string }>;

  // Video / Canvas telemetry
  detections: Detection[];
  video_width: number;
  video_height: number;
  original_filename: string | null;

  // Telemetry metadata
  anchor_locked?: boolean;
  hand_detected?: boolean;
  reasons?: string[];
  thumbnails?: any[];
}

interface InferenceStoreState {
  stats: InferenceStats;
  activeModel: string;
  availableModels: SOPModel[];
  isVideoLoading: boolean;
  isRecording: boolean;
  isModelModalOpen: boolean;
  triggerUploadFn: (() => void) | null;
  setActiveModel: (modelName: string) => void;
  setAvailableModels: (models: SOPModel[]) => void;
  setVideoLoading: (loading: boolean) => void;
  setIsRecording: (recording: boolean) => void;
  setIsModelModalOpen: (open: boolean) => void;
  setTriggerUploadFn: (fn: (() => void) | null) => void;
  setStats: (newStats: Partial<InferenceStats>) => void;
  replaceStats: (stats: InferenceStats) => void;
  resetStats: () => void;
}

const initialStats: InferenceStats = {
  session_id: '',
  status: 'idle',
  frames_processed: 0,
  total_frames: 0,
  progress: 0,
  fps: 0,
  latency_ms: 0,

  active_model: '',
  sequence: [],
  doing: [],
  pending: {},
  warnings: {},
  taken: [],
  steps: [],
  total_steps: 0,
  steps_done_count: 0,
  steps_progress: 0,
  is_anomaly_frame: false,
  is_order_warning: false,
  verdict: null,
  activity_log: [],

  detections: [],
  thumbnails: [],
  video_width: 0,
  video_height: 0,
  original_filename: null,

  anchor_locked: true,
  hand_detected: false,
  reasons: [],
};

export const useInferenceStore = create<InferenceStoreState>((set) => ({
  stats: initialStats,
  activeModel: '',
  availableModels: [],
  isVideoLoading: false,
  isRecording: false,
  isModelModalOpen: false,
  triggerUploadFn: null,
  setActiveModel: (modelName) => set({ activeModel: modelName }),
  setAvailableModels: (models) => set({ availableModels: models }),
  setIsRecording: (recording) => set({ isRecording: recording }),
  setVideoLoading: (loading) => set({ isVideoLoading: loading }),
  setIsModelModalOpen: (open) => set({ isModelModalOpen: open }),
  setTriggerUploadFn: (fn) => set({ triggerUploadFn: fn }),
  setStats: (newStats) => set((state) => ({
    stats: {
      ...state.stats,
      ...newStats,
    },
  })),
  replaceStats: (stats) => set({ stats }),
  resetStats: () => set({ stats: initialStats }),
}));
