import { create } from 'zustand';
import axios from 'axios';
import { getApiBaseUrl } from '../lib/api';
import {
  TrainingJobStatus,
  TrainingStepItem,
  TrainingHyperparams,
  TrainingJobState,
  TrainingRunSummary,
  TrainingVideoMetadata,
  TrainingDetectorOption,
} from '../types/trainingTypes';

export type TrainingViewTab = 'monitor' | 'dataset' | 'history';

interface TrainingStoreState {
  appMode: 'inference' | 'training';
  setAppMode: (mode: 'inference' | 'training') => void;

  activeTab: TrainingViewTab;
  setActiveTab: (tab: TrainingViewTab) => void;

  hyperparams: TrainingHyperparams;
  setHyperparams: (params: Partial<TrainingHyperparams>) => void;

  steps: TrainingStepItem[];
  setSteps: (steps: TrainingStepItem[]) => void;
  addStep: (step: Omit<TrainingStepItem, 'id'>) => void;
  updateStep: (id: number, patch: Partial<TrainingStepItem>) => void;
  removeStep: (id: number) => void;

  selectedDatasetVideo: string | null;
  selectedDatasetVideoName: string | null;
  selectedDatasetVideoPath: string | null;
  selectedDatasetFile: File | null;
  videoMetadata: TrainingVideoMetadata | null;
  lastError: string | null;
  setSelectedDatasetVideo: (
    url: string | null,
    name: string | null,
    videoPath?: string | null,
    fileObj?: File | null
  ) => void;
  setVideoMetadata: (meta: TrainingVideoMetadata | null) => void;

  job: TrainingJobState;
  history: TrainingRunSummary[];
  availableDetectors: TrainingDetectorOption[];
  defaultOutputDir: string;
  savedOutputDir: string;
  workspaceDir: string;

  isStarting: boolean;
  isStopping: boolean;

  // Actions
  fetchTrainingConfig: () => Promise<void>;
  saveOutputFolder: (folderPath: string) => Promise<{ ok: boolean; error?: string; workspace_dir?: string }>;
  startTraining: () => Promise<boolean>;
  stopTraining: () => Promise<boolean>;
  fetchStatus: () => Promise<void>;
  fetchHistory: () => Promise<void>;
  deployModel: (jobId: string) => Promise<{ success: boolean; message: string }>;
  resetJob: () => void;
}

const defaultHyperparams: TrainingHyperparams = {
  model_name: '',
  prompt: '',
  base_detector: '',
  output_dir: '',
};

const initialJobState: TrainingJobState = {
  job_id: '',
  status: 'idle',
  current_epoch: 0,
  total_epochs: 0,
  progress_percent: 0,
  train_loss: 0,
  val_loss: null,
  accuracy: null,
  eta_seconds: 0,
  elapsed_seconds: 0,
  logs: ['[System] Training Studio initialized. Ready to train.'],
  metrics_history: [],
  steps: [],
  selected_video: null,
  model_name: '',
  created_at: new Date().toISOString(),
};

export const useTrainingStore = create<TrainingStoreState>((set, get) => ({
  appMode: 'training',
  setAppMode: (mode) => set({ appMode: mode }),

  activeTab: 'monitor',
  setActiveTab: (tab) => set({ activeTab: tab }),

  hyperparams: defaultHyperparams,
  setHyperparams: (params) =>
    set((state) => ({ hyperparams: { ...state.hyperparams, ...params } })),

  steps: [],
  setSteps: (steps) => set({ steps }),
  addStep: (step) =>
    set((state) => {
      const nextId = state.steps.length > 0 ? Math.max(...state.steps.map((s) => s.id)) + 1 : 1;
      return { steps: [...state.steps, { ...step, id: nextId }] };
    }),
  updateStep: (id, patch) =>
    set((state) => ({
      steps: state.steps.map((s) => (s.id === id ? { ...s, ...patch } : s)),
    })),
  removeStep: (id) =>
    set((state) => ({
      steps: state.steps.filter((s) => s.id !== id),
    })),

  selectedDatasetVideo: null,
  selectedDatasetVideoName: null,
  selectedDatasetVideoPath: null,
  selectedDatasetFile: null,
  videoMetadata: null,
  lastError: null,
  setSelectedDatasetVideo: (url, name, videoPath = null, fileObj = null) =>
    set((state) => ({
      selectedDatasetVideo: url,
      selectedDatasetVideoName: name,
      selectedDatasetVideoPath: url ? videoPath : null,
      selectedDatasetFile: url ? fileObj : null,
      videoMetadata: url ? state.videoMetadata : null,
      lastError: null,
    })),
  setVideoMetadata: (meta) => set({ videoMetadata: meta }),

  job: initialJobState,
  history: [],
  availableDetectors: [],
  defaultOutputDir: '',
  savedOutputDir: '',
  workspaceDir: '',
  isStarting: false,
  isStopping: false,

  fetchTrainingConfig: async () => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await axios.get(`${baseUrl}/training/config`);
      if (res.data) {
        const saved = res.data.saved_output_dir || '';
        set((state) => ({
          availableDetectors: Array.isArray(res.data.detectors) ? res.data.detectors : [],
          defaultOutputDir: res.data.default_output_dir || '',
          savedOutputDir: saved,
          workspaceDir: res.data.workspace_dir || '',
          hyperparams: {
            ...state.hyperparams,
            output_dir: state.hyperparams.output_dir || saved,
          },
        }));
      }
    } catch {
      try {
        const baseUrl = getApiBaseUrl();
        const detRes = await axios.get(`${baseUrl}/training/detectors`);
        if (Array.isArray(detRes.data)) {
          set({ availableDetectors: detRes.data });
        }
      } catch {
        // Ignore if backend is offline
      }
    }
  },

  saveOutputFolder: async (folderPath: string) => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await axios.post(`${baseUrl}/training/output-folder`, {
        output_dir: folderPath.trim(),
      });
      if (res.data?.ok) {
        const saved = res.data.saved_output_dir ?? folderPath.trim();
        const ws = res.data.workspace_dir || '';
        set((state) => ({
          savedOutputDir: saved,
          workspaceDir: ws,
          hyperparams: {
            ...state.hyperparams,
            output_dir: saved,
          },
        }));
        return { ok: true, workspace_dir: ws };
      }
      return { ok: false, error: res.data?.error || 'Could not save output folder' };
    } catch (err: any) {
      return {
        ok: false,
        error: err?.response?.data?.detail || err?.message || 'Failed to save output folder',
      };
    }
  },

  resetJob: () => {
    set({ job: initialJobState, steps: [], lastError: null });
    const baseUrl = getApiBaseUrl();
    axios.post(`${baseUrl}/training/reset`).catch(() => {
      // Backend may be offline or already idle
    });
  },

  startTraining: async () => {
    const {
      hyperparams,
      selectedDatasetVideoName,
      selectedDatasetVideoPath,
      selectedDatasetFile,
    } = get();
    set({ isStarting: true, lastError: null });
    try {
      const baseUrl = getApiBaseUrl();
      let resolvedVideoPath = selectedDatasetVideoPath;

      // Upload browser-selected File to backend disk first so train.py --input can read it
      if (!resolvedVideoPath && selectedDatasetFile) {
        const formData = new FormData();
        formData.append('file', selectedDatasetFile);
        const upRes = await axios.post(`${baseUrl}/training/upload-video`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (upRes.data?.video_path) {
          resolvedVideoPath = upRes.data.video_path;
          set({ selectedDatasetVideoPath: resolvedVideoPath });
        }
      }

      const payload = {
        model_name: hyperparams.model_name.trim(),
        prompt: hyperparams.prompt.trim(),
        base_detector: hyperparams.base_detector,
        output_dir: hyperparams.output_dir.trim(),
        dataset_video_name: selectedDatasetVideoName,
        video_path: resolvedVideoPath,
      };

      const res = await axios.post(`${baseUrl}/training/start`, payload);
      if (res.data && res.data.job) {
        const usedOut = hyperparams.output_dir.trim();
        set((state) => ({
          job: res.data.job,
          steps: Array.isArray(res.data.job.steps) ? res.data.job.steps : [],
          savedOutputDir: usedOut || state.savedOutputDir,
          activeTab: 'monitor',
          isStarting: false,
          lastError: null,
        }));
        return true;
      }
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        'Failed to start training pipeline';
      console.error('Failed to start training:', err);
      set({ lastError: msg });
    } finally {
      set({ isStarting: false });
    }
    return false;
  },

  stopTraining: async () => {
    const { job } = get();
    if (!job.job_id) return false;
    set({ isStopping: true });
    try {
      const baseUrl = getApiBaseUrl();
      const res = await axios.post(`${baseUrl}/training/stop`, { job_id: job.job_id });
      if (res.data && res.data.job) {
        set({ job: res.data.job, isStopping: false });
        return true;
      }
    } catch (err) {
      console.error('Failed to stop training:', err);
    } finally {
      set({ isStopping: false });
    }
    return false;
  },

  fetchStatus: async () => {
    const { job } = get();
    try {
      const baseUrl = getApiBaseUrl();
      const url = job.job_id
        ? `${baseUrl}/training/status?job_id=${encodeURIComponent(job.job_id)}`
        : `${baseUrl}/training/status`;
      const res = await axios.get(url);
      if (res.data && res.data.job) {
        set((state) => ({
          job: res.data.job,
          steps: Array.isArray(res.data.job.steps) ? res.data.job.steps : state.steps,
        }));
      }
    } catch {
      // Backend may be offline or starting up
    }
  },

  fetchHistory: async () => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await axios.get(`${baseUrl}/training/history`);
      if (Array.isArray(res.data)) {
        set({ history: res.data });
      }
    } catch {
      // Ignore
    }
  },

  deployModel: async (jobId: string) => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await axios.post(`${baseUrl}/training/deploy`, { job_id: jobId });
      return { success: true, message: res.data?.message || 'Model deployed successfully' };
    } catch (err: any) {
      return { success: false, message: err?.response?.data?.detail || 'Failed to deploy model' };
    }
  },
}));
