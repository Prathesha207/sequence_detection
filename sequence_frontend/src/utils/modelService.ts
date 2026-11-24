import { getApiBaseUrl } from '../lib/api';
import { useInferenceStore } from '../store/inferenceStore';
import { showToast } from '../lib/toast';

export interface ModelStepInfo {
  step: number;
  label: string;
}

/**
 * Fetches and loads procedural sequence steps for a model into store.
 * Call this ONLY after a video is uploaded or camera inference is started.
 */
export async function loadModelSteps(modelName: string): Promise<boolean> {
  if (!modelName) return false;
  try {
    const res = await fetch(`${getApiBaseUrl()}/models/${encodeURIComponent(modelName)}/steps`);
    if (res.ok) {
      const steps: ModelStepInfo[] = await res.json();
      if (Array.isArray(steps) && steps.length > 0) {
        useInferenceStore.getState().setStats({
          active_model: modelName,
          steps: steps.map((s) => ({
            id: s.step,
            step: s.step,
            label: s.label,
            st: 'waiting' as const,
            status: 'waiting',
          })),
          total_steps: steps.length,
          sequence: [],
          doing: null,
          steps_done_count: 0,
          steps_progress: 0,
          verdict: null,
        });
        return true;
      }
    }
  } catch (err) {
    console.warn(`[modelService] Could not fetch steps for ${modelName}:`, err);
  }
  return false;
}

/**
 * Selects an active model in application state and prompts file upload (or loads steps directly if in camera mode).
 */
export async function selectAndLoadModel(
  modelName: string,
  autoPromptUpload: boolean = true,
  isCameraMode: boolean = false
): Promise<boolean> {
  if (!modelName) {
    useInferenceStore.getState().setActiveModel('');
    useInferenceStore.getState().setStats({
      active_model: '',
      steps: [],
      total_steps: 0,
      sequence: [],
      doing: null,
      steps_done_count: 0,
      steps_progress: 0,
      verdict: null,
    });
    return true;
  }

  // Set the selected active model name
  useInferenceStore.getState().setActiveModel(modelName);

  if (isCameraMode) {
    // In camera mode, there is no video to upload — immediately load the procedural steps!
    await loadModelSteps(modelName);
    showToast('success', `SOP Model loaded for camera: ${modelName}`);
    return true;
  }

  // For video mode: keep steps empty until video is uploaded
  useInferenceStore.getState().setStats({
    active_model: modelName,
    steps: [],
    total_steps: 0,
    sequence: [],
    doing: null,
    steps_done_count: 0,
    steps_progress: 0,
    verdict: null,
  });

  showToast('info', `Model selected: ${modelName}. Upload a video to start.`);

  if (autoPromptUpload) {
    // Immediately prompt user to upload video file for this model
    setTimeout(() => {
      useInferenceStore.getState().triggerUploadFn?.();
    }, 250);
  }
  return true;
}

/**
 * Fetches all available trained sequence models from backend.
 */
export async function fetchAvailableModels(): Promise<any[]> {
  try {
    const res = await fetch(`${getApiBaseUrl()}/models`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        useInferenceStore.getState().setAvailableModels(data);
        return data;
      }
    }
  } catch (err) {
    console.warn('[modelService] Could not list models:', err);
  }
  return [];
}
