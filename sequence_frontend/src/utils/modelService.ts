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

  let resolvedName = modelName;
  let loadedSteps: ModelStepInfo[] | null = null;
  let loadedOnEngine = true;

  try {
    const res = await fetch(`${getApiBaseUrl()}/models/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      showToast('error', data.detail || `Failed to load model "${modelName}"`);
      return false;
    }
    if (data?.model) resolvedName = data.model;
    if (Array.isArray(data?.steps)) loadedSteps = data.steps;
    loadedOnEngine = data?.loaded !== false;
  } catch (err) {
    console.warn(`[modelService] Failed to sync selected model ${modelName}:`, err);
    showToast('error', `Could not reach backend to load model "${modelName}"`);
    return false;
  }

  // Persist selected active model name in store and refresh model list active flags
  useInferenceStore.getState().setActiveModel(resolvedName);
  void fetchAvailableModels();

  if (isCameraMode) {
    if (loadedSteps && loadedSteps.length > 0) {
      useInferenceStore.getState().setStats({
        active_model: resolvedName,
        steps: loadedSteps.map((s) => ({
          id: s.step,
          step: s.step,
          label: s.label,
          st: 'waiting' as const,
          status: 'waiting',
        })),
        total_steps: loadedSteps.length,
        sequence: [],
        doing: null,
        steps_done_count: 0,
        steps_progress: 0,
        verdict: null,
      });
    } else {
      await loadModelSteps(resolvedName);
    }
    showToast(
      loadedOnEngine ? 'success' : 'info',
      loadedOnEngine
        ? `SOP Model loaded for camera: ${resolvedName}`
        : `SOP Model selected for camera: ${resolvedName}`
    );
    return true;
  }

  // For video mode: keep steps empty until video is uploaded
  useInferenceStore.getState().setStats({
    active_model: resolvedName,
    steps: [],
    total_steps: 0,
    sequence: [],
    doing: null,
    steps_done_count: 0,
    steps_progress: 0,
    verdict: null,
  });

  showToast(
    loadedOnEngine ? 'success' : 'info',
    loadedOnEngine
      ? `Model loaded: ${resolvedName}. Upload a video to start.`
      : `Model selected: ${resolvedName}. Will load when training finishes.`
  );

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

/**
 * Deletes a user-trained workspace model on the backend and refreshes state.
 */
export async function deleteWorkspaceModel(
  modelName: string,
  isCameraMode: boolean = false
): Promise<{ ok: boolean; nextActiveModel?: string | null }> {
  if (!modelName) return { ok: false };
  try {
    const res = await fetch(`${getApiBaseUrl()}/models/${encodeURIComponent(modelName)}`, {
      method: 'DELETE',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      showToast('error', data.detail || `Failed to delete model "${modelName}"`);
      return { ok: false };
    }

    const wasActive = useInferenceStore.getState().activeModel === modelName;
    const nextActive: string | null = data?.next_active_model || null;

    await fetchAvailableModels();
    showToast('success', `Deleted trained model "${modelName}"`);

    if (wasActive) {
      if (nextActive) {
        await selectAndLoadModel(nextActive, false, isCameraMode);
      } else {
        await selectAndLoadModel('', false, isCameraMode);
      }
    }
    return { ok: true, nextActiveModel: nextActive };
  } catch (err) {
    console.warn(`[modelService] Failed to delete model ${modelName}:`, err);
    showToast('error', `Could not reach backend to delete "${modelName}"`);
    return { ok: false };
  }
}
