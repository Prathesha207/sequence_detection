import React, { useEffect, useMemo, useState } from 'react';
import { useInferenceStore } from '../../store/inferenceStore';
import { AnomalyStatus, DetectionMetrics, LogEntry, CameraConfig } from '../../types';
import { Check, AlertCircle, Clock, CheckCircle2, Cpu, Layers, RefreshCw, Loader2, Trash2, X } from 'lucide-react';
import { CameraImageAdjustmentsCard } from '../CameraImageAdjustmentsCard';
import { deleteWorkspaceModel, fetchAvailableModels, selectAndLoadModel } from '../../utils/modelService';

interface SequenceSidebarProps {
  isOpen?: boolean;
  onToggle?: () => void;
  anomalyStatus?: AnomalyStatus;
  metrics?: DetectionMetrics;
  isStandby?: boolean;
  isRunning?: boolean;
  logs?: LogEntry[];
  hasActiveVideo?: boolean;
  isCameraSource?: boolean;
  isCameraConnected?: boolean;
  cameraConfig?: CameraConfig;
  onUpdateCameraConfig?: (newConfig: Partial<CameraConfig>) => void;
  isRecording?: boolean;
  onShowToast?: (type: 'error' | 'success' | 'info', message: string) => void;
  [key: string]: any;
}

const formatSeconds = (val?: number | null): string => {
  if (val == null || !Number.isFinite(val)) return '';
  return `${val.toFixed(1)} s`;
};

export const SequenceSidebar: React.FC<SequenceSidebarProps> = ({
  metrics,
  hasActiveVideo,
  isCameraSource,
  isCameraConnected,
  cameraConfig,
  onUpdateCameraConfig,
  isRecording,
  onShowToast,
  isRunning = false,
}) => {
  const stats = useInferenceStore((state) => state.stats);
  const activeModel = useInferenceStore((state) => state.activeModel);
  const availableModels = useInferenceStore((state) => state.availableModels);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [selectingModel, setSelectingModel] = useState<string | null>(null);
  const [confirmDeleteModel, setConfirmDeleteModel] = useState<string | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  const framesCount = Number(stats.frames_processed || metrics?.framesProcessed || 0);

  // Inference / inspection activity check:
  // For video: active if video has been loaded or processed
  // For camera: active only when live inference is running or frames processed
  const isVideoLoaded = Boolean(
    hasActiveVideo ||
    stats.session_id ||
    framesCount > 0 ||
    stats.status === 'RUNNING' ||
    stats.status === 'processing' ||
    stats.status === 'completed'
  );

  const isCameraInferenceActive = Boolean(
    isRunning ||
    framesCount > 0 ||
    stats.status === 'RUNNING' ||
    stats.status === 'processing'
  );

  const isInspectionActive = isCameraSource ? isCameraInferenceActive : isVideoLoaded;

  const rawSteps = stats.steps || [];
  const sequence: number[] = Array.isArray(stats.sequence) ? stats.sequence : [];
  const doing = stats.doing;
  const pendingMap: Record<string, number> = (stats.pending && typeof stats.pending === 'object') ? stats.pending : {};
  const warningsMap: Record<string, string> = (stats.warnings && typeof stats.warnings === 'object') ? stats.warnings : {};
  const takenList: number[] = Array.isArray(stats.taken) ? stats.taken.map(Number) : [];
  const isAnomaly = stats.is_anomaly_frame === true || stats.status === 'ANOMALY';
  const isOrderWarning = stats.is_order_warning === true || stats.status === 'ORDER_WARNING' || Object.keys(warningsMap).length > 0;
  const verdict = stats.verdict;

  const modelName = isInspectionActive
    ? (stats.active_model || stats.model_name || activeModel || (rawSteps.length > 0 ? 'Active SOP Model' : ''))
    : (activeModel || '');

  // Map StepTracker lifecycle cleanly (done / confirming / warning / doing / taken_back / missing / waiting)
  const stepsToRender = useMemo(() => {
    if ((!modelName && !isInspectionActive) || rawSteps.length === 0) return [];

    return rawSteps.map((s: any, idx: number) => {
      const stepId = Number(s.step || s.id || idx + 1);
      const label = String(s.label || s.action_query || s.text || `Step ${stepId}`);
      const rawStatus = String(s.status || s.st || '').toLowerCase();

      const posInSeq = sequence.indexOf(stepId);
      const isDoneInSeq = posInSeq !== -1;
      const expectedIndex = stepId - 1;
      const pendingAt = pendingMap[String(stepId)];
      const warningNote = warningsMap[String(stepId)];
      const isInDoing =
        doing === stepId ||
        (Array.isArray(doing) && doing.map(Number).includes(stepId)) ||
        rawStatus === 'doing';
      const isTaken = Boolean(s.taken) || takenList.includes(stepId);

      let st: 'done' | 'confirming' | 'warning' | 'doing' | 'ooo' | 'taken_back' | 'missing' | 'waiting' = 'waiting';
      let note = s.note || '';

      if (rawStatus === 'ooo' || rawStatus === 'out_of_order' || (isDoneInSeq && posInSeq !== expectedIndex && rawStatus !== 'taken_back')) {
        st = 'ooo';
        if (!note) {
          note = isDoneInSeq && posInSeq < expectedIndex
            ? 'Done before an earlier step'
            : 'Done out of sequence order';
        }
      } else if (rawStatus === 'done' || isDoneInSeq) {
        st = 'done';
      } else if (rawStatus === 'confirming' || pendingAt != null) {
        st = 'confirming';
      } else if (rawStatus === 'warning' || Boolean(warningNote)) {
        st = 'warning';
        if (!note && warningNote) note = warningNote;
      } else if (isInDoing) {
        st = 'doing';
      } else if (rawStatus === 'taken_back') {
        st = 'taken_back';
        if (!note) note = 'Removed / taken back before completion';
      } else if (rawStatus === 'missing') {
        st = 'missing';
      }

      // Detection confidence and timing (only report real ML numbers)
      const atSeconds = s.t != null
        ? s.t
        : (pendingAt != null ? pendingAt : (s.done_at_frame ? s.done_at_frame / 30 : s.at));
      const tookSeconds = s.took != null ? s.took : (s.duration != null ? s.duration : null);
      const confidence = typeof s.detected === 'number'
        ? s.detected
        : typeof s.confidence === 'number'
          ? Math.round(s.confidence * 100)
          : null;

      return {
        id: stepId,
        label,
        st,
        taken: isTaken,
        at: atSeconds,
        took: tookSeconds,
        det: confidence,
        note,
      };
    });
  }, [modelName, rawSteps, sequence, doing, pendingMap, warningsMap, takenList]);

  const totalSteps = stats.total_steps || stepsToRender.length || 0;
  const trainedOrder = Array.from({ length: totalSteps }, (_, i) => i + 1);

  // Video frames and telemetry
  const totalFrames = Number(stats.total_frames || 0);
  const rawProgress = Number(stats.progress || 0);

  // Compute done count from sequence (tr.shown)
  const doneCount = sequence.length > 0
    ? sequence.length
    : stepsToRender.filter((s) => s.st === 'done' || s.st === 'ooo').length;
  const stepPercent = totalSteps > 0 ? Math.min(100, Math.round((doneCount / totalSteps) * 100)) : 0;

  // Live & verdict flags aligned with infer.py live_state()
  const outOfOrderSteps = stepsToRender.filter((s) => s.st === 'ooo');
  const warningSteps = stepsToRender.filter((s) => s.st === 'warning');
  const hasAnomaly = verdict === 'ANOMALY' || (verdict !== 'NORMAL' && (isAnomaly || outOfOrderSteps.length > 0));
  const hasProvisionalWarning = !hasAnomaly && (isOrderWarning || warningSteps.length > 0);
  const isFinished = stats.status === 'completed' || Boolean(stats.final) || verdict !== null || (doneCount === totalSteps && totalSteps > 0);

  // Total video playback / inference progress (never frozen at 0% during video playback)
  const videoProgress = isFinished
    ? 100
    : totalFrames > 0
      ? Math.min(100, Math.max(0, Math.round((framesCount / totalFrames) * 100)))
      : Math.min(100, Math.max(0, Math.round(rawProgress)));
  const activeBarPercent = isCameraSource ? (isFinished ? 100 : stepPercent) : Math.max(stepPercent, videoProgress);

  const displayFps = Number(stats.fps || metrics?.fps || 0);
  const rawLatency = stats.latency_ms || metrics?.inferenceTimeMs;
  const displayLatency = Math.round(Number(rawLatency) || (displayFps > 0 ? 1000 / displayFps : 32));

  const isConnected = Boolean(isCameraConnected);

  // Load workspace trained models list for the pre-inference model switcher card
  const refreshModelsList = async () => {
    setIsLoadingModels(true);
    try {
      await fetchAvailableModels();
    } finally {
      setIsLoadingModels(false);
    }
  };

  useEffect(() => {
    if (!isInspectionActive) {
      void refreshModelsList();
    }
  }, [isInspectionActive]);

  const handleSelectTrainedModel = async (targetName: string) => {
    if (!targetName || selectingModel || deletingModel) return;
    setConfirmDeleteModel(null);
    setSelectingModel(targetName);
    try {
      await selectAndLoadModel(targetName, false, Boolean(isCameraSource));
    } finally {
      setSelectingModel(null);
    }
  };

  const handleDeleteTrainedModel = async (targetName: string) => {
    if (!targetName || deletingModel || selectingModel) return;
    setDeletingModel(targetName);
    try {
      await deleteWorkspaceModel(targetName, Boolean(isCameraSource));
    } finally {
      setDeletingModel(null);
      setConfirmDeleteModel(null);
    }
  };

  // If no model selected, steps not yet loaded, or inference not yet started: render pre-inference placeholder state
  if (!modelName || stepsToRender.length === 0 || !isInspectionActive) {
    return (
      <aside className="w-full lg:w-[350px] xl:w-[360px] flex-shrink-0 self-start max-h-full flex flex-col gap-3 overflow-y-auto custom-scrollbar">
        {/* Unified Model Selected + Trained / Pre-trained Models Switcher Card */}
        <div className="bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-4 shadow-sm text-[var(--text-primary)] flex flex-col gap-3">
          {/* Active Model Summary */}
          <div className="flex items-start justify-between gap-2.5">
            <div className="flex items-start gap-2.5 min-w-0">
              <div className="w-9 h-9 rounded-lg bg-[var(--accent-main)]/10 border border-[var(--accent-main)]/30 flex items-center justify-center text-[var(--accent-main)] shrink-0 mt-0.5">
                <Cpu className="w-4 h-4 stroke-[1.75]" />
              </div>
              <div className="min-w-0">
                <h3 className="text-xs font-extrabold text-[var(--text-primary)] truncate">
                  {modelName ? `Model Selected: ${modelName}` : 'No SOP Model Selected'}
                </h3>
                <p className="text-[11px] text-[var(--text-muted)] leading-snug mt-0.5">
                  {modelName
                    ? (isCameraSource
                        ? 'Press START INFERENCE to begin live camera inspection.'
                        : 'Upload a video to load sequence steps and start inference.')
                    : 'Choose a trained or pre-trained model below to begin.'}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => void refreshModelsList()}
              disabled={isLoadingModels}
              title="Refresh models list"
              className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card-subtle)] transition-colors disabled:opacity-50 shrink-0"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoadingModels ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Models Section Header */}
          <div className="flex items-center justify-between gap-2 pt-2.5 border-t border-[var(--border-color)]">
            <div className="flex items-center gap-1.5 min-w-0">
              <Layers className="w-3.5 h-3.5 text-[var(--accent-main)] shrink-0" />
              <span className="text-[11px] font-extrabold uppercase tracking-wider text-[var(--text-primary)] truncate">
                Available Models
              </span>
            </div>
            <span className="px-1.5 py-0.5 rounded bg-[var(--bg-card-subtle)] border border-[var(--border-color)] text-[10px] font-bold text-[var(--text-muted)] tabular-nums">
              {availableModels.length}
            </span>
          </div>

          {isLoadingModels && availableModels.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-5 text-[11px] text-[var(--text-muted)]">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent-main)]" />
              <span>Loading trained models...</span>
            </div>
          ) : availableModels.length === 0 ? (
            <div className="py-5 px-3 text-center text-[11px] text-[var(--text-muted)] bg-[var(--bg-card-subtle)]/50 border border-dashed border-[var(--border-color)] rounded-lg">
              No trained models found.
            </div>
          ) : (
            <div className="flex flex-col gap-1.5 max-h-[208px] overflow-y-auto custom-scrollbar pr-0.5">
              {availableModels.map((rawModel) => {
                const m = rawModel as Record<string, any>;
                const mName = String(m.name || '');
                const stepsCount = Number(m.steps ?? m.num_steps ?? 0);
                const trainedAt = typeof m.trained_at === 'string' ? m.trained_at : '';
                const isPretrained = Boolean(m.is_pretrained || m.source === 'wheel');
                const hasError = Boolean(m.error);
                const isSelected = modelName ? mName === modelName : Boolean(m.active);
                const isSwitchingThis = selectingModel === mName;
                const isConfirmingDelete = confirmDeleteModel === mName;
                const isDeletingThis = deletingModel === mName;
                const isBusy = selectingModel !== null || deletingModel !== null;
                const canSelect = !hasError && !isBusy && !isConfirmingDelete;

                return (
                  <div
                    key={`${m.source || 'model'}-${mName}`}
                    role="button"
                    tabIndex={canSelect ? 0 : -1}
                    aria-disabled={!canSelect}
                    onClick={() => {
                      if (canSelect) void handleSelectTrainedModel(mName);
                    }}
                    onKeyDown={(e) => {
                      if (canSelect && (e.key === 'Enter' || e.key === ' ')) {
                        e.preventDefault();
                        void handleSelectTrainedModel(mName);
                      }
                    }}
                    className={`group w-full text-left px-3 py-2.5 rounded-lg border transition-all flex items-center justify-between gap-2 ${
                      hasError
                        ? 'bg-[var(--bg-card-subtle)]/40 border-[var(--border-color)] opacity-75'
                        : isSelected
                          ? 'bg-[var(--accent-main)]/8 border-[var(--accent-main)]/50 shadow-xs cursor-pointer'
                          : 'bg-[var(--bg-card-subtle)]/60 border-[var(--border-color)] hover:border-[var(--accent-main)]/40 hover:bg-[var(--bg-card-subtle)] cursor-pointer'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span
                          className={`w-2 h-2 rounded-sm shrink-0 ${
                            hasError
                              ? 'bg-[var(--status-anomaly-text)]'
                              : isSelected
                                ? 'bg-[var(--accent-main)]'
                                : 'bg-[var(--border-color)] group-hover:bg-[var(--accent-main)]/60'
                          }`}
                        />
                        <span className="text-xs font-bold text-[var(--text-primary)] truncate">
                          {mName}
                        </span>
                        {isPretrained && (
                          <span className="px-1.5 py-0.5 rounded bg-[var(--bg-card)] border border-[var(--border-color)] text-[9px] font-bold uppercase tracking-wide text-[var(--text-muted)] shrink-0">
                            Pre-trained
                          </span>
                        )}
                      </div>
                      <div className="mt-1 pl-3.5 flex items-center gap-2 text-[10px] text-[var(--text-muted)] truncate">
                        {hasError ? (
                          <span className="text-[var(--status-anomaly-text)] truncate">{String(m.error)}</span>
                        ) : (
                          <>
                            <span className="font-semibold text-[var(--text-secondary)]">
                              {stepsCount} {stepsCount === 1 ? 'step' : 'steps'}
                            </span>
                            {trainedAt && (
                              <>
                                <span>·</span>
                                <span className="truncate">{trainedAt}</span>
                              </>
                            )}
                          </>
                        )}
                      </div>
                    </div>

                    <div className="shrink-0 flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                      {isDeletingThis ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-500/10 border border-rose-500/30 text-[10px] font-bold text-rose-600 dark:text-rose-400">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          Deleting
                        </span>
                      ) : isConfirmingDelete ? (
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            disabled={isBusy}
                            onClick={() => void handleDeleteTrainedModel(mName)}
                            className="px-2 py-0.5 rounded-md bg-rose-600 hover:bg-rose-700 text-white text-[10px] font-bold transition-colors disabled:opacity-50"
                          >
                            Delete
                          </button>
                          <button
                            type="button"
                            disabled={isBusy}
                            onClick={() => setConfirmDeleteModel(null)}
                            title="Cancel delete"
                            className="p-1 rounded-md bg-[var(--bg-card)] border border-[var(--border-color)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ) : (
                        <>
                          {isSwitchingThis ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[var(--accent-main)]/15 border border-[var(--accent-main)]/30 text-[10px] font-bold text-[var(--accent-main)]">
                              <Loader2 className="w-3 h-3 animate-spin" />
                              Loading
                            </span>
                          ) : isSelected ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[var(--accent-main)]/15 border border-[var(--accent-main)]/30 text-[10px] font-bold text-[var(--accent-main)]">
                              <Check className="w-3 h-3 stroke-[2.5]" />
                              Active
                            </span>
                          ) : !hasError ? (
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() => void handleSelectTrainedModel(mName)}
                              className="px-2 py-0.5 rounded-md bg-[var(--bg-card)] border border-[var(--border-color)] text-[10px] font-bold text-[var(--text-muted)] group-hover:text-[var(--accent-main)] group-hover:border-[var(--accent-main)]/40 transition-colors disabled:opacity-50"
                            >
                              Load
                            </button>
                          ) : null}

                          {!isPretrained && (
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() => setConfirmDeleteModel(mName)}
                              title={`Delete trained model "${mName}"`}
                              className="p-1 rounded-md text-[var(--text-muted)] hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/25 transition-colors disabled:opacity-40"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Camera Adjustments Card: shown in camera mode when connected and not recording */}
        {isCameraSource && isConnected && !isRecording && (
          <CameraImageAdjustmentsCard
            cameraId={cameraConfig?.id}
            config={cameraConfig}
            onUpdateConfig={onUpdateCameraConfig}
            isLive={isConnected}
            isRecording={isRecording}
            onShowToast={onShowToast}
          />
        )}
      </aside>
    );
  }

  return (
    <aside className="w-full lg:w-[350px] xl:w-[360px] flex-shrink-0 self-start max-h-full flex flex-col min-h-0">
      <div className="bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-3.5 shadow-sm text-[var(--text-primary)] flex flex-col max-h-full min-h-0 overflow-hidden">

      {/* 1. Compact Header */}
      <header className="flex justify-between items-baseline gap-2 pb-1 shrink-0">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 min-w-0">
            <h2 className="text-base font-extrabold tracking-tight">Sequence</h2>
            {stats.state && stats.state !== 'IDLE' && (
              <span
                className={`px-1.5 py-0.5 rounded text-[9.5px] font-bold uppercase tracking-wider border shrink-0 ${
                  hasAnomaly
                    ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/30'
                    : hasProvisionalWarning
                      ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30'
                      : String(stats.state).toUpperCase() === 'STEP IN PROGRESS'
                        ? 'bg-[var(--accent-main)]/12 text-[var(--accent-main)] border-[var(--accent-main)]/30'
                        : 'bg-[var(--bg-card-subtle)] text-[var(--text-muted)] border-[var(--border-color)]'
                }`}
              >
                {String(stats.state)}
              </span>
            )}
          </div>
          <p className="text-[11px] text-[var(--text-muted)] truncate block font-medium" title={modelName}>
            {modelName} · {totalSteps} steps
          </p>
        </div>
        <div className="font-mono text-xs flex items-baseline gap-1 shrink-0">
          <span className="text-sm font-extrabold text-[var(--accent-main)]">{doneCount}</span>
          <span className="text-xs text-[var(--text-muted)] font-normal">of</span>
          <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{totalSteps}</span>
          <span className="text-xs font-bold text-[var(--accent-main)] ml-0.5">({activeBarPercent}%)</span>
        </div>
      </header>

      {/* 2. Compact Total Video Progress Bar in Theme Color */}
      <div className="h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full my-2 overflow-hidden border border-[var(--border-color)] shrink-0">
        <div
          className="h-full transition-all duration-200"
          style={{
            width: `${activeBarPercent}%`,
            backgroundColor: 'var(--accent-main)',
          }}
        />
      </div>

      {/* 3. Live Telemetry Strip: FPS in theme color, Latency, Frames */}
      <div className="flex items-center justify-between px-2.5 py-1.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-[var(--border-color)] text-[11px] font-mono text-[var(--text-secondary)] mb-3 shrink-0">
        <div className="flex items-center gap-1">
          <span className="text-[var(--text-muted)]">FPS:</span>
          <b className="text-[var(--accent-main)]">{displayFps > 0 ? displayFps.toFixed(1) : '30.0'}</b>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[var(--text-muted)]">Latency:</span>
          <b>{displayLatency}ms</b>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[var(--text-muted)]">Frames:</span>
          <b>#{framesCount}</b>
        </div>
      </div>

      {/* 4. Verdict / Live Status Box */}
      {(isFinished || hasAnomaly || hasProvisionalWarning) && (
        <section
          className={`mb-3 p-3 rounded-lg border text-xs shrink-0 ${hasAnomaly
            ? 'bg-rose-50 border-rose-200 text-rose-900 dark:bg-rose-950/40 dark:border-rose-900 dark:text-rose-200'
            : hasProvisionalWarning
              ? 'bg-amber-50 border-amber-200 text-amber-900 dark:bg-amber-950/40 dark:border-amber-900 dark:text-amber-200'
              : 'bg-emerald-50 border-emerald-200 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-200'
            }`}
        >
          <div className="flex items-center gap-1.5 font-bold text-sm">
            {hasAnomaly ? (
              <>
                <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0" />
                <span className="text-rose-700 dark:text-rose-300">Sequence Anomaly</span>
              </>
            ) : hasProvisionalWarning ? (
              <>
                <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
                <span className="text-amber-700 dark:text-amber-300">Order Warning?</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                <span className="text-emerald-700 dark:text-emerald-300">Normal</span>
              </>
            )}
          </div>
          <p className="mt-1 text-[11px] leading-tight">
            {hasAnomaly
              ? 'The steps did not follow the trained sequence.'
              : hasProvisionalWarning
                ? 'Step started out of order — waiting to confirm or complete earlier steps.'
                : 'The steps followed the trained sequence.'}
          </p>
          <div className="mt-1.5 text-[10px] font-mono opacity-90">
            Order done: {sequence.length > 0 ? sequence.join(' → ') : 'none'}
            {stats.t != null && stats.t > 0 && ` · ${stats.t.toFixed(1)} s`}
            {framesCount > 0 && ` · ${framesCount} frames`}
          </div>
        </section>
      )}

      {/* 5. Connected Steps Timeline */}
      <ol className="relative pl-0 my-1.5 list-none space-y-0.5 flex-1 min-h-0 max-h-[290px] overflow-y-auto custom-scrollbar pr-1">
        {stepsToRender.map((s, idx) => {
          const isDone = s.st === 'done';
          const isOoo = s.st === 'ooo';
          const isTakenBack = s.st === 'taken_back';
          const isWarning = s.st === 'warning';
          const isConfirming = s.st === 'confirming';
          const isDoing = s.st === 'doing';
          const isMissing = s.st === 'missing';
          const isWaiting = s.st === 'waiting';
          const isLast = idx === stepsToRender.length - 1;
          const nextSt = !isLast ? stepsToRender[idx + 1].st : '';

          return (
            <li key={s.id} className="flex gap-2.5 relative pb-3 text-xs">

              {/* Vertical Connecting Line */}
              {!isLast && (
                <div
                  className={`absolute left-[11px] top-[24px] bottom-0 w-[2px] transition-colors ${isDone && (nextSt === 'done' || nextSt === 'doing' || nextSt === 'confirming')
                    ? 'bg-emerald-500'
                    : (nextSt === 'ooo' || nextSt === 'taken_back')
                      ? 'bg-rose-500'
                      : (nextSt === 'warning')
                        ? 'bg-amber-500'
                        : 'bg-slate-200 dark:bg-slate-700'
                    }`}
                />
              )}

              {/* Step Node Icon */}
              <div
                className={`relative z-10 flex-none w-6 h-6 rounded-full border-2 flex items-center justify-center font-bold text-xs transition-all ${isDone
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-xs'
                  : (isOoo || isTakenBack)
                    ? 'bg-rose-600 border-rose-600 text-white shadow-xs animate-pulse'
                    : isWarning
                      ? 'bg-amber-500 border-amber-500 text-white shadow-xs'
                      : isConfirming
                        ? 'border-2 border-[var(--accent-main)] bg-[var(--accent-main)]/10 text-[var(--accent-main)]'
                        : isDoing
                          ? 'border-2 border-amber-500 bg-[var(--bg-card)] text-amber-500'
                          : 'bg-[var(--bg-card)] border-slate-300 dark:border-slate-700 text-slate-400'
                  }`}
              >
                {isDone ? (
                  <Check className="w-3.5 h-3.5 stroke-[2.5]" />
                ) : (isOoo || isTakenBack || isWarning) ? (
                  <span className="font-extrabold text-[11px]">!</span>
                ) : isConfirming ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : isDoing ? (
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-ping" />
                ) : isMissing ? (
                  <span className="text-[10px]">-</span>
                ) : (
                  <span className="text-[10px]">{s.id}</span>
                )}
              </div>

              {/* Step Details */}
              <div className="flex-1 min-w-0 pt-0.5">
                <b className={`block text-xs font-bold leading-tight ${isConfirming ? 'text-[var(--accent-main)]' : (isDoing || isWarning) ? 'text-amber-600 dark:text-amber-400' : isMissing ? 'text-[var(--text-muted)]' : ''}`}>
                  {s.label}
                </b>
                <div className="text-[11px] text-[var(--text-muted)] mt-0.5 flex flex-wrap items-center gap-1">
                  {isDone && (
                    <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
                      Done{s.at != null ? ` at ${formatSeconds(s.at)}` : ''}
                      {s.took != null ? ` · took ${formatSeconds(s.took)}` : ''}
                      {s.det != null ? ` · detected ${s.det}%` : ''}
                    </span>
                  )}
                  {isConfirming && (
                    <span className="text-[var(--accent-main)] font-bold flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Confirming{s.at != null ? ` (${formatSeconds(s.at)})` : '...'}
                    </span>
                  )}
                  {isDoing && (
                    <span className="text-amber-600 dark:text-amber-400 font-bold flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {s.taken ? 'Taken · In progress' : 'In progress'}
                    </span>
                  )}
                  {isWarning && (
                    <span className="text-amber-600 dark:text-amber-400 font-bold">
                      Taken out of order?
                    </span>
                  )}
                  {isOoo && (
                    <span className="text-rose-600 dark:text-rose-400 font-semibold">
                      Done out of order
                      {s.at != null ? ` at ${formatSeconds(s.at)}` : ''}
                      {s.took != null ? ` · took ${formatSeconds(s.took)}` : ''}
                      {s.det != null ? ` · detected ${s.det}%` : ''}
                    </span>
                  )}
                  {isTakenBack && (
                    <span className="text-rose-600 dark:text-rose-400 font-semibold">
                      Taken back / undone
                    </span>
                  )}
                  {isMissing && <span>Not completed</span>}
                  {isWaiting && <span>Waiting</span>}
                </div>
                {s.note && (
                  <span className={`block text-[11px] font-bold mt-0.5 ${isWarning ? 'text-amber-600 dark:text-amber-400' : 'text-rose-600 dark:text-rose-400'}`}>
                    {s.note}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      {/* 6. Sequence Order Verification Chips */}
      <div className="pt-2.5 border-t border-[var(--border-color)] mt-1 shrink-0">
        <h3 className="text-[10px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
          Sequence Order Verification
        </h3>

        {/* Trained Order Row */}
        <div className="flex items-center gap-2.5 text-xs mb-1.5">
          <span className="w-12 text-[var(--text-muted)] font-semibold text-[11px]">Trained</span>
          <div className="flex gap-1.5 flex-wrap">
            {trainedOrder.map((stepNum) => (
              <span
                key={stepNum}
                className="w-6 h-6 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 flex items-center justify-center font-bold text-[11px]"
              >
                {stepNum}
              </span>
            ))}
          </div>
        </div>

        {/* Done Order Row */}
        <div className="flex items-center gap-2.5 text-xs">
          <span className="w-12 text-[var(--text-muted)] font-semibold text-[11px]">Done</span>
          <div className="flex gap-1.5 flex-wrap">
            {sequence.length > 0 ? (
              sequence.map((stepNum, idx) => {
                const isMatching = trainedOrder[idx] === stepNum;
                return (
                  <span
                    key={idx}
                    className={`w-6 h-6 rounded-md flex items-center justify-center font-bold text-[11px] border ${isMatching
                      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-300 dark:border-emerald-800'
                      : 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300 border-rose-300 dark:border-rose-800 animate-pulse'
                      }`}
                  >
                    {stepNum}
                  </span>
                );
              })
            ) : (
              <span className="text-slate-400 text-xs italic">none yet</span>
            )}
          </div>
        </div>
      </div>

      </div>
    </aside>
  );
};

export default SequenceSidebar;
