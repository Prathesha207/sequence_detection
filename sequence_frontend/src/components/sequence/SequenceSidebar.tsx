import React, { useMemo } from 'react';
import { useInferenceStore } from '../../store/inferenceStore';
import { AnomalyStatus, DetectionMetrics, LogEntry, CameraConfig } from '../../types';
import { Check, AlertCircle, Clock, CheckCircle2, Cpu } from 'lucide-react';
import { CameraImageAdjustmentsCard } from '../CameraImageAdjustmentsCard';

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

  // Prior to active inspection, only display a model if the user has explicitly selected one!
  const modelName = isInspectionActive
    ? (stats.active_model || activeModel || '')
    : (activeModel || '');

  const rawSteps = stats.steps || [];
  const sequence: number[] = Array.isArray(stats.sequence) ? stats.sequence : [];
  const doing = stats.doing;
  const isAnomaly = stats.is_anomaly_frame || stats.status === 'ANOMALY';
  const verdict = stats.verdict;

  // Robustly normalize step status from both backend fields and sequence state
  const stepsToRender = useMemo(() => {
    if (!modelName || rawSteps.length === 0) return [];
    let foundFirstDoing = false;

    return rawSteps.map((s: any, idx: number) => {
      const stepId = Number(s.step || s.id || idx + 1);
      const label = String(s.label || s.action_query || s.text || `Step ${stepId}`);
      const rawStatus = String(s.status || s.st || '').toLowerCase();

      const posInSeq = sequence.indexOf(stepId);
      const isDoneInSeq = posInSeq !== -1;
      const expectedIndex = stepId - 1; // 0-indexed position

      // Out of order detection
      const isOoo = rawStatus === 'ooo' || rawStatus === 'taken_back' || (isDoneInSeq && posInSeq !== expectedIndex);
      const isDone = (rawStatus === 'done' || isDoneInSeq) && !isOoo;

      // Ensure ONLY ONE active in-progress step at any time
      let isDoing = false;
      if (!isDone && !isOoo && !foundFirstDoing) {
        if (
          doing === stepId ||
          (Array.isArray(doing) && doing.includes(stepId)) ||
          rawStatus === 'doing' ||
          (doing == null && sequence.length === expectedIndex && (stats.status === 'RUNNING' || stats.state === 'RUNNING'))
        ) {
          isDoing = true;
          foundFirstDoing = true;
        }
      }

      let st: 'done' | 'doing' | 'ooo' | 'waiting' = 'waiting';
      let note = s.note || '';

      if (isOoo) {
        st = 'ooo';
        if (!note) {
          note = posInSeq < expectedIndex ? 'Done before an earlier step' : 'Done out of sequence order';
        }
      } else if (isDone) {
        st = 'done';
      } else if (isDoing) {
        st = 'doing';
      }

      // Detection confidence and timing
      const atSeconds = s.t != null ? s.t : (s.done_at_frame ? s.done_at_frame / 30 : s.at);
      const tookSeconds = s.took != null ? s.took : (s.duration != null ? s.duration : null);
      const confidence = typeof s.detected === 'number'
        ? s.detected
        : typeof s.confidence === 'number'
          ? Math.round(s.confidence * 100)
          : (isDone ? 99 : null);

      return {
        id: stepId,
        label,
        st,
        at: atSeconds,
        took: tookSeconds,
        det: confidence,
        note,
      };
    });
  }, [modelName, rawSteps, sequence, doing, stats.status, stats.state]);

  const totalSteps = stats.total_steps || stepsToRender.length || 0;
  const trainedOrder = Array.from({ length: totalSteps }, (_, i) => i + 1);

  // Video frames and telemetry
  const totalFrames = Number(stats.total_frames || 0);
  const rawProgress = Number(stats.progress || 0);

  // Compute done count from sequence
  const doneCount = sequence.length > 0
    ? sequence.length
    : stepsToRender.filter((s) => s.st === 'done' || s.st === 'ooo').length;
  const stepPercent = totalSteps > 0 ? Math.min(100, Math.round((doneCount / totalSteps) * 100)) : 0;

  // Determine deviation items for verdict
  const outOfOrderSteps = stepsToRender.filter((s) => s.st === 'ooo');
  const hasAnomaly = isAnomaly || outOfOrderSteps.length > 0 || verdict === 'ANOMALY';
  const isFinished = stats.status === 'completed' || verdict !== null || (doneCount === totalSteps && totalSteps > 0);

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

  // If no model selected, steps not yet loaded, or inference not yet started: render pre-inference placeholder state
  if (!modelName || stepsToRender.length === 0 || !isInspectionActive) {
    return (
      <aside className="w-full lg:w-[350px] xl:w-[360px] flex-shrink-0 h-full max-h-full flex flex-col gap-3 overflow-y-auto pr-1 custom-scrollbar">
        <div className="bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-4 shadow-sm text-[var(--text-primary)] select-none">
          <div className="flex flex-col items-center text-center py-5">
            <div className="w-10 h-10 rounded-xl bg-[var(--accent-main,#7886C7)]/10 border border-[var(--accent-main,#7886C7)]/30 flex items-center justify-center text-[var(--accent-main,#7886C7)] mb-2.5">
              <Cpu className="w-5 h-5 stroke-[1.75]" />
            </div>
            <h3 className="text-xs font-extrabold text-[var(--text-primary)] mb-0.5">
              {modelName ? 'Model Selected' : 'No SOP Model Selected'}
            </h3>
            <p className="text-[11px] text-[var(--text-muted)] max-w-[220px] leading-relaxed">
              {modelName
                ? (isCameraSource
                    ? `${modelName} · Press START INFERENCE to begin live camera inspection.`
                    : `${modelName} · Upload a video to load sequence steps and start inference.`)
                : 'Select an SOP model to begin inference.'}
            </p>
          </div>
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
    <aside className="w-full lg:w-[350px] xl:w-[360px] flex-shrink-0 h-full max-h-full bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-3.5 shadow-sm text-[var(--text-primary)] overflow-y-auto pr-1 custom-scrollbar">

      {/* 1. Compact Header */}
      <header className="flex justify-between items-baseline gap-2 pb-1">
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-extrabold tracking-tight">Sequence</h2>
          <p className="text-[11px] text-[var(--text-muted)] truncate block font-medium" title={modelName}>
            {modelName} · {totalSteps} steps
          </p>
        </div>
        <div className="font-mono text-xs flex items-baseline gap-1 shrink-0">
          <span className="text-sm font-extrabold text-[var(--accent-main,#7886C7)]">{doneCount}</span>
          <span className="text-xs text-[var(--text-muted)] font-normal">of</span>
          <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{totalSteps}</span>
          <span className="text-xs font-bold text-[var(--accent-main,#7886C7)] ml-0.5">({activeBarPercent}%)</span>
        </div>
      </header>

      {/* 2. Compact Total Video Progress Bar in Theme Color */}
      <div className="h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full my-2 overflow-hidden border border-[var(--border-color)]">
        <div
          className="h-full transition-all duration-200"
          style={{
            width: `${activeBarPercent}%`,
            backgroundColor: 'var(--accent-main, #7886C7)',
          }}
        />
      </div>

      {/* 3. Live Telemetry Strip: FPS in theme color, Latency, Frames */}
      <div className="flex items-center justify-between px-2.5 py-1.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-[var(--border-color)] text-[11px] font-mono text-[var(--text-secondary)] mb-3">
        <div className="flex items-center gap-1">
          <span className="text-[var(--text-muted)]">FPS:</span>
          <b className="text-[var(--accent-main,#7886C7)]">{displayFps > 0 ? displayFps.toFixed(1) : '30.0'}</b>
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

      {/* 4. Verdict Box (Shown on completion or active anomaly) */}
      {(isFinished || hasAnomaly) && (
        <section
          className={`mb-3 p-3 rounded-lg border text-xs ${hasAnomaly
            ? 'bg-rose-50 border-rose-200 text-rose-900 dark:bg-rose-950/40 dark:border-rose-900 dark:text-rose-200'
            : 'bg-emerald-50 border-emerald-200 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-900 dark:text-emerald-200'
            }`}
        >
          <div className="flex items-center gap-1.5 font-bold text-sm">
            {hasAnomaly ? (
              <>
                <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0" />
                <span className="text-rose-700 dark:text-rose-300">Sequence Anomaly</span>
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
      <ol className="relative pl-0 my-2 list-none space-y-0.5">
        {stepsToRender.map((s, idx) => {
          const isDone = s.st === 'done';
          const isOoo = s.st === 'ooo';
          const isDoing = s.st === 'doing';
          const isWaiting = s.st === 'waiting';
          const isLast = idx === stepsToRender.length - 1;

          return (
            <li key={s.id} className="flex gap-2.5 relative pb-3 text-xs">

              {/* Vertical Connecting Line */}
              {!isLast && (
                <div
                  className={`absolute left-[11px] top-[24px] bottom-0 w-[2px] transition-colors ${isDone && (stepsToRender[idx + 1].st === 'done' || stepsToRender[idx + 1].st === 'doing')
                    ? 'bg-emerald-500'
                    : stepsToRender[idx + 1].st === 'ooo'
                      ? 'bg-rose-500'
                      : 'bg-slate-200 dark:bg-slate-700'
                    }`}
                />
              )}

              {/* Step Node Icon: Compact w-6 h-6, single yellow circle with center dot */}
              <div
                className={`relative z-10 flex-none w-6 h-6 rounded-full border-2 flex items-center justify-center font-bold text-xs transition-all ${isDone
                  ? 'bg-emerald-600 border-emerald-600 text-white shadow-xs'
                  : isOoo
                    ? 'bg-rose-600 border-rose-600 text-white shadow-xs animate-pulse'
                    : isDoing
                      ? 'border-2 border-amber-500 bg-[var(--bg-card)] text-amber-500'
                      : 'bg-[var(--bg-card)] border-slate-300 dark:border-slate-700 text-slate-400'
                  }`}
              >
                {isDone ? (
                  <Check className="w-3.5 h-3.5 stroke-[2.5]" />
                ) : isOoo ? (
                  <span className="font-extrabold text-[11px]">!</span>
                ) : isDoing ? (
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                ) : (
                  <span className="text-[10px]">{s.id}</span>
                )}
              </div>

              {/* Step Details */}
              <div className="flex-1 min-w-0 pt-0.5">
                <b className={`block text-xs font-bold leading-tight ${isDoing ? 'text-amber-600 dark:text-amber-400' : ''}`}>
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
                  {isDoing && (
                    <span className="text-amber-600 dark:text-amber-400 font-bold flex items-center gap-1">
                      <Clock className="w-3 h-3" /> In progress
                    </span>
                  )}
                  {isOoo && (
                    <span className="text-rose-600 dark:text-rose-400 font-semibold">
                      Done out of order{s.at != null ? ` at ${formatSeconds(s.at)}` : ''}
                      {s.took != null ? ` · took ${formatSeconds(s.took)}` : ''}
                      {s.det != null ? ` · detected ${s.det}%` : ''}
                    </span>
                  )}
                  {isWaiting && <span>Waiting</span>}
                </div>
                {s.note && (
                  <span className="block text-[11px] font-bold text-rose-600 dark:text-rose-400 mt-0.5">
                    {s.note}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      {/* 6. Sequence Order Verification Chips */}
      <div className="pt-2.5 border-t border-[var(--border-color)] mt-1">
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

    </aside>
  );
};

export default SequenceSidebar;
