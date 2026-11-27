import React, { useEffect, useRef, useState } from 'react';
import type { StreamSourceType, CameraConfig } from '../../types';
import { useTrainingStore } from '../../store/trainingStore';
import { VideoUploadCard } from '../canvas/VideoUploadCard';
import { CameraOfflineCard } from '../canvas/CameraOfflineCard';
import { CameraStandbyCard } from '../canvas/CameraStandbyCard';
import { CameraLeftToolbar } from '../canvas/CameraLeftToolbar';
import { TrainingVideoCard } from './TrainingVideoCard';
import { TrainingDetailsCard } from './TrainingDetailsCard';
import { TrainingTelemetryView } from './TrainingTelemetryView';
import { getApiBaseUrl } from '../../lib/api';

interface TrainingWorkspaceProps {
  sourceType: StreamSourceType;
  isCameraConnected?: boolean;
  isStreaming?: boolean;
  cameraError?: string | null;
  cameraConfig?: CameraConfig;
  isRecording?: boolean;
  isSavingRecording?: boolean;
  recordingDuration?: number;
  onToggleRecording?: () => void;
  onStartStream?: () => void;
  onStopStream?: () => void;
  onRetryConnection?: () => void;
  onRequestSwitchMode?: (type: StreamSourceType) => void;
  onDeployToInference?: (modelName: string) => void;
  onShowToast?: (type: 'success' | 'error' | 'info' | 'warning', message: string) => void;
}

const formatDuration = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds <= 0) return '00:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const formatFileSize = (bytes: number): string => {
  if (!bytes || bytes <= 0) return '0 MB';
  const mb = bytes / (1024 * 1024);
  if (mb < 1) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${mb.toFixed(1)} MB`;
};

export const TrainingWorkspace: React.FC<TrainingWorkspaceProps> = ({
  sourceType,
  isCameraConnected = false,
  isStreaming = false,
  cameraError,
  cameraConfig,
  isRecording = false,
  isSavingRecording = false,
  recordingDuration = 0,
  onToggleRecording,
  onStartStream,
  onStopStream,
  onRetryConnection,
  onRequestSwitchMode,
  onDeployToInference,
  onShowToast,
}) => {
  const {
    job,
    hyperparams,
    setHyperparams,
    startTraining,
    stopTraining,
    isStarting,
    fetchStatus,
    fetchHistory,
    fetchTrainingConfig,
    saveOutputFolder,
    availableDetectors,
    defaultOutputDir,
    savedOutputDir,
    workspaceDir,
    deployModel,
    resetJob,
    selectedDatasetVideo,
    selectedDatasetVideoName,
    videoMetadata,
    setSelectedDatasetVideo,
    setVideoMetadata,
  } = useTrainingStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [streamCacheBuster, setStreamCacheBuster] = useState<number>(Date.now());
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (isStreaming) {
      setStreamCacheBuster(Date.now());
    }
  }, [isStreaming]);

  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    };
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchHistory();
    fetchTrainingConfig();
  }, [fetchStatus, fetchHistory, fetchTrainingConfig]);

  useEffect(() => {
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
      fetchHistory();
      return;
    }
    if (job.status !== 'training' && job.status !== 'preparing' && job.status !== 'evaluating') {
      return;
    }
    const interval = setInterval(() => {
      fetchStatus();
    }, 1000);
    return () => clearInterval(interval);
  }, [job.status, fetchStatus, fetchHistory]);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSelectedDatasetVideo(url, file.name, null, file);
      setVideoMetadata({
        durationSeconds: 0,
        durationFormatted: '--:--',
        width: 0,
        height: 0,
        sizeBytes: file.size,
        sizeFormatted: formatFileSize(file.size),
      });
      onShowToast?.('success', `Selected training clip: ${file.name}`);
    }
    e.target.value = '';
  };

  const handleVideoMetadataLoaded = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const videoEl = e.currentTarget;
    setVideoMetadata({
      durationSeconds: videoEl.duration || 0,
      durationFormatted: formatDuration(videoEl.duration || 0),
      width: videoEl.videoWidth || 0,
      height: videoEl.videoHeight || 0,
      sizeBytes: videoMetadata?.sizeBytes || 0,
      sizeFormatted: videoMetadata?.sizeFormatted || '-- MB',
    });
  };

  const handleStartFromCard = async () => {
    if (!hyperparams.model_name.trim()) {
      onShowToast?.('warning', 'Please enter a Training Name');
      return;
    }
    if (!hyperparams.prompt.trim()) {
      onShowToast?.('warning', 'Please enter a Training Prompt');
      return;
    }
    const ok = await startTraining();
    if (ok) {
      onShowToast?.('success', `Started training for '${hyperparams.model_name.trim()}'`);
    } else {
      const errMsg = useTrainingStore.getState().lastError;
      onShowToast?.('error', errMsg || 'Failed to start training pipeline');
    }
  };

  const handleDeploy = async () => {
    if (!job.job_id) return;
    const res = await deployModel(job.job_id);
    if (res.success) {
      onShowToast?.('success', `Deployed '${job.model_name}' as active SOP model!`);
      onDeployToInference?.(job.model_name);
    } else {
      onShowToast?.('error', res.message);
    }
  };

  const isRunning =
    job.status === 'training' || job.status === 'preparing' || job.status === 'evaluating';
  const showTrainingTelemetry =
    isRunning || job.status === 'completed' || job.status === 'failed';

  if (showTrainingTelemetry) {
    return (
      <TrainingTelemetryView
        job={job}
        fallbackModelName={hyperparams.model_name}
        isRunning={isRunning}
        onStop={stopTraining}
        onReset={() => resetJob()}
        onNewTraining={() => {
          setSelectedDatasetVideo(null, null);
          setHyperparams({ model_name: '', prompt: '' });
          resetJob();
        }}
        onDeploy={handleDeploy}
      />
    );
  }

  const isVideoSource = sourceType === 'uploaded-video' || sourceType === 'sample-pond';
  const isCameraSource = sourceType === 'oak-camera' || sourceType === 'webcam';
  const showCardBg =
    isVideoSource ||
    Boolean(selectedDatasetVideo) ||
    (isCameraSource && (!isCameraConnected || !isStreaming));

  return (
    <div
      className="relative w-full flex-1 h-full min-h-[350px] lg:min-h-0 overflow-hidden border border-[var(--border-color)] shadow-sm select-none group rounded-3xl transition-colors duration-200"
      style={{ backgroundColor: showCardBg ? 'var(--bg-card)' : '#000000' }}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleVideoUpload}
        accept="video/*"
        className="hidden"
      />

      {/* Video Mode Step 1: Upload Card */}
      {isVideoSource && !selectedDatasetVideo && (
        <VideoUploadCard
          uploadProgress={null}
          isSelectingVideo={false}
          title="Upload Video for Training"
          subtitle={
            <>
              Click here to upload a training video, then enter your{' '}
              <span className="font-bold text-[var(--accent-main)]">Training Name &amp; Prompt</span>.
            </>
          }
          onCardClick={() => fileInputRef.current?.click()}
        />
      )}

      {/* Selected Video (Uploaded OR Recorded from OAK Camera) + Training Details */}
      {selectedDatasetVideo && (
        <div className="absolute inset-0 w-full h-full p-4 sm:p-5 z-10 overflow-y-auto custom-scrollbar flex flex-col rounded-3xl bg-[var(--bg-card)]">
          <div className="flex flex-col lg:flex-row gap-4 sm:gap-5 w-full flex-1 min-h-0 items-stretch">
            <TrainingVideoCard
              videoUrl={selectedDatasetVideo}
              videoName={selectedDatasetVideoName}
              metadata={videoMetadata}
              onMetadataLoaded={handleVideoMetadataLoaded}
              replaceLabel={isCameraSource ? 'Re-record Video' : 'Replace Video'}
              onReplaceVideo={() => {
                if (isCameraSource) {
                  setSelectedDatasetVideo(null, null);
                  onStartStream?.();
                } else {
                  fileInputRef.current?.click();
                }
              }}
            />

            <TrainingDetailsCard
              modelName={hyperparams.model_name}
              prompt={hyperparams.prompt}
              baseDetector={hyperparams.base_detector}
              outputDir={hyperparams.output_dir}
              savedOutputDir={savedOutputDir}
              workspaceDir={workspaceDir}
              defaultOutputDir={defaultOutputDir}
              availableDetectors={availableDetectors}
              isStarting={isStarting}
              onChangeModelName={(val) => setHyperparams({ model_name: val })}
              onChangePrompt={(val) => setHyperparams({ prompt: val })}
              onChangeDetector={(val) => setHyperparams({ base_detector: val })}
              onChangeOutputDir={(val) => setHyperparams({ output_dir: val })}
              onSaveOutputDir={async (val) => {
                const res = await saveOutputFolder(val);
                if (res.ok) {
                  onShowToast?.('success', `Saved workspace folder: ${res.workspace_dir}`);
                } else {
                  onShowToast?.('error', res.error || 'Failed to save output folder');
                }
              }}
              onStartTraining={handleStartFromCard}
              onClearVideo={() => {
                setSelectedDatasetVideo(null, null);
                onShowToast?.('info', 'Training video cleared');
              }}
            />
          </div>
        </div>
      )}

      {/* Camera Mode Cards (shown when no recorded video is open) */}
      {isCameraSource && !selectedDatasetVideo && !isCameraConnected && (
        <CameraOfflineCard
          onSwitchToVideo={() => onRequestSwitchMode?.('uploaded-video')}
          onRetryConnection={onRetryConnection}
          onCanvasClick={() => {}}
          errorMessage={cameraError}
        />
      )}

      {isCameraSource && !selectedDatasetVideo && isCameraConnected && !isStreaming && (
        <CameraStandbyCard
          onStartStream={onStartStream}
          onSwitchToVideo={() => onRequestSwitchMode?.('uploaded-video')}
          onCanvasClick={() => {}}
        />
      )}

      {isCameraSource && !selectedDatasetVideo && isCameraConnected && isStreaming && (
        <div className={`relative w-full h-full flex items-center justify-center bg-black ${isRecording ? 'ring-2 ring-inset ring-[var(--status-bad)]' : ''}`}>
          <img
            src={`${getApiBaseUrl()}/oak/inference/stream/live?t=${streamCacheBuster}`}
            className="w-full h-full object-contain"
            alt="OAK Camera Training Feed"
            style={{
              filter: cameraConfig
                ? `brightness(${Math.max(0.2, 1 + ((cameraConfig.brightness ?? 0) / 100))}) contrast(${Math.max(0.2, (cameraConfig.contrast ?? 50) / 50)})`
                : undefined,
            }}
            onError={() => {
              if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
              retryTimeoutRef.current = setTimeout(() => {
                setStreamCacheBuster(Date.now());
              }, 1200);
            }}
          />

          {/* Top-left Status Badge: LIVE STREAM or RECORDING (rounded-xl theme card, no oval) */}
          <div className={`absolute top-3 left-3 z-30 flex items-center h-8 px-3 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border shadow-xs select-none ${
            isRecording ? 'border-[var(--status-bad-border)]' : 'border-[var(--border-color)]'
          }`}>
            {isRecording ? (
              <div className="flex items-center gap-2 text-[var(--status-bad-text)] font-bold text-xs tracking-wide">
                <span className="w-2 h-2 rounded-full bg-[var(--status-bad)] animate-pulse shrink-0" />
                <span>RECORDING</span>
                <span className="font-mono text-[var(--text-primary)] tabular-nums">
                  {formatDuration(recordingDuration)}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-[var(--accent-main)] font-bold text-xs">
                <span className="w-2 h-2 rounded-full bg-[var(--accent-main)] animate-pulse shrink-0" />
                <span>LIVE STREAM</span>
              </div>
            )}
          </div>

          {/* Left-Side Vertical Toolbar: Stream, Record & Stop Stream */}
          <CameraLeftToolbar
            isStreaming={isStreaming}
            isRecording={isRecording}
            isSavingRecording={isSavingRecording}
            recordingDuration={recordingDuration}
            onStartStream={onStartStream}
            onStopStream={onStopStream}
            onToggleRecording={onToggleRecording}
            recordTooltip="Record Training Video"
          />
        </div>
      )}
    </div>
  );
};
