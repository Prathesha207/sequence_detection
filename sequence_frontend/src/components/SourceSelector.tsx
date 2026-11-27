import React from 'react';
import { StreamSourceType } from '../types';
import { Video, Camera, Play, Square, Trash2, Sparkles } from 'lucide-react';

import { useInferenceStore } from '../store/inferenceStore';

interface SourceSelectorProps {
  sourceType: StreamSourceType;
  onSourceChange: (type: StreamSourceType) => void;

  appMode?: 'inference' | 'training';
  onModeChange?: (mode: 'inference' | 'training') => void;
  onStartTraining?: () => void;
  onStopTraining?: () => void;
  isTrainingRunning?: boolean;
  isTrainingStarting?: boolean;
  hasTrainingVideo?: boolean;
  onClearTrainingVideo?: () => void;

  onOpenSettings?: () => void;
  onCustomVideoUploaded?: (videoUrl: string, fileName: string, sessionId?: string) => void;
  customVideoName?: string;
  customVideoUrl?: string;
  videoSessionId?: string | null;
  hasActiveVideo?: boolean;
  onClearCustomVideo?: () => void;

  isRunning?: boolean;
  isStarting?: boolean;
  isRecording?: boolean;
  isSavingRecording?: boolean;
  recordingDuration?: number;
  onToggleRecording?: () => void;
  onToggleRunning?: () => void;
  onStopInference?: () => void;
  onResumeInference?: () => void;
  onRequestSwitchMode?: (type: StreamSourceType) => void;
  isStreaming?: boolean;
  onStartStream?: () => void;
  onStopStream?: () => void;
  isCameraConnected?: boolean;
  cameraStartingState?: 'idle' | 'waking_camera' | 'waiting_frame' | 'ready';
  cameraRecordSessionId?: string | null;
  onClearCameraRecord?: () => void;
}

export const SourceSelector: React.FC<SourceSelectorProps> = ({
  sourceType,
  onSourceChange,
  appMode = 'training',
  onModeChange,
  onStartTraining,
  onStopTraining,
  isTrainingRunning = false,
  isTrainingStarting: _isTrainingStarting = false,
  hasTrainingVideo: _hasTrainingVideo = false,
  onClearTrainingVideo: _onClearTrainingVideo,

  onOpenSettings: _onOpenSettings,
  customVideoName: _customVideoName,
  customVideoUrl: _customVideoUrl,
  hasActiveVideo = false,
  onClearCustomVideo,
  isRunning = false,
  isStarting = false,
  isRecording: _isRecording = false,
  isSavingRecording: _isSavingRecording = false,
  recordingDuration: _recordingDuration = 0,
  onToggleRecording: _onToggleRecording,
  onToggleRunning,
  onStopInference,
  onResumeInference: _onResumeInference,
  onRequestSwitchMode,
  isStreaming: _isStreaming = false,
  onStartStream: _onStartStream,
  onStopStream: _onStopStream,
  isCameraConnected: _isCameraConnected = true,
  cameraStartingState: _cameraStartingState = 'ready',
  cameraRecordSessionId: _cameraRecordSessionId,
  onClearCameraRecord: _onClearCameraRecord,
}) => {
  const isVideoLoading = useInferenceStore((state) => state.isVideoLoading);

  const handleSourceClick = (targetType: StreamSourceType) => {
    if (isRunning) return;

    if (onRequestSwitchMode) {
      onRequestSwitchMode(targetType);
    } else {
      onSourceChange(targetType);
    }
  };

  const isVideoMode = sourceType === 'sample-pond' || sourceType === 'uploaded-video';
  const isCameraMode = sourceType === 'oak-camera' || sourceType === 'webcam';

  return (
    <div className="w-full flex-shrink-0">
      <div className="flex flex-row items-center justify-between gap-2 sm:gap-4 p-2 sm:p-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-sm flex-wrap">

        {/* Source Toggle Tabs */}
        <div className="flex bg-[var(--bg-card-subtle)] p-1 rounded-xl border border-[var(--border-color)] flex-shrink-0">
          <button
            disabled={isRunning}
            onClick={() => handleSourceClick('uploaded-video')}
            title={isRunning ? 'Stop inference before switching source' : 'Switch to Video File mode'}
            className={`flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
              } ${sourceType === 'uploaded-video'
                ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-xs'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
          >
            <Video className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-[var(--accent-main)]" />
            <span>Video<span className="hidden sm:inline"> File</span></span>
          </button>

          <button
            disabled={isRunning}
            onClick={() => handleSourceClick('oak-camera')}
            title={isRunning ? 'Stop inference before switching source' : 'Switch to OAK Camera mode'}
            className={`flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
              } ${sourceType === 'oak-camera'
                ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-xs'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
          >
            <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-[var(--accent-main)]" />
            <span>OAK<span className="hidden sm:inline"> Camera</span></span>
          </button>
        </div>

        {/* Navigation & Contextual Controls */}
        <div className="flex items-center justify-end gap-1.5 sm:gap-2 flex-shrink-0 ml-auto">
          {/* Page Navigation 1: TRAINING */}
          <button
            type="button"
            onClick={() => onModeChange?.('training')}
            title="Go to Training page"
            className={`h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-5 rounded-xl font-bold text-[11px] sm:text-xs shadow-sm transition-all shrink-0 cursor-pointer active:scale-95 ${appMode === 'training'
                ? 'bg-[var(--accent-main)] text-white hover:opacity-90'
                : 'bg-[var(--bg-card-subtle)] hover:bg-[var(--accent-pond-subtle)] hover:border-[var(--accent-main)] border border-[var(--border-color)] text-[var(--text-primary)]'
              }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span className="whitespace-nowrap">TRAINING</span>

          </button>

          {/* Page Navigation 2: INFERENCE */}
          <button
            type="button"
            onClick={() => onModeChange?.('inference')}
            title="Go to Inference page"
            className={`h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-5 rounded-xl font-bold text-[11px] sm:text-xs shadow-sm transition-all shrink-0 cursor-pointer active:scale-95 ${appMode === 'inference'
                ? 'bg-[var(--accent-main)] text-white hover:opacity-90'
                : 'bg-[var(--bg-card-subtle)] hover:bg-[var(--accent-pond-subtle)] hover:border-[var(--accent-main)] border border-[var(--border-color)] text-[var(--text-primary)]'
              }`}
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span className="whitespace-nowrap">INFERENCE</span>
          </button>

          {/* Stop Inference button when actively running */}
          {appMode === 'inference' && isRunning && (
            <button
              onClick={() => {
                if (onStopInference) onStopInference();
                else if (onToggleRunning) onToggleRunning();
              }}
              title="Stop AI inference"
              className="h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-4 rounded-xl bg-[var(--status-bad)] hover:opacity-90 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>{isCameraMode ? 'FINISH INSPECTION' : 'STOP'}</span>
            </button>
          )}

          {/* Clear Video button in inference mode */}
          {appMode === 'inference' && isVideoMode && hasActiveVideo && !isRunning && (
            <button
              disabled={isStarting || isVideoLoading}
              onClick={() => {
                onClearCustomVideo?.();
              }}
              title="Clear video and results"
              className="h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-4 rounded-xl bg-rose-600 hover:bg-rose-700 active:bg-rose-800 text-white font-bold text-[11px] sm:text-xs shadow-md shadow-rose-600/30 active:scale-95 cursor-pointer transition-all shrink-0 border border-rose-500"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>CLEAR</span>
            </button>
          )}
        </div>

      </div>
    </div>
  );
};
