import { useMemo, useRef, useEffect } from 'react';
import type { AnomalyStatus, DetectionBox, LogEntry } from '../types';

export function useAnomalyStatus({
  hasActiveStream,
  isRunning,
  isStarting,
  isCameraSource,
  framesProcessed,
  backendStats,
  addLog,
  detections,
}: {
  hasActiveStream: boolean;
  isRunning: boolean;
  isStarting: boolean;
  isCameraSource: boolean;
  framesProcessed: number;
  backendStats: any;
  addLog: (message: string, level?: LogEntry['level']) => void;
  detections: DetectionBox[];
}) {
  const activeDetections = useMemo(() => {
    if (!hasActiveStream && detections.length === 0) return [];
    return detections;
  }, [hasActiveStream, detections]);

  const backendStatus = backendStats.status;
  const isAnomalyFrame = backendStats.is_anomaly_frame === true;
  const verdict = backendStats.verdict;

  const anomalyStatus: AnomalyStatus = useMemo(() => {
    if (!hasActiveStream && detections.length === 0) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: isCameraSource ? 'NO CAMERA' : 'STANDBY',
        subMessage: isCameraSource
          ? 'No Luxonis OAK-D / USB camera connected'
          : 'Waiting for video stream...',
        verdict: null,
      };
    }

    const hasInferenceResult = framesProcessed > 0 || detections.length > 0;

    if (!isRunning && !hasInferenceResult) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: isCameraSource
          ? (hasActiveStream ? 'LIVE STREAM' : 'READY')
          : (hasActiveStream ? 'VIDEO READY' : 'STANDBY'),
        subMessage: isCameraSource
          ? (hasActiveStream ? 'Live camera feed active • Click Start Inference' : 'Camera ready • Click Start Stream or Start Inference')
          : (hasActiveStream ? 'Video loaded • Click Start Inference to begin analysis' : 'Waiting for video stream...'),
        verdict: null,
      };
    }

    if (isRunning && (isStarting || backendStatus === 'WARMING' || framesProcessed === 0)) {
      return {
        isAnomaly: false,
        type: 'WARMING',
        message: 'WARMING',
        subMessage: 'Warming up AI engine and acquiring targets...',
        verdict: null,
      };
    }

    const isAnomaly = isAnomalyFrame || backendStatus === 'ANOMALY' || verdict === 'ANOMALY';
    const doing = backendStats.doing;
    const doingStepObj = Array.isArray(backendStats.steps) ? backendStats.steps.find((s: any) => s.step === doing || s.id === doing) : null;
    const doingLabel = doingStepObj ? doingStepObj.label : doing ? `Step ${doing}` : undefined;

    return {
      isAnomaly,
      type: isAnomaly ? 'DEVIATION' : 'NONE',
      message: isAnomaly ? 'DEVIATION DETECTED' : 'ON TRACK',
      subMessage: isAnomaly
        ? 'Step performed out of sequence or deviation detected.'
        : `Inspecting SOP cycle. Current action: ${doingLabel || 'Step 1'}.`,
      doingStep: doing,
      doingLabel,
      verdict,
      detectedCount: detections.length,
      expectedCount: backendStats.total_steps || 5,
      difference: 0,
      foreignSpecies: [],
      foreignCount: 0,
    };
  }, [hasActiveStream, detections, isCameraSource, framesProcessed, isRunning, isStarting, backendStatus, isAnomalyFrame, verdict, backendStats]);

  const prevAnomalyRef = useRef(anomalyStatus.isAnomaly);
  const isAnomaly = anomalyStatus.isAnomaly;
  const subMessage = anomalyStatus.subMessage;

  useEffect(() => {
    if (!hasActiveStream || !isRunning) {
      prevAnomalyRef.current = false;
      return;
    }
    if (prevAnomalyRef.current !== isAnomaly) {
      if (isAnomaly) {
        addLog(`Sequence Deviation: ${subMessage}`, 'anomaly');
      } else {
        addLog(`Sequence Normalized: In proper order`, 'success');
      }
      prevAnomalyRef.current = isAnomaly;
    }
  }, [isAnomaly, subMessage, addLog, hasActiveStream, isRunning]);

  return {
    activeDetections,
    anomalyStatus,
  };
}
