import { DetectionBox, SOPStep } from '../types';

export const resetBBoxCache = () => {
  // No-op for SOP sequence
};

/**
 * Normalizes raw YOLO / ML detections into 0-100% bounding box entities for canvas overlay
 */
export const mapDetections = (data: any, vw: number, vh: number, _fallbackExpected?: any): DetectionBox[] => {
  if (!data || !Array.isArray(data.detections)) {
    return [];
  }

  const width = vw > 0 ? vw : 1280;
  const height = vh > 0 ? vh : 720;
  const boxes: DetectionBox[] = [];

  data.detections.forEach((d: any, idx: number) => {
    const rawBox = d.bbox || d.box;
    if (!Array.isArray(rawBox) || rawBox.length !== 4) return;

    const x1 = rawBox[0];
    const y1 = rawBox[1];
    const x2 = rawBox[2];
    const y2 = rawBox[3];

    const px = (x1 / width) * 100;
    const py = (y1 / height) * 100;
    const pw = ((x2 - x1) / width) * 100;
    const ph = ((y2 - y1) / height) * 100;

    const rawLabel = d.name || d.class_name || d.species || d.label;
    const label = rawLabel ? String(rawLabel).replace(/_/g, ' ') : `obj-${idx + 1}`;
    const conf = typeof d.confidence === 'number' ? d.confidence : typeof d.conf === 'number' ? d.conf : 0.9;
    const isAnomaly = d.isAnomaly === true || d.anomaly === true;

    boxes.push({
      id: d.id !== undefined && d.id !== null ? String(d.id) : `det-${idx}`,
      species: label,
      confidence: conf,
      isAnomaly,
      x: Math.max(0, Math.min(100, px)),
      y: Math.max(0, Math.min(100, py)),
      width: Math.max(0, Math.min(100, pw)),
      height: Math.max(0, Math.min(100, ph)),
      bbox: [x1, y1, x2, y2],
      handDetected: label.toLowerCase() === 'hand',
    });
  });

  return boxes;
};

/**
 * Translates raw backend ML dictionary into frontend-friendly SOPStep array and telemetry.
 */
export const mapBackendResultToSOPSteps = (data: any): {
  steps: SOPStep[];
  sequence: number[];
  doing: number | null;
  totalSteps: number;
  stepsDoneCount: number;
  stepsProgress: number;
  isAnomaly: boolean;
  verdict: 'NORMAL' | 'ANOMALY' | null;
} => {
  if (!data) {
    return {
      steps: [],
      sequence: [],
      doing: null,
      totalSteps: 5,
      stepsDoneCount: 0,
      stepsProgress: 0,
      isAnomaly: false,
      verdict: null,
    };
  }

  const rawSteps = Array.isArray(data.steps) ? data.steps : [];
  const sequence: number[] = Array.isArray(data.sequence) ? data.sequence : [];
  const doing: number | null = data.doing != null ? Number(data.doing) : null;
  const isAnomaly = data.is_anomaly_frame === true || data.status === 'ANOMALY';
  const verdict = data.verdict === 'NORMAL' || data.verdict === 'ANOMALY' ? data.verdict : null;

  // Track the order in which steps were completed
  const steps: SOPStep[] = rawSteps.map((s: any, idx: number) => {
    const stepId = Number(s.step || idx + 1);
    const label = String(s.label || `Step ${stepId}`);
    const rawStatus = String(s.status || '').toLowerCase();

    const isDone = rawStatus === 'done' || sequence.includes(stepId);
    const isDoing = doing === stepId || rawStatus === 'doing';
    const posInSeq = sequence.indexOf(stepId);

    let st: SOPStep['st'] = 'waiting';
    let note = '';

    if (isDone) {
      // Check if done in order
      const expectedIndex = stepId - 1;
      const wasOutOfOrder = posInSeq !== -1 && posInSeq !== expectedIndex;
      if (wasOutOfOrder) {
        st = 'ooo';
        note = posInSeq < expectedIndex ? 'Done before an earlier step' : 'Done after a later step';
      } else {
        st = 'done';
      }
    } else if (isDoing) {
      st = 'doing';
    } else if (rawStatus === 'taken_back') {
      st = 'ooo';
      note = 'Action taken back / reverted';
    }

    return {
      id: stepId,
      label,
      st,
      at: s.done_at_frame ? Number((s.done_at_frame / 30).toFixed(1)) : null,
      took: null,
      det: s.confidence ? Math.round(s.confidence * 100) : 98,
      note: s.note || note,
    };
  });

  const totalSteps = Number(data.total_steps || steps.length || 5);
  const stepsDoneCount = Number(data.steps_done_count || sequence.length);
  const stepsProgress = Number(data.steps_progress || (stepsDoneCount / totalSteps) * 100);

  return {
    steps,
    sequence,
    doing,
    totalSteps,
    stepsDoneCount,
    stepsProgress,
    isAnomaly,
    verdict,
  };
};
