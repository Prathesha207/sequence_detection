import React from 'react';
import type { DetectionBox } from '../../types';

interface BoundingBoxOverlayProps {
  detections: DetectionBox[];
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  showAllBoxes?: boolean;
  isHandPresent?: boolean;
  isSceneAnomaly?: boolean;
  isCountMismatch?: boolean;
}

interface ThemePalette {
  border: string;
  bg: string;
  tagBg: string;
  arrowBorderDown: string;
  arrowBorderUp: string;
}

const PALETTE_THEME: ThemePalette = {
  border: 'border-[var(--accent-main,#7886C7)]',
  bg: 'bg-[var(--accent-main,#7886C7)]/10',
  tagBg: 'bg-[var(--accent-main,#7886C7)]',
  arrowBorderDown: 'border-t-[var(--accent-main,#7886C7)]',
  arrowBorderUp: 'border-b-[var(--accent-main,#7886C7)]',
};

const PALETTE_ANOMALY: ThemePalette = {
  border: 'border-[var(--status-bad,#EF4444)] animate-pulse',
  bg: 'bg-[var(--status-bad,#EF4444)]/20',
  tagBg: 'bg-[var(--status-bad,#EF4444)]',
  arrowBorderDown: 'border-t-[var(--status-bad,#EF4444)]',
  arrowBorderUp: 'border-b-[var(--status-bad,#EF4444)]',
};

/**
 * Capitalizes the first letter of each word neatly (e.g., 'pen marker' -> 'Pen Marker')
 */
export const formatLabel = (raw: string): string => {
  if (!raw) return '';
  return raw
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((word) => {
      if (!word) return '';
      // Keep short all-caps abbreviations unchanged
      if (/^[A-Z0-9]+$/.test(word) && word.length <= 4 && !word.toLowerCase().startsWith('obj')) {
        return word;
      }
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(' ');
};

/**
 * Assigns consistent theme color to all regular detections (anomalies get red alert styling)
 */
const getThemePalette = (isAnomaly: boolean): ThemePalette => {
  if (isAnomaly) return PALETTE_ANOMALY;
  return PALETTE_THEME;
};

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  detections,
  selectedDetectionId,
  onSelectDetection,
}) => {
  if (!detections || detections.length === 0) return null;

  return (
    <div className="absolute inset-0 w-full h-full pointer-events-none z-20">
      {detections
        .filter((box) => Number.isFinite(box.x) && Number.isFinite(box.y) && box.width > 0 && box.height > 0)
        .map((box, idx) => {
          const isSelected = String(box.id) === String(selectedDetectionId);
          const isAnomaly = Boolean(box.isAnomaly);
          const isHand = Boolean(box.species?.toLowerCase() === 'hand' || box.handDetected);

          const palette = getThemePalette(isAnomaly);
          const displayLabel = formatLabel(box.species);
          const conf = box.confidence > 0 ? `${Math.round(box.confidence * 100)}%` : '';

          // If the box is placed right against the top of the viewport, position the badge inside with space
          const isNearTop = box.y < 8;

          return (
            <div
              key={`bbox-${idx}-${box.id}`}
              role="button"
              tabIndex={0}
              onClick={(e) => {
                e.stopPropagation();
                onSelectDetection(String(box.id) === String(selectedDetectionId) ? null : String(box.id));
              }}
              style={{
                left: `${box.x}%`,
                top: `${box.y}%`,
                width: `${box.width}%`,
                height: `${box.height}%`,
              }}
              className={`absolute border-2 rounded-xl pointer-events-auto cursor-pointer transition-all duration-150 ${palette.border} ${palette.bg} ${isSelected ? 'ring-2 ring-white shadow-lg' : ''
                }`}
            >
              {/* Floating Label Badge with pointer caret - completely decoupled from touching box border */}
              {isNearTop ? (
                /* Top Edge Fallback: Floating inside the top with gap and upward caret */
                <div
                  className="absolute left-1/2 -translate-x-1/2 flex flex-col items-center pointer-events-none z-30 select-none"
                  style={{ top: '6px' }}
                >
                  <div
                    className={`w-0 h-0 border-x-4 border-x-transparent border-b-[5px] ${palette.arrowBorderUp} -mb-[0.5px]`}
                  />
                  <div
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold text-white tracking-wide shadow-md flex items-center gap-1.5 whitespace-nowrap ${palette.tagBg}`}
                  >
                    <span>{displayLabel}</span>
                    {conf && <span className="opacity-95 font-bold text-[11px]">{conf}</span>}
                  </div>
                </div>
              ) : (
                /* Standard Placement: Floating above box with a clean gap and downward caret */
                <div
                  className="absolute left-1/2 -translate-x-1/2 flex flex-col items-center pointer-events-none z-30 select-none"
                  style={{ bottom: 'calc(100% + 5px)' }}
                >
                  <div
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold text-white tracking-wide shadow-md flex items-center gap-1.5 whitespace-nowrap ${palette.tagBg}`}
                  >
                    <span>{displayLabel}</span>
                    {conf && <span className="opacity-95 font-bold text-[11px]">{conf}</span>}
                  </div>
                  <div
                    className={`w-0 h-0 border-x-4 border-x-transparent border-t-[5px] ${palette.arrowBorderDown} -mt-[0.5px]`}
                  />
                </div>
              )}
            </div>
          );
        })}
    </div>
  );
};
