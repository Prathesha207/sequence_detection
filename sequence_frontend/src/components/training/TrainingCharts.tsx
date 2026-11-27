import React from 'react';
import { TrainingEpochMetric } from '../../types/trainingTypes';
import { TrendingDown, Award } from 'lucide-react';

interface TrainingChartsProps {
  metrics: TrainingEpochMetric[];
  totalEpochs: number;
}

export const TrainingCharts: React.FC<TrainingChartsProps> = ({ metrics, totalEpochs }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="w-full h-44 flex flex-col items-center justify-center rounded-xl bg-[var(--bg-card-subtle)] border border-dashed border-[var(--border-color)] text-xs text-[var(--text-muted)] p-4 text-center">
        <span>No telemetry data yet.</span>
        <span className="text-[11px] mt-1">Curves will render in real-time as epochs progress.</span>
      </div>
    );
  }

  const width = 500;
  const height = 140;
  const padLeft = 40;
  const padRight = 16;
  const padTop = 16;
  const padBottom = 26;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  // Max / Min calculations for Loss (using only real ML loss values)
  const valPoints = metrics.filter((m) => m.val_loss != null);
  const accPoints = metrics.filter((m) => m.accuracy != null);
  const maxLoss = Math.max(
    1.0,
    ...metrics.map((m) => Math.max(m.train_loss, m.val_loss ?? 0))
  );
  const minLoss = 0.0;

  // Dynamic X-axis over actual recorded ML points so curves never overflow SVG bounds
  const totalPoints = Math.max(2, metrics.length);
  const xPoint = (index: number) =>
    padLeft + (index / Math.max(1, totalPoints - 1)) * chartW;
  const yLoss = (loss: number) =>
    padTop + chartH - ((loss - minLoss) / (maxLoss - minLoss)) * chartH;

  // Loss paths
  const trainPath = metrics
    .map((m, i) => `${i === 0 ? 'M' : 'L'} ${xPoint(i).toFixed(1)} ${yLoss(m.train_loss).toFixed(1)}`)
    .join(' ');

  const valPath = valPoints
    .map((m, i) => {
      const idx = metrics.indexOf(m);
      return `${i === 0 ? 'M' : 'L'} ${xPoint(idx).toFixed(1)} ${yLoss(m.val_loss as number).toFixed(1)}`;
    })
    .join(' ');

  // Accuracy Scale (0 to 100%)
  const yAcc = (acc: number) => padTop + chartH - (acc / 100) * chartH;
  const accPath = accPoints
    .map((m, i) => {
      const idx = metrics.indexOf(m);
      return `${i === 0 ? 'M' : 'L'} ${xPoint(idx).toFixed(1)} ${yAcc(m.accuracy as number).toFixed(1)}`;
    })
    .join(' ');

  const latest = metrics[metrics.length - 1];
  const latestAcc = accPoints.length > 0 ? accPoints[accPoints.length - 1] : null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
      {/* 1. Loss Curve Chart */}
      <div className="flex flex-col p-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-xs">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-primary)]">
            <TrendingDown className="w-3.5 h-3.5 text-[var(--accent-main)]" />
            <span>Loss Convergence</span>
          </div>
          <div className="flex items-center gap-3 text-[10px] font-medium text-[var(--text-secondary)]">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-1 bg-[var(--accent-main)] rounded-full" />
              Train: {latest?.train_loss.toFixed(3) ?? '--'}
            </span>
            {latest?.val_loss != null && (
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-1 bg-sky-400 rounded-full border-t border-dashed" />
                Val: {latest.val_loss.toFixed(3)}
              </span>
            )}
          </div>
        </div>

        <div className="w-full overflow-hidden">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-36">
            {/* Horizontal Grid lines */}
            {[0, 0.5, 1.0].map((ratio) => {
              const y = padTop + chartH * ratio;
              const val = (maxLoss * (1 - ratio)).toFixed(1);
              return (
                <g key={ratio}>
                  <line
                    x1={padLeft}
                    y1={y}
                    x2={width - padRight}
                    y2={y}
                    stroke="var(--border-color)"
                    strokeDasharray="3 3"
                    strokeWidth="0.8"
                  />
                  <text
                    x={padLeft - 6}
                    y={y + 3}
                    textAnchor="end"
                    fontSize="9"
                    fill="var(--text-muted)"
                  >
                    {val}
                  </text>
                </g>
              );
            })}

            {/* Validation Loss line (only rendered when ML emits val_loss) */}
            {valPath && (
              <path
                d={valPath}
                fill="none"
                stroke="#38BDF8"
                strokeWidth="1.8"
                strokeDasharray="4 3"
              />
            )}

            {/* Training Loss line (accent main) */}
            <path
              d={trainPath}
              fill="none"
              stroke="var(--accent-main)"
              strokeWidth="2.2"
              strokeLinecap="round"
            />

            {/* Current point */}
            {latest && (
              <circle
                cx={xPoint(metrics.length - 1)}
                cy={yLoss(latest.train_loss)}
                r="3.5"
                fill="var(--accent-main)"
                stroke="#ffffff"
                strokeWidth="1.5"
              />
            )}

            {/* X-axis labels */}
            <text x={padLeft} y={height - 8} fontSize="9" fill="var(--text-muted)">
              {metrics[0]?.label || 'Point 1'}
            </text>
            <text
              x={width - padRight}
              y={height - 8}
              textAnchor="end"
              fontSize="9"
              fill="var(--text-muted)"
            >
              {latest?.label || `Stage ${totalEpochs}`}
            </text>
          </svg>
        </div>
      </div>

      {/* 2. Accuracy Curve Chart */}
      <div className="flex flex-col p-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-xs">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-primary)]">
            <Award className="w-3.5 h-3.5 text-emerald-500" />
            <span>Step Accuracy</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-medium text-emerald-600 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/20">
            <span>Current: {latestAcc?.accuracy != null ? `${latestAcc.accuracy.toFixed(1)}%` : '--'}</span>
          </div>
        </div>

        <div className="w-full overflow-hidden">
          {accPoints.length === 0 ? (
            <div className="w-full h-36 flex items-center justify-center rounded-xl bg-[var(--bg-card-subtle)] border border-dashed border-[var(--border-color)] text-[11px] text-[var(--text-muted)] text-center px-4">
              Self-check accuracy is evaluated by ML after training finishes (Stage 8).
            </div>
          ) : (
            <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-36">
              {/* Grid lines: 0%, 50%, 100% */}
              {[0, 0.5, 1.0].map((ratio) => {
                const y = padTop + chartH * ratio;
                const val = `${Math.round(100 * (1 - ratio))}%`;
                return (
                  <g key={ratio}>
                    <line
                      x1={padLeft}
                      y1={y}
                      x2={width - padRight}
                      y2={y}
                      stroke="var(--border-color)"
                      strokeDasharray="3 3"
                      strokeWidth="0.8"
                    />
                    <text
                      x={padLeft - 6}
                      y={y + 3}
                      textAnchor="end"
                      fontSize="9"
                      fill="var(--text-muted)"
                    >
                      {val}
                    </text>
                  </g>
                );
              })}

              {/* Accuracy line (emerald) */}
              <path
                d={accPath}
                fill="none"
                stroke="#10B981"
                strokeWidth="2.2"
                strokeLinecap="round"
              />

              {/* Current point */}
              {latestAcc && latestAcc.accuracy != null && (
                <circle
                  cx={xPoint(metrics.indexOf(latestAcc))}
                  cy={yAcc(latestAcc.accuracy)}
                  r="3.5"
                  fill="#10B981"
                  stroke="#ffffff"
                  strokeWidth="1.5"
                />
              )}

              {/* X-axis labels */}
              <text x={padLeft} y={height - 8} fontSize="9" fill="var(--text-muted)">
                {metrics[0]?.label || 'Point 1'}
              </text>
              <text
                x={width - padRight}
                y={height - 8}
                textAnchor="end"
                fontSize="9"
                fill="var(--text-muted)"
              >
                {latest?.label || `Stage ${totalEpochs}`}
              </text>
            </svg>
          )}
        </div>
      </div>
    </div>
  );
};
