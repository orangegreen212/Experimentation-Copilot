import { FlaskConical, Target, ShieldCheck, AlertTriangle } from 'lucide-react';
import type { ExperimentReport } from '@/lib/types';
import { KpiTile } from './kpi-tile';
import {
  selectPrimaryStat,
  primaryEffectParts,
  hypothesisResultLabel,
  decisionToneFor,
} from '@/lib/report-format';

export function KpiGrid({ report }: { report: ExperimentReport }) {
  const evaluation = report.hypothesisEvaluation;
  const verdict = evaluation?.verdict ?? null;
  const primary = selectPrimaryStat(report);
  const decision = report.decision;

  const isSupported = verdict === 'SUPPORTED';
  const isPartial = verdict === 'PARTIALLY_SUPPORTED';
  const isRejected = verdict === 'NOT_SUPPORTED';

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <KpiTile
        icon={<FlaskConical className="h-4 w-4" />}
        label="Hypothesis"
        value={hypothesisResultLabel(verdict)}
        tone={isSupported ? 'go' : isPartial ? 'caution' : isRejected ? 'no' : 'neutral'}
      />
      <KpiTile
        icon={<Target className="h-4 w-4" />}
        label="Effect Size"
        value={primary ? primaryEffectParts(primary).primary : 'N/A'}
        tone={primary?.significant ? 'go' : 'neutral'}
      />
      <KpiTile
        icon={<ShieldCheck className="h-4 w-4" />}
        label="Validity"
        value={report.experimentValidity ?? 'N/A'}
        tone={
          report.experimentValidity === 'VALID'
            ? 'go'
            : report.experimentValidity === 'INVALID'
            ? 'no'
            : 'neutral'
        }
      />
      <KpiTile
        icon={<AlertTriangle className="h-4 w-4" />}
        label="Recommendation"
        value={decision ? decision.replace(/_/g, ' ') : 'N/A'}
        tone={decisionToneFor(decision)}
      />
    </div>
  );
}
