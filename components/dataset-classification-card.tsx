'use client';

import type { DatasetInfo } from '@/lib/types';

interface DatasetClassificationCardProps {
  dataset: DatasetInfo;
}

interface Row {
  label: string;
  value: string;
  /** Short explanation shown as a native tooltip on hover (via the
   *  `title` attribute) — no extra dependency, works everywhere the
   *  card renders, including when this component is shared/embedded
   *  outside the main app shell. */
  hint: string;
  /** Candidate fields are a naming/shape heuristic, never a promise of
   *  statistical eligibility — dimmed + "Candidates" suffix distinguishes
   *  them from resolved structural facts (Format, User ID, Variant, ...). */
  isCandidate?: boolean;
}

function joinOrNone(values: string[] | undefined): string {
  if (!values || values.length === 0) return 'None detected';
  return values.join(', ');
}

/**
 * Full, explicit breakdown of everything DatasetInfo resolved or flagged —
 * complements the compact green summary banner above it. Deliberately uses
 * "Candidates" in the label for the last three rows: the classifier found a
 * plausible column, that's it — it's never a statement that the field is
 * eligible or that any downstream statistical module will use it.
 */
export function DatasetClassificationCard({ dataset }: DatasetClassificationCardProps) {
  const rows: Row[] = [
    {
      label: 'Format',
      value: dataset.type,
      hint: 'The overall shape the classifier detected for this file (e.g. raw user-level rows vs. a pre-aggregated summary).',
    },
    {
      label: 'Users',
      value: dataset.users.toLocaleString(),
      hint: 'Distinct user/subject count after deduplication — the sample size used for every statistical test below.',
    },
    {
      label: 'User ID',
      value: dataset.userIdColumn ?? 'Not resolved',
      hint: 'The column identified as the unique user/subject identifier.',
    },
    {
      label: 'Variant',
      value: dataset.variantColumn ?? 'Not resolved',
      hint: 'The column identified as the experiment arm/group assignment (e.g. control vs. treatment).',
    },
    {
      label: 'Variants',
      value: joinOrNone(dataset.variantValues),
      hint: 'The distinct values found in the Variant column.',
    },
    {
      label: 'Primary Metric',
      value: dataset.primaryMetric || dataset.metricLabel,
      hint: 'The outcome column the hypothesis test will be run on.',
    },
    {
      label: 'Additional Metrics',
      value: joinOrNone(dataset.additionalMetrics),
      hint: 'Other numeric/event columns detected that could be analyzed as secondary metrics, but are not the primary metric.',
    },
    {
      label: 'Stratification Candidates',
      value: joinOrNone(dataset.stratificationCandidates),
      hint: 'Columns that look suitable for stratified analysis (e.g. pre-experiment segments). A candidate only — eligibility is re-checked when stratification actually runs.',
      isCandidate: true,
    },
    {
      label: 'Guardrail Candidates',
      value: joinOrNone(dataset.guardrailCandidates),
      hint: 'Columns that look like they could serve as guardrail metrics (e.g. must-not-regress signals like errors or latency). A candidate only, not an active guardrail until configured.',
      isCandidate: true,
    },
    {
      label: 'Covariate Candidates',
      value: joinOrNone(dataset.covariateCandidates),
      hint: 'Pre-experiment numeric columns that look usable as CUPED covariates for variance reduction. A candidate only — CUPED re-validates eligibility before applying one.',
      isCandidate: true,
    },
  ];

  return (
    <div className="rounded-lg border border-black/10 bg-white px-4 py-3 animate-slide-up">
      <p className="mb-3 text-[13px] font-medium text-black">Dataset Classification</p>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-2.5 sm:grid-cols-2">
        {rows.map((row) => (
          <div key={row.label} className="flex flex-col gap-0.5">
            <dt
              className="text-[10px] font-medium uppercase tracking-wide text-neutral-400 cursor-help"
              title={row.hint}
            >
              {row.label}
            </dt>
            <dd
              className={
                row.isCandidate && row.value === 'None detected'
                  ? 'text-[13px] text-neutral-400'
                  : 'text-[13px] text-neutral-800'
              }
            >
              {row.value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
