import type { ExperimentDefinition } from './types';

/**
 * Gives the DRAFT -> READY -> RUNNING -> ... status field (previously
 * just a free-choosable label with no enforced meaning — see
 * ExperimentStatus's backend docstring: "nothing in this phase
 * transitions status automatically") actual teeth, WITHOUT touching
 * the backend: this only decides which transitions the UI allows and
 * why, mirroring the shape-validity rules the backend already
 * enforces at save time (exactly one PRIMARY hypothesis/metric,
 * exactly one Control variant — see ExperimentDefinitionBase's
 * `_validate_*` model validators) rather than duplicating or
 * second-guessing them. Nothing here is sent to or checked by the
 * server; a definition missing a piece is still perfectly save-able,
 * this only gates the label the analyst can attach to it.
 */
export interface ReadinessCheck {
  label: string;
  met: boolean;
}

export function getReadinessChecks(definition: ExperimentDefinition): ReadinessCheck[] {
  const hasPrimaryHypothesis = definition.hypotheses.some(
    (h) => h.role === 'primary' && h.hypothesis.statement.trim() && h.hypothesis.primaryMetric.trim()
  );
  const hasControl = definition.variants.some((v) => v.isControl);
  const hasTreatment = definition.variants.some((v) => !v.isControl);
  const hasPrimaryMetric = definition.metrics.some((m) => m.role === 'primary' && m.name.trim());
  const hasDataSource = Boolean(definition.dataSource?.datasetId);

  return [
    { label: 'Primary hypothesis defined', met: hasPrimaryHypothesis },
    { label: 'Control + at least one treatment variant', met: hasControl && hasTreatment },
    { label: 'Primary metric defined', met: hasPrimaryMetric },
    { label: 'Data source connected', met: hasDataSource },
  ];
}

/** Statuses that assert "this is actually runnable/concluded" — gated
 *  on readiness. DRAFT and NEEDS_INVESTIGATION/INVALID are always
 *  reachable (a problem is exactly what those last two describe). */
const GATED_STATUSES = new Set(['ready', 'running', 'shipped', 'completed']);

export function isStatusGated(status: string): boolean {
  return GATED_STATUSES.has(status);
}

export function isReady(definition: ExperimentDefinition): boolean {
  return getReadinessChecks(definition).every((c) => c.met);
}
