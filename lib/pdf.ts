/**
 * Client-side PDF export for an ExperimentReport.
 *
 * This renders EXACTLY the fields already returned by the backend and
 * already shown in <ReportCard> — it does not run any statistics, call
 * the LLM, or hit any new backend endpoint. The backend does not
 * currently expose a PDF/report-download endpoint (see the frontend
 * integration manifest); this module is a pragmatic, fully-frontend
 * substitute that formats the same JSON the UI already has in memory.
 *
 * If a real backend-generated PDF is ever added, this module should be
 * replaced by a simple `fetch` + blob download in lib/api.ts instead.
 */

import { jsPDF } from 'jspdf';
import type { ExperimentReport, SegmentDimensionResult } from './types';

const PAGE_MARGIN = 40;
const LINE_HEIGHT = 14;
const PAGE_WIDTH = 595.28; // A4 in points
const PAGE_HEIGHT = 841.89;
const CONTENT_WIDTH = PAGE_WIDTH - PAGE_MARGIN * 2;

// jsPDF's core fonts ('helvetica' etc.) only have glyph-width metrics for
// WinAnsi/CP1252, NOT full Unicode. Backend-generated text legitimately
// contains characters outside that range (e.g. Greek alpha in "at
// \u03b1=0.05" from power_analysis.py, or "\u2192" arrows in funnel/LLM
// text like "0.57% \u2192 1.25%"). Feeding those straight into
// splitTextToSize/text() with no width metric produces exactly what was
// seen in exported reports: the arrow rendering as garbage ("!'"), and
// text after an unmapped glyph (e.g. the rest of "\u03b1=0.05") silently
// getting cut off. Normalize to ASCII-safe equivalents at the PDF
// boundary only — this never touches what's shown on-screen or what the
// backend returns, just what gets handed to jsPDF.
const PDF_CHAR_REPLACEMENTS: [RegExp, string][] = [
  [/\u2192/g, '->'], // →
  [/\u2190/g, '<-'], // ←
  [/\u03b1/g, 'alpha'], // α
  [/\u03b2/g, 'beta'], // β
  [/\u2264/g, '<='], // ≤
  [/\u2265/g, '>='], // ≥
  [/\u2248/g, '~'], // ≈
  [/\u00d7/g, 'x'], // ×
  [/\u00f7/g, '/'], // ÷
  [/\u2713/g, 'OK'], // ✓
  [/\u2717/g, 'X'], // ✗
  // Typographic punctuation the LLM-authored recommendations/next-steps
  // text frequently contains. These are NOT in WinAnsi/CP1252 either, and
  // previously hit the same silent-truncation bug as the arrows above:
  // once splitTextToSize/text() hit one of these, everything after it in
  // that string was dropped (e.g. "post\u2011launch ... related guardrails"
  // was rendered as "post launch ... related guard", and "long\u2011term
  // impact." as "lon"). This was the root cause of reports appearing to
  // "cut off" mid-word in the exported PDF.
  [/[\u2010\u2011\u2012\u2013\u2014]/g, '-'], // hyphen/non-breaking hyphen/figure dash/en dash/em dash
  [/[\u2018\u2019]/g, "'"], // curly single quotes
  [/[\u201c\u201d]/g, '"'], // curly double quotes
  [/\u2026/g, '...'], // ellipsis
  [/\u00a0/g, ' '], // non-breaking space
  [/\u2022/g, '-'], // bullet char appearing mid-string (the bullet glyph we draw ourselves is added separately, not via this replacement)
];

// Last-resort safety net: after the explicit replacements above, strip any
// character still outside the printable WinAnsi/CP1252 range (0x20-0xFF,
// keeping \n and \t) rather than silently truncating the rest of the
// string, which is what jsPDF's core "helvetica" font does when it meets
// an unmapped glyph. Better an occasional "?" than losing every word after
// it.
function stripUnmappableGlyphs(text: string): string {
  let out = '';
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (ch === '\n' || ch === '\t' || (code >= 0x20 && code <= 0xff)) {
      out += ch;
    } else {
      out += '?';
    }
  }
  return out;
}

function sanitizeForPdf(text: string): string {
  const replaced = PDF_CHAR_REPLACEMENTS.reduce(
    (acc, [pattern, replacement]) => acc.replace(pattern, replacement),
    text
  );
  return stripUnmappableGlyphs(replaced);
}

// --- Color palette (mirrors the on-screen <ReportCard> status colors) ----
const COLOR = {
  green: [22, 130, 70] as [number, number, number], // GO / PASS / significant / reliable effect
  red: [180, 35, 35] as [number, number, number], // NO-GO / FAIL / violated
  amber: [180, 120, 10] as [number, number, number], // MEDIUM confidence / not significant / caution
  gray: [110, 110, 110] as [number, number, number], // neutral / n/a
  black: [20, 20, 20] as [number, number, number],
};

// Maps a status-ish token to a semantic color. Falls back to black for
// anything not recognized, so plain text is never mis-colored.
function statusColor(raw: string): [number, number, number] {
  const v = raw.trim().toUpperCase();
  if (['GO', 'PASS', 'YES', 'VALID', 'HIGH', 'SIGNIFICANT', 'RELIABLE EFFECT', 'OK'].some((t) => v === t || v.startsWith(t)))
    return COLOR.green;
  if (['NO-GO', 'NOGO', 'FAIL', 'NO', 'INVALID', 'VIOLATED', 'LOW'].some((t) => v === t || v.startsWith(t)))
    return COLOR.red;
  if (['MEDIUM', 'NOT SIGNIFICANT', 'CAUTION', 'NOT ASSESSED'].some((t) => v === t || v.startsWith(t)))
    return COLOR.amber;
  return COLOR.black;
}

class PdfWriter {
  doc: jsPDF;
  y: number;

  constructor() {
    this.doc = new jsPDF({ unit: 'pt', format: 'a4' });
    this.y = PAGE_MARGIN;
  }

  private ensureSpace(height: number) {
    if (this.y + height > PAGE_HEIGHT - PAGE_MARGIN) {
      this.doc.addPage();
      this.y = PAGE_MARGIN;
    }
  }

  h1(text: string) {
    this.ensureSpace(28);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(18);
    this.doc.setTextColor(0, 0, 0);
    this.doc.text(sanitizeForPdf(text), PAGE_MARGIN, this.y);
    this.y += 24;
  }

  h2(text: string) {
    this.ensureSpace(22);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(13);
    this.doc.setTextColor(0, 0, 0);
    this.doc.text(sanitizeForPdf(text), PAGE_MARGIN, this.y);
    this.y += 18;
  }

  h3(text: string) {
    this.ensureSpace(16);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(11);
    this.doc.setTextColor(20, 20, 20);
    this.doc.text(sanitizeForPdf(text), PAGE_MARGIN, this.y);
    this.y += 14;
  }

  meta(text: string) {
    this.ensureSpace(12);
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(9);
    this.doc.setTextColor(110, 110, 110);
    this.doc.text(sanitizeForPdf(text), PAGE_MARGIN, this.y);
    this.y += 12;
  }

  body(text: string) {
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(10);
    this.doc.setTextColor(30, 30, 30);
    const lines: string[] = this.doc.splitTextToSize(sanitizeForPdf(text), CONTENT_WIDTH);
    for (const line of lines) {
      this.ensureSpace(LINE_HEIGHT);
      this.doc.text(line, PAGE_MARGIN, this.y);
      this.y += LINE_HEIGHT;
    }
  }

  bullet(text: string) {
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(10);
    this.doc.setTextColor(30, 30, 30);
    const lines: string[] = this.doc.splitTextToSize(sanitizeForPdf(text), CONTENT_WIDTH - 14);
    lines.forEach((line, i) => {
      this.ensureSpace(LINE_HEIGHT);
      if (i === 0) this.doc.text('\u2022', PAGE_MARGIN, this.y);
      this.doc.text(line, PAGE_MARGIN + 14, this.y);
      this.y += LINE_HEIGHT;
    });
  }

  keyValueRow(label: string, value: string, opts: { color?: [number, number, number] } = {}) {
    this.ensureSpace(LINE_HEIGHT);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(9.5);
    this.doc.setTextColor(80, 80, 80);
    this.doc.text(sanitizeForPdf(label), PAGE_MARGIN, this.y);
    const color = opts.color ?? COLOR.black;
    this.doc.setFont('helvetica', opts.color ? 'bold' : 'normal');
    this.doc.setTextColor(...color);
    this.doc.text(sanitizeForPdf(value), PAGE_MARGIN + 170, this.y);
    this.y += LINE_HEIGHT;
  }

  // Same as keyValueRow but auto-colors the value by status keyword
  // (GO/PASS -> green, NO-GO/FAIL -> red, MEDIUM/NOT SIGNIFICANT -> amber).
  statusRow(label: string, value: string) {
    this.keyValueRow(label, value, { color: statusColor(value) });
  }

  // Bullet whose leading "[TOKEN]" tag is colored by status, e.g.
  // "[PASS] Sample Ratio Mismatch: ..." renders "[PASS]" in green.
  statusBullet(tag: string, rest: string) {
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(10);
    const tagText = `[${tag}]`;
    const color = statusColor(tag);
    this.ensureSpace(LINE_HEIGHT);
    this.doc.setTextColor(...COLOR.black);
    this.doc.text('\u2022', PAGE_MARGIN, this.y);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setTextColor(...color);
    this.doc.text(sanitizeForPdf(tagText), PAGE_MARGIN + 14, this.y);
    const tagWidth = this.doc.getTextWidth(sanitizeForPdf(tagText + ' '));
    this.doc.setFont('helvetica', 'normal');
    this.doc.setTextColor(...COLOR.black);
    const lines: string[] = this.doc.splitTextToSize(sanitizeForPdf(rest), CONTENT_WIDTH - 14 - tagWidth);
    lines.forEach((line, i) => {
      if (i > 0) this.ensureSpace(LINE_HEIGHT);
      this.doc.text(line, PAGE_MARGIN + 14 + tagWidth, this.y);
      if (i < lines.length - 1) this.y += LINE_HEIGHT;
    });
    this.y += LINE_HEIGHT;
  }

  // A filled rounded badge, used for the headline Decision (GO / NO-GO).
  badge(text: string) {
    const color = statusColor(text);
    this.ensureSpace(30);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(13);
    const label = sanitizeForPdf(text);
    const padX = 10;
    const w = this.doc.getTextWidth(label) + padX * 2;
    const h = 20;
    this.doc.setFillColor(...color);
    this.doc.roundedRect(PAGE_MARGIN, this.y - 14, w, h, 4, 4, 'F');
    this.doc.setTextColor(255, 255, 255);
    this.doc.text(label, PAGE_MARGIN + padX, this.y);
    this.doc.setTextColor(...COLOR.black);
    this.y += h + 4;
  }

  spacer(px = 8) {
    this.y += px;
  }

  divider() {
    this.ensureSpace(10);
    this.doc.setDrawColor(220, 220, 220);
    this.doc.line(PAGE_MARGIN, this.y, PAGE_WIDTH - PAGE_MARGIN, this.y);
    this.y += 12;
  }
}

function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value == null) return 'N/A';
  return `${(value * 100).toFixed(digits)}%`;
}

function fmtPValue(p: number | null | undefined): string {
  if (p == null) return 'N/A';
  return p < 0.001 ? '<0.001' : p.toFixed(3);
}

function writeDimensionResult(w: PdfWriter, dim: SegmentDimensionResult) {
  w.h3(`Dimension: ${dim.dimension}`);
  w.meta(
    `Multiple-testing correction: ${dim.multipleTestingMethod} | Heterogeneous effect: ${
      dim.hasHeterogeneousEffect ? 'Yes' : 'No'
    }`
  );
  if (dim.reliableSegmentValues.length > 0) {
    w.body(`Reliable segment values (survive correction): ${dim.reliableSegmentValues.join(', ')}`);
  }
  dim.segmentEffects.forEach((seg) => {
    if (seg.sampleSizeStatus === 'insufficient') {
      w.bullet(
        `${seg.segmentValue}: insufficient sample size (control n=${seg.controlN}, variant n=${seg.variantN})${
          seg.skipDetail ? ' — ' + seg.skipDetail : ''
        }`
      );
      return;
    }
    const s = seg.statResult;
    if (!s) {
      w.bullet(`${seg.segmentValue}: n=${seg.controlN}/${seg.variantN}, no test result available`);
      return;
    }
    const adjusted = s.adjustedPValue != null ? `, adj. p=${fmtPValue(s.adjustedPValue)}` : '';
    w.statusBullet(
      s.significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT',
      `${seg.segmentValue}: control=${s.control}, variant=${s.variant}, delta=${s.delta}, ` +
        `p=${fmtPValue(s.pValue)}${adjusted} (n=${seg.controlN}/${seg.variantN})`
    );
  });
  w.spacer(6);
}

export function downloadReportPdf(
  report: ExperimentReport,
  opts: { datasetName?: string; experimentId?: string; prompt?: string } = {}
): void {
  const w = new PdfWriter();

  // --- Title / summary -------------------------------------------------
  w.h1('Experiment Review Report');
  w.meta(
    [
      opts.datasetName ? `Dataset: ${opts.datasetName}` : null,
      opts.experimentId ? `Experiment ID: ${opts.experimentId}` : null,
      `Generated: ${new Date().toLocaleString()}`,
    ]
      .filter(Boolean)
      .join('   |   ')
  );
  w.spacer(10);

  w.h2('Experiment Summary');
  if (opts.prompt) {
    w.body(`Prompt: ${opts.prompt}`);
    w.spacer(4);
  }
  w.body(report.executiveSummary);
  w.spacer(6);
  // NOTE: `report.confidence` is the LEGACY data/experiment reliability
  // assessment (SRM, quality checks, power/MDE — see
  // backend/app/schemas/report.py's module docstring), NOT the decision
  // confidence. It must never be labeled "Recommendation Confidence" —
  // that label belongs only to `report.recommendationConfidence`, shown
  // in the Decision section below. Using the same label for both here
  // previously made a report show e.g. "Recommendation Confidence: HIGH
  // (5/5)" in the summary and "Recommendation Confidence: MEDIUM" in the
  // Decision section a few lines later, which read as a contradiction.
  w.statusRow('Confidence in Results', `${report.confidence} (${report.confidenceStars}/5)`);
  w.body(report.confidenceReason);
  if (report.srmWarning) {
    w.body('Warning: Sample Ratio Mismatch (SRM) detected.');
  }
  w.spacer(10);

  // --- Hypothesis Evaluation ---------------------------------------------
  // Explicit SUPPORTED/REJECTED verdict, deliberately separate from the
  // Decision section below — mirrors <HypothesisEvaluationSection> in
  // report-card.tsx. Renders only when a hypothesis was provided; nothing
  // here is computed in the browser, it's a straight readout of
  // report.hypothesisEvaluation (app/stats/hypothesis_evaluator.py).
  if (report.hypothesis) {
    const evaluation = report.hypothesisEvaluation;
    const verdict = evaluation?.verdict ?? null;
    const verdictLabel =
      verdict === 'SUPPORTED'
        ? 'SUPPORTED'
        : verdict === 'PARTIALLY_SUPPORTED'
        ? 'PARTIALLY SUPPORTED'
        : verdict === 'NOT_SUPPORTED'
        ? 'REJECTED'
        : 'EVALUATION UNAVAILABLE';

    w.h2('Hypothesis Evaluation');
    w.body(report.hypothesis.statement);
    w.spacer(4);
    w.badge(verdictLabel);
    if (evaluation && verdict != null) {
      w.statusRow('Observed Effect', `${fmtPct(evaluation.observedEffectRelative)} relative`);
      if (evaluation.expectedEffectRelative != null) {
        w.keyValueRow('Expected Effect', `${fmtPct(evaluation.expectedEffectRelative)} relative`);
      }
      w.statusRow('Statistical Significance', evaluation.statisticallySignificant ? 'Yes' : 'No');
      if (report.experimentValidity) w.keyValueRow('Experiment Validity', report.experimentValidity);
      if (
        verdict === 'NOT_SUPPORTED' &&
        report.decisionAudit?.powerEvidence &&
        report.decisionAudit.powerEvidence.status === 'warning'
      ) {
        w.spacer(4);
        const pe = report.decisionAudit.powerEvidence;
        w.body(
          `Note: this run was underpowered (${pe.value}${pe.detail ? ' — ' + pe.detail : ''}). ` +
            'A non-significant result here may mean the sample was too small to detect a real ' +
            'effect of this size, not that no effect exists.'
        );
      }
    } else if (evaluation?.evaluationNote) {
      w.body(evaluation.evaluationNote);
    }
    w.spacer(10);
  }

  // --- Decision ----------------------------------------------------------
  if (report.decision) {
    w.h2('Decision');
    w.badge(report.decision);
    if (report.experimentValidity) w.statusRow('Experiment Validity', report.experimentValidity);
    if (report.guardrailStatus) w.statusRow('Guardrail Status', report.guardrailStatus);
    if (report.guardrailRequestState) {
      w.keyValueRow('Guardrail Request State', report.guardrailRequestState.replace(/_/g, ' '));
    }
    if (report.practicalSignificance != null) {
      w.statusRow('Practical Significance', report.practicalSignificance ? 'Yes' : 'No');
    }
    if (report.recommendationConfidence) {
      w.statusRow('Recommendation Confidence', report.recommendationConfidence);
    }
    if (report.decisionReason) {
      w.spacer(4);
      w.body(report.decisionReason);
    }
    w.spacer(10);
  }

  // --- Decision Audit Trail (Phase 7) ------------------------------------
  // Renders exactly the deterministic DecisionAuditTrail already computed
  // by the backend (app/stats/decision_audit.py) — no new formatting logic
  // or wording decisions happen here.
  const audit = report.decisionAudit;
  if (audit) {
    w.h2('Decision Audit Trail');
    w.meta('Why the system reached this exact decision');
    w.statusRow('Decision', audit.headline);
    w.spacer(4);
    w.h3('Why this decision');
    audit.rationale.forEach((r) => w.bullet(r));
    w.spacer(4);
    if (audit.supportingFacts.length > 0) {
      w.h3('Evidence supporting decision');
      audit.supportingFacts.forEach((f) =>
        w.statusBullet(f.value, `${f.label}${f.detail ? ' — ' + f.detail : ''}`)
      );
      w.spacer(4);
    }
    if (audit.warnings.length > 0) {
      w.h3('Warnings / limitations');
      audit.warnings.forEach((f) =>
        w.statusBullet(f.value, `${f.label}${f.detail ? ' — ' + f.detail : ''}`)
      );
      w.spacer(4);
    }
    w.h3('Decision impact');
    w.body(audit.decisionImpact);
    w.spacer(10);
  }

  // --- Data Quality --------------------------------------------------
  w.h2('Data Quality & Assumptions');
  report.qualityChecks.forEach((c) => {
    w.statusBullet(c.passed ? 'PASS' : 'FAIL', `${c.label}: ${c.detail}`);
  });
  w.spacer(10);

  // --- Statistical Results ---------------------------------------------
  w.h2('Statistical Results');
  report.stats.forEach((s) => {
    w.h3(s.metric);
    w.body(`${s.testName} — ${s.selectionReason}`);
    w.keyValueRow('Control', s.control);
    w.keyValueRow('Variant', s.variant);
    w.keyValueRow('Delta', s.delta);
    w.keyValueRow('p-value', fmtPValue(s.pValue));
    if (s.adjustedPValue != null) w.keyValueRow('Adjusted p-value', fmtPValue(s.adjustedPValue));
    w.keyValueRow('95% CI', `[${s.ciLower}, ${s.ciUpper}]`);
    w.statusRow('Significant', s.significant ? 'Yes' : 'No');
    w.spacer(6);
  });
  if (report.bootstrapCiLower != null && report.bootstrapCiUpper != null) {
    w.h3('Bootstrap Cross-check');
    w.keyValueRow(
      '95% CI (difference)',
      `[${report.bootstrapCiLower.toFixed(4)}, ${report.bootstrapCiUpper.toFixed(4)}]`
    );
    w.keyValueRow('Iterations', String(report.bootstrapIterations ?? 10000));
    w.spacer(6);
  }
  w.h3('Power Analysis');
  w.keyValueRow('MDE', report.mde);
  w.keyValueRow('Sample Size', report.sampleSizeNote);
  w.spacer(10);

  // --- Segment Analysis (Phase 5) --------------------------------------
  w.h2('Segment Analysis');
  w.meta('Exploratory. Does not override the primary experiment decision.');
  const seg = report.segmentation;
  if (!seg || !seg.ran) {
    w.body(seg?.reason ?? 'Segmentation was not run for this experiment.');
  } else if (seg.dimensionResults.length === 0) {
    w.body(seg.reason);
  } else {
    w.body(seg.reason);
    w.spacer(4);
    seg.dimensionResults.forEach((dim) => writeDimensionResult(w, dim));
    if (seg.skippedDimensions.length > 0) {
      w.h3('Skipped Dimensions');
      seg.skippedDimensions.forEach((d) => w.bullet(`${d.column} (${d.reason}): ${d.detail}`));
    }
  }
  w.spacer(10);

  // --- Guardrails (Decision Support) ------------------------------------
  w.h2('Guardrails');
  const ds = report.decisionSupport;
  if (!ds || ds.guardrailFindings.length === 0) {
    // Guardrail root-cause fix: never say "Not available" for both
    // "never requested" and "requested but not found" — see
    // report.guardrailRequestState (GuardrailRequestState).
    if (
      report.guardrailRequestState === 'REQUESTED_NOT_FOUND' ||
      report.guardrailRequestState === 'PARTIALLY_AVAILABLE'
    ) {
      w.body('Requested — not found.');
      (report.guardrailResolutions ?? [])
        .filter((r) => !r.resolved)
        .forEach((r) => w.bullet(`${r.requestedName}: not available in this dataset`));
    } else {
      w.body('Not specified.');
    }
  } else {
    ds.guardrailFindings.forEach((g) => {
      w.statusBullet(
        g.violated ? 'VIOLATED' : 'PASS',
        `${g.metric}` +
          (g.relativeChange != null ? `, change=${fmtPct(g.relativeChange)}` : '') +
          (g.statisticallySignificant ? ', statistically significant' : '')
      );
    });
  }
  w.spacer(10);

  // --- Recommendations & Next Steps -------------------------------------
  w.h2('Recommendations');
  report.recommendations.forEach((r) => w.bullet(r));
  w.spacer(6);
  w.h2('Next Steps');
  report.nextSteps.forEach((s) => w.bullet(s));

  // --- Run Information (Phase 8) — compact summary only, never a debug log ---
  if (report.runMetadata) {
    const rm = report.runMetadata;
    w.spacer(10);
    w.h2('Run Information');
    w.keyValueRow('Dataset', `${rm.datasetName} (${rm.datasetClassification})`);
    w.keyValueRow('Analysis Mode', rm.analysisMode);
    w.keyValueRow('Users / Variants', `${rm.userCount} users, ${rm.variantCount} variants`);
    // See the matching note in components/report-card.tsx: this is the
    // model REQUESTED for the run (Settings.model), not proof the
    // Planner stage used an LLM — with the default keyword planner it
    // never does. Labeled "Planner" this looked like a claim about the
    // Planner stage specifically, which was misleading.
    w.keyValueRow('Requested LLM Model', rm.selectedModel ? rm.selectedModel : `none (${rm.plannerBackend} planner)`);
    w.keyValueRow('Report Backend', rm.reportBackend);
    w.keyValueRow('Execution Status', rm.executionStatus);
    if (report.reportFallbackReason) {
      w.spacer(4);
      w.body(`Note: ${report.reportFallbackReason}`);
    }
  }

  const fileName = `experiment-report${opts.experimentId ? '-' + opts.experimentId : ''}.pdf`;
  w.doc.save(fileName);
}
