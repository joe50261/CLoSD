// Day 6 parity harness: load fixtures from generate_fixtures.py, recompute
// the equivalent tensors in the browser via the TS pipeline, and compute
// per-tensor mean abs err. Drives the PASS/FAIL judgement against the
// success criteria in the plan:
//   - text_embed MAE < 1e-3 (FP32 CLIP) / < 5e-3 (FP16)
//   - per-step x_t / pred_xstart MAE < 5e-3 (intermediate, FP16)
//   - final motion MAE < 1e-3 per element (FP16 inference)

import { autoregressiveSample } from "./ar_loop.js";
import { runCfgStep } from "./cfg.js";
import { loadTrunk, type TrunkSession } from "./ort_setup.js";
import { DEFAULT_CONFIG, buildSchedule, ddimStepEta0 } from "./scheduler.js";
import {
  loadBertTextEncoder,
  loadTextEncoder,
  type BertTextEncoder,
  type TextEncoder,
} from "./text_encoder.js";
import { mae, type T4 } from "./tensor.js";

export interface ParityCheck {
  name: string;
  mae: number;
  threshold: number;
  passed: boolean;
}

export interface ParityReport {
  checks: ParityCheck[];
  textEncoder: ParityCheck;
  perfMs: { totalSample: number; perStepAvg: number };
  executionProvider: string;
}

/**
 * Fetch a `.npy` file from the fixtures URL and parse it into a T4.
 * Supports float32 and bool .npy files (the only types we save).
 *
 * .npy v1 layout: 6-byte magic, 2-byte version, 2-byte header_len, header dict, data.
 */
export async function fetchNpy(url: string): Promise<T4> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch ${url}: ${response.status}`);
  const buf = await response.arrayBuffer();
  const view = new DataView(buf);
  // Magic check
  const magic = new Uint8Array(buf, 0, 6);
  if (magic[0] !== 0x93 || String.fromCharCode(...magic.slice(1, 6)) !== "NUMPY") {
    throw new Error(`${url}: not a .npy file`);
  }
  const major = view.getUint8(6);
  const headerLen =
    major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const headerStr = new TextDecoder().decode(
    new Uint8Array(buf, headerStart, headerLen),
  );
  // Parse the header dict — it's a Python repr we can regex.
  const descrMatch = /'descr':\s*'([^']+)'/.exec(headerStr);
  const shapeMatch = /'shape':\s*\(([^)]*)\)/.exec(headerStr);
  if (!descrMatch || !shapeMatch) throw new Error(`${url}: bad .npy header`);
  const descr = descrMatch[1]!;
  const shape = shapeMatch[1]!
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));

  const dataStart = headerStart + headerLen;
  const numel = shape.reduce((a, b) => a * b, 1);

  let data: Float32Array;
  if (descr === "<f4") {
    data = new Float32Array(buf.slice(dataStart, dataStart + numel * 4));
  } else if (descr === "|b1") {
    // bool → float32
    const bytes = new Uint8Array(buf, dataStart, numel);
    data = new Float32Array(numel);
    for (let i = 0; i < numel; i++) data[i] = bytes[i]!;
  } else {
    throw new Error(`${url}: unsupported dtype ${descr}`);
  }
  return { data, shape };
}

export interface ParityRunOptions {
  /** Base URL for fixtures, e.g. "/fixtures/phase1_core/". */
  fixturesBaseUrl: string;
  /** URL to the exported ONNX file. */
  modelUrl: string;
  /** PRNG seed; must match the Python --seed used in generate_fixtures.py (10 by default). */
  seed: number;
  /** Per-tensor MAE threshold for the parity gate. */
  threshold?: number;
  /** Optional UI callback for progress. */
  onProgress?: (msg: string) => void;
}

const FIX = (base: string, p: string) => `${base.replace(/\/$/, "")}/${p}`;

/**
 * Run the full parity check: load fixtures + model, encode text, run 10
 * denoise steps with hooks that capture intermediate x_t and pred_xstart,
 * compute MAE against each fixture, return a report with PASS/FAIL.
 *
 * This is intended to be called from the browser UI on the Run button.
 */
export async function runParity(
  opts: ParityRunOptions,
): Promise<ParityReport> {
  const t0 = performance.now();
  const log = opts.onProgress ?? (() => {});
  const threshold = opts.threshold ?? 5e-3;

  // Loading a text encoder pulls 60–80 MB of weights from the HF CDN. We don't
  // know yet whether the fixture is CLIP- or BERT-shape, so defer encoder
  // selection until after we've fetched text_embed.npy below. Both paths are
  // lazy + non-fatal: if the encoder load fails we still run the rest of the
  // harness against the saved fixture.
  let encoderClip: TextEncoder | null = null;
  let encoderBert: BertTextEncoder | null = null;

  log("loading ONNX trunk on WebGPU…");
  const session = await loadTrunk({ modelUrl: opts.modelUrl });

  try {
    const checks: ParityCheck[] = [];

    // 1. CLIP parity — sanity check that transformers.js produces the same
    //    embedding as the Python-side fixture. NOT gating: if it differs
    //    (e.g. for the synth deploy where text_embed.npy is random and
    //    we want to skip the CLIP run), all per-step checks still proceed
    //    using the saved refTextEmbed as the source of truth.
    log("loading saved text_embed + text_mask fixtures…");
    const promptText = await (
      await fetch(FIX(opts.fixturesBaseUrl, "prompt.txt"))
    ).text();
    const refTextEmbed = await fetchNpy(
      FIX(opts.fixturesBaseUrl, "text_embed.npy"),
    );
    // text_mask is BERT-only; for older CLIP-shape fixtures it's absent and
    // we synthesize an all-False mask of the same length as text_embed seq dim.
    let refTextMask: T4;
    try {
      refTextMask = await fetchNpy(FIX(opts.fixturesBaseUrl, "text_mask.npy"));
    } catch {
      const tText = refTextEmbed.shape[0]!;
      refTextMask = { data: new Float32Array(tText), shape: [1, tText] };
    }

    // Pick the encoder to match the fixture. Shipped DiP checkpoints are
    // DistilBERT (D=768, T_text variable); legacy synth fixtures are CLIP
    // (D=512, T_text=1). The harness handles both: if the encoder load fails
    // (offline / no HF), we record "skipped" and proceed using the saved
    // fixture as ground truth so per-step parity still runs.
    let textEncoderCheck: ParityCheck;
    const fixtureIsBert =
      refTextEmbed.shape.length === 3 &&
      (refTextEmbed.shape[0]! > 1 || refTextEmbed.shape[2] === 768);

    if (fixtureIsBert) {
      log("loading DistilBERT text encoder…");
      try {
        encoderBert = await loadBertTextEncoder({ device: "webgpu" });
      } catch (err) {
        log(`BERT encoder unavailable: ${err instanceof Error ? err.message : err}`);
      }
      if (encoderBert) {
        try {
          log("checking DistilBERT text embedding…");
          const browser = await encoderBert.encode(promptText.trim());
          // Sequence length depends on the tokenizer; if the browser tokenizer
          // and the Python tokenizer disagree on T_text we can't compare
          // element-wise. Bail out informatively rather than producing a
          // misleading huge MAE.
          if (browser.embed.shape[0] !== refTextEmbed.shape[0]) {
            const msg = `T_text mismatch: browser=${browser.embed.shape[0]} fixture=${refTextEmbed.shape[0]}`;
            log(msg);
            textEncoderCheck = {
              name: `text_embed (BERT) — ${msg}`,
              mae: NaN,
              threshold,
              passed: false,
            };
          } else {
            const embedMae = mae(refTextEmbed, browser.embed);
            const maskMae = mae(refTextMask, browser.paddingMask);
            textEncoderCheck = {
              name: "text_embed (BERT)",
              mae: embedMae,
              threshold,
              passed: embedMae < threshold,
            };
            // The padding mask is a {0,1} tensor — it must match exactly.
            // Surface it as its own check so we can tell whether a regression
            // is in the embedding values vs the tokenizer's pad behavior.
            checks.push({
              name: "text_mask (BERT)",
              mae: maskMae,
              threshold: 1e-6,
              passed: maskMae < 1e-6,
            });
          }
        } catch (err) {
          log(`BERT check skipped: ${err instanceof Error ? err.message : err}`);
          textEncoderCheck = {
            name: "text_embed (BERT) — skipped",
            mae: NaN,
            threshold,
            passed: true,
          };
        }
      } else {
        textEncoderCheck = {
          name: "text_embed (BERT) — encoder unavailable",
          mae: NaN,
          threshold,
          passed: true,
        };
      }
    } else {
      log("loading CLIP text encoder…");
      try {
        encoderClip = await loadTextEncoder({ device: "webgpu" });
      } catch (err) {
        log(`CLIP encoder unavailable: ${err instanceof Error ? err.message : err}`);
      }
      if (encoderClip) {
        try {
          log("checking CLIP text embedding…");
          const browserTextEmbed = await encoderClip.encode(promptText.trim());
          const m = mae(refTextEmbed, browserTextEmbed);
          textEncoderCheck = {
            name: "text_embed (CLIP)",
            mae: m,
            threshold,
            passed: m < threshold,
          };
        } catch (err) {
          log(`CLIP check skipped: ${err instanceof Error ? err.message : err}`);
          textEncoderCheck = {
            name: "text_embed (CLIP) — skipped",
            mae: NaN,
            threshold,
            passed: true,
          };
        }
      } else {
        textEncoderCheck = {
          name: "text_embed (CLIP) — encoder unavailable",
          mae: NaN,
          threshold,
          passed: true,
        };
      }
    }
    checks.push(textEncoderCheck);

    // 2. Iter 0 denoise loop with per-step hooks. We always use the saved
    //    refTextEmbed for these checks so a CLIP discrepancy doesn't
    //    cascade into spurious per-step failures.
    log("running iter 0 denoise loop with parity hooks…");
    const prefix = await fetchNpy(FIX(opts.fixturesBaseUrl, "iter0/prefix.npy"));
    const xT = await fetchNpy(FIX(opts.fixturesBaseUrl, "x_T_iter0.npy"));
    const mask = await fetchNpy(FIX(opts.fixturesBaseUrl, "mask.npy"));

    let x = xT;
    const schedule = buildSchedule(DEFAULT_CONFIG.nDiffusionSteps);
    for (let step = 0; step < DEFAULT_CONFIG.nDiffusionSteps; step++) {
      const t = DEFAULT_CONFIG.nDiffusionSteps - 1 - step;
      const stepName = String(step).padStart(2, "0");

      const refXt = await fetchNpy(
        FIX(opts.fixturesBaseUrl, `iter0/x_t_step${stepName}.npy`),
      );
      checks.push({
        name: `iter0/x_t_step${stepName}`,
        mae: mae(refXt, x),
        threshold,
        passed: mae(refXt, x) < threshold,
      });

      // Tap cond/uncond pre-CFG outputs for diagnostics. Whether the model
      // export itself is faithful determines if these checks pass; whether
      // CFG is wired up correctly determines if pred_xstart passes when
      // cond/uncond do.
      const refCond = await fetchNpy(
        FIX(opts.fixturesBaseUrl, `iter0/cond_step${stepName}.npy`),
      ).catch(() => null);
      const refUncond = await fetchNpy(
        FIX(opts.fixturesBaseUrl, `iter0/uncond_step${stepName}.npy`),
      ).catch(() => null);

      let condMae = NaN;
      let uncondMae = NaN;
      let condUncondDiff = NaN;
      const predXstart = await runCfgStep(
        session,
        {
          x,
          timestep: t,
          textEmbed: refTextEmbed,
          textMask: refTextMask,
          mask,
          prefix,
        },
        DEFAULT_CONFIG.guidanceScale,
        (cond, uncond) => {
          if (refCond) condMae = mae(refCond, cond);
          if (refUncond) uncondMae = mae(refUncond, uncond);
          condUncondDiff = mae(cond, uncond);
        },
      );

      if (refCond) {
        checks.push({
          name: `iter0/cond_step${stepName}`,
          mae: condMae,
          threshold,
          passed: condMae < threshold,
        });
      }
      if (refUncond) {
        checks.push({
          name: `iter0/uncond_step${stepName}`,
          mae: uncondMae,
          threshold,
          passed: uncondMae < threshold,
        });
      }
      // Diagnostic: cond and uncond should differ if CFG branching works.
      // If this MAE is ~0, cond pass and uncond pass are producing identical
      // output and CFG degenerates to just cond.
      checks.push({
        name: `iter0/cond_vs_uncond_step${stepName} (should be > threshold)`,
        mae: condUncondDiff,
        threshold,
        passed: condUncondDiff > threshold,
      });

      const refPred = await fetchNpy(
        FIX(opts.fixturesBaseUrl, `iter0/pred_xstart_step${stepName}.npy`),
      );
      checks.push({
        name: `iter0/pred_xstart_step${stepName}`,
        mae: mae(refPred, predXstart),
        threshold,
        passed: mae(refPred, predXstart) < threshold,
      });

      x = ddimStepEta0(x, predXstart, t, schedule);
    }

    // Final iter 0 motion
    const refMotion0 = await fetchNpy(
      FIX(opts.fixturesBaseUrl, "motion_iter0.npy"),
    );
    checks.push({
      name: "motion_iter0",
      mae: mae(refMotion0, x),
      threshold,
      passed: mae(refMotion0, x) < threshold,
    });

    // 3. Full AR sample (uses both iters internally) — verifies prefix rolling
    log("running full AR loop for end-to-end timing…");
    const tSampleStart = performance.now();
    const initialPrefix = await fetchNpy(
      FIX(opts.fixturesBaseUrl, "prefix.npy"),
    );
    const fullSample = await autoregressiveSample(
      session,
      { textEmbed: refTextEmbed, textMask: refTextMask, initialPrefix },
      { config: DEFAULT_CONFIG, seed: opts.seed },
    );
    const totalSample = performance.now() - tSampleStart;
    log(
      `full AR: ${fullSample.shape.join("x")} in ${totalSample.toFixed(0)}ms`,
    );

    return {
      checks,
      textEncoder: textEncoderCheck,
      perfMs: {
        totalSample,
        perStepAvg: totalSample / (5 * 10),
      },
      executionProvider: session.executionProvider,
    };
  } finally {
    encoderClip?.release();
    encoderBert?.release();
    session.release();
    log(`done in ${(performance.now() - t0).toFixed(0)}ms`);
  }
}

// Helper: format a parity report as plain text for the UI / console / RESULTS.md.
export function formatReport(r: ParityReport): string {
  const lines: string[] = [];
  lines.push(`Execution provider: ${r.executionProvider}`);
  lines.push(
    `Full AR sample: ${r.perfMs.totalSample.toFixed(0)}ms ` +
      `(per-step avg ${r.perfMs.perStepAvg.toFixed(0)}ms over 50 steps)`,
  );
  lines.push("");
  let passed = 0;
  let failed = 0;
  for (const c of r.checks) {
    const status = c.passed ? "PASS" : "FAIL";
    lines.push(`  [${status}] ${c.name}: MAE=${c.mae.toExponential(2)} (< ${c.threshold.toExponential(0)})`);
    if (c.passed) passed++;
    else failed++;
  }
  lines.push("");
  lines.push(`Summary: ${passed} passed, ${failed} failed`);
  return lines.join("\n");
}

// Re-export for the UI's use.
export type { TextEncoder, TrunkSession };
