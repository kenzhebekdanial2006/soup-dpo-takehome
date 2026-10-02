"""Generate an honest, two-page PDF and a concise Markdown report from existing evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common import utc_now, write_json


def read(directory, filename):
    path = directory / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_report(directory):
    status = read(directory, "status.json") or {"status": "NOT RUN", "completed": False}
    budget = read(directory, "memory_budget.json")
    trained = read(directory, "training_result.json")
    verified = read(directory, "verification.json")
    parity = read(directory, "gradient_parity.json")
    evaluation = read(directory, "evaluation.json")
    soup_ship = read(directory, "soup_ship.json")
    negative = read(directory / "control", "verification.json")
    gates = {
        "t4_run_completed": status.get("completed", False),
        "reference_stable": bool(trained and trained.get("reference_stable")),
        "adapter_changed_and_active": bool(verified and verified.get("passed")),
        "gradient_parity": bool(parity and parity.get("passed")),
        "no_op_control_rejected": bool(negative and not negative.get("passed")
                                      and not negative["tensor_delta"]["passed"]
                                      and not negative["functional_change"]["passed"]),
        "real_domain_validation": False,
        "broad_forgetting_validation": False,
    }
    # Synthetic tickets + four forced-choice questions NEVER authorize deployment.
    verdict = "DON'T SHIP"
    evidence_label = str(directory).replace("\\", "/")
    report = []
    report.append(("Decision and run", f"{verdict}. Evidence directory: {evidence_label}; run status: {status['status']}. "
                   "Do not deploy this adapter. Synthetic heldout results cannot establish Russian customer-support quality. "
                   "Missing evidence is UNRUN, never a pass."))
    report.append(("Recipe and data", "Qwen/Qwen2.5-1.5B-Instruct, immutable snapshot SHA recorded before training; Soup 0.75.2. "
                   "T4/fp16; unquantized frozen base, streaming from RAM with 2 buffers, q_proj/v_proj LoRA r=8 alpha=16, "
                   "AdamW, lr=5e-5, 1 epoch, pair batch 1, accumulation 4, max_length 512, seed 20260930. "
                   "500 template-defined fictional tickets authored with Codex: 400 train/100 heldout, grouped by scenario, 10 heldout groups. "
                   "Shared policy/response templates remain a shortcut risk; no real tickets or independent labels."))
    if budget:
        memory = f"Pre-run planned upper peak: {budget['planned_peak_upper_GiB']:.3f} GiB. "
        memory += f"Base {budget['base_parameters']:,} parameters ({budget['base_resident_fp16_bytes']/1024**3:.3f} GiB resident alternative); "
        memory += f"host decoder store {budget['host_decoder_store_fp16_bytes']/1024**3:.3f} GiB; LoRA {budget['lora_parameters']:,} parameters. "
        c = budget["components"]
        memory += f"Two layer buffers {c['decoder_pool']/1024**3:.3f}, tied vocabulary {c['embedding_or_large_vocab_slot']/1024**3:.3f}, "
        memory += f"adapter+grad+Adam {(c['adapter_weights_fp32']+c['adapter_gradients_fp32']+c['adam_m_and_v_fp32'])/1024**3:.3f}, "
        memory += f"activations {(c['checkpoint_boundary_activations']+c['one_recomputed_layer_intermediates'])/1024**3:.3f}, "
        memory += f"policy logits+backward {(c['policy_logits_fp16']+c['policy_logsoftmax_and_backward_fp32_planning_allowance'])/1024**3:.3f}, "
        memory += "context/workspace/slack 1.000 GiB. Reference has no second base; sequential logits use max(policy, reference), not their sum. "
        memory += f"Frozen reference adapter allowance {c['frozen_reference_adapter_fp32_allowance']/1024**3:.3f} GiB. "
        if trained:
            actual = trained["training_peak_allocated_bytes"] / 1024**3
            memory += f"Measured training allocated peak {actual:.3f} GiB, reserved {trained['training_peak_reserved_bytes']/1024**3:.3f}; "
            memory += f"gap vs plan {(actual-budget['planned_peak_upper_GiB']):+.3f} GiB. "
        else:
            memory += "Actual peak: NOT MEASURED. Gap: NOT COMPUTABLE. "
        memory += "Plan assumes fully padded length 512 and conservative logit temporaries; real lengths/lifetimes differ. "
        memory += "Reserved includes allocator slack; nvidia-smi adds driver/other processes and 1 s sampling misses short peaks. "
        memory += "Break down the observed gap from raw samples and runtime stats; no attribution to a single cause is claimed."
    else:
        memory = "No model snapshot or T4 training peak measured yet. scripts/memory_budget.py computes weights, two buffers, tied embedding, "
        memory += "FP32 LoRA/grad/Adam, checkpoint boundaries, recomputation, paired logits and sequential no-grad reference BEFORE training. "
        memory += "The pending comparison uses in-process allocated/reserved high-water marks and raw nvidia-smi, not Soup's prediction as measured evidence."
    report.append(("Memory before vs after", memory))
    verification = "Falling loss may reflect sample order/length, reference drift, label bugs or training tensors that never load into the serving model. "
    verification += "Part 2 compares saved tensors to their exact initialized snapshot (not to zero); requires 90% of LoRA B matrices to change, all finite. "
    verification += "Ordinary PEFT reload must round-trip every adapter tensor with no missing-key warnings; identical response-token log-probabilities "
    verification += "must change beyond repeated-base noise and return to base when the adapter is disabled. "
    if verified:
        verification += f"Observed passed={verified['passed']}; changed B fraction={verified['tensor_delta'].get('changed_B_fraction', 0):.3f}; "
        verification += f"functional delta={verified['functional_change']['max_token_logp_delta']:.6g}, noise={verified['functional_change']['baseline_repeat_noise']:.6g}. "
    else:
        verification += "Real trained-adapter output: UNRUN. "
    verification += f"T4 streamed-vs-resident DPO backward: {parity['passed'] if parity else 'UNRUN'}; lr=0 control: "
    verification += f"{'correctly rejected' if gates['no_op_control_rejected'] else 'UNRUN or failed'}. "
    verification += "Tensor/output checks miss wrong gradients and bad labels; parity covers three batches at this stack/model/shape, not all steps."
    report.append(("Prove training", verification))
    failures = [
        ("Inert saved adapter", "Round-trip keys, tensor deltas, enabled/disabled probes; data doctor/preflight cannot prove it. Historical .inner. defect is fixed upstream; our artifact still needs checking."),
        ("Wrong streamed gradients", "Compare all LoRA gradients to resident DPO with deliberately nonzero B. Forward/loss equality is insufficient. NF4 aliasing defect does not apply to our quantization=none; no NF4 correctness claim."),
        ("Reference drift / extra model", "Capture ref logps before/after; record ref instance/ref-adapter and runtime buffers. Same frozen base must be reused. Soup's fit gate does not establish a stationary reference."),
        ("Masking, EOS, truncation", "Run doctor on chat projection, lint on actual DPO pairs; audit all 500 tokenized pairs and inspect prepared TRL IDs. Chat doctor does not prove the DPO trainer's masking."),
        ("Duplicates / length shortcuts", "Group split, exact-prompt checks, sum and mean margins; lint catches heuristics. Shared synthetic templates and label validity remain unresolved, even if lint is OK."),
        ("Precision / skipped steps / OOM", "Require T4 fp16, materialized LoRA, finite nonzero gradients, step count and saved-state change. Raw CUDA peaks plus nvidia-smi. Preflight sizes fit, not learning; measured SFT probe is not DPO evidence."),
        ("Misleading SHIP", "Run Soup offline on measured preference accuracy and explicitly named four-question forced-choice sanity set. A numeric SHIP can miss label correctness, confidence, real-domain generalization and adapter provenance."),
    ]
    report.append(("Silent failures: check and Soup coverage", " ".join(f"{name}: {details}" for name, details in failures)))
    outcome = f"Soup ship output: {'recorded in soup_ship.json' if soup_ship else 'UNRUN'}. "
    if evaluation:
        outcome += f"Heldout paired accuracy delta {evaluation['paired_accuracy_delta']:+.3f}; group-bootstrap 95% interval {evaluation['cluster_bootstrap_95_ci']}. "
    outcome += "Required before SHIP: complete and pass the T4 evidence chain; obtain independently annotated real Russian tickets with "
    outcome += "scenario separation; manually inspect generated replies and safety errors; run broader Russian/general regression tests and seeds; "
    outcome += "resolve every MAJOR diagnostic. Package this run's weights with its snapshot/config/data hashes. No failed run is removed."
    report.append(("Verdict and required changes", outcome))
    report.append(("Surprise and remaining concern", "The most surprising source finding was that identical forward/loss values can coexist with "
                   "wrong backward gradients, and that a healthy training adapter can become inert when saved with wrapper-specific keys. "
                   "This is a source-review observation, not an experience claimed from an unrun experiment. My main concern is whether the "
                   "synthetic task rewards reusable phrasing instead of useful support behavior; neither loss nor four general probes can settle that."))
    report.append(("AI assistance", "Codex prepared the dataset templates, implementation, documentation and source analysis. Accepted suggestions: "
                   "immutable base snapshot, actual initialization comparison, ordinary PEFT reload, lr=0 negative control and backward parity. "
                   "Corrected by checking source: doctor rejects DPO pairs; use lint plus chat projection; T4 precision is fp16; never report pending "
                   "GPU checks as passed. Local tests and T4 evidence must be distinguished. The applicant has not independently reviewed or run this "
                   "work unless their review is separately recorded; no applicant verification is invented."))
    write_json(directory / "verdict.json", {"timestamp": utc_now(), "verdict": verdict, "gates": gates})
    return report


def main():
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    directory = Path(args.evidence)
    directory.mkdir(parents=True, exist_ok=True)
    report = create_report(directory)
    output = Path("reports")
    output.mkdir(exist_ok=True)
    md = "# Soup DPO take-home — DON'T SHIP\n\n" + "\n\n".join(f"**{title}.** {body}" for title, body in report) + "\n"
    (output / "report.md").write_text(md, encoding="utf-8")
    style = ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=11.2, spaceAfter=6)
    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#16324f"), spaceAfter=10)
    story = [Paragraph("Soup DPO take-home | DON'T SHIP", heading)]
    for index, (title, body) in enumerate(report):
        if index == 4:
            story += [PageBreak(), Paragraph("Evidence, failure modes and limits", heading)]
        story += [Paragraph(f"<b>{escape(title)}.</b> {escape(body)}", style), Spacer(1, 1 * mm)]
    pdf_path = output / "report.pdf"
    document = SimpleDocTemplate(str(pdf_path), pagesize=(210 * mm, 297 * mm),
                                 leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(194 * mm, 9 * mm, f"{doc.page}")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    # Fail rather than quietly submit a report exceeding the requested page limit.
    from pypdf import PdfReader

    pages = len(PdfReader(str(pdf_path)).pages)
    if pages > 2:
        raise RuntimeError(f"Report exceeded 2-page limit ({pages} pages)")
    print({"report": str(pdf_path), "pages": pages, "evidence": str(directory)})


if __name__ == "__main__":
    main()
