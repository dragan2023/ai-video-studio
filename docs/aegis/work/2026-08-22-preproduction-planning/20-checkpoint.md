# Todo Checkpoint Draft

- Completed: durable preproduction contract, thick-script evidence retention, deterministic transition/first-frame planning, LLM-backed imported-shot H3 enrichment, visible plan controls and render gate, GPT Image-compatible configuration docs.
- LLM enrichment behavior: the preproduction endpoint now fails closed without a configured Planner LLM; it calls the existing structured Shot Director contract per imported shot, validates complete H3 fields and Beat coverage, preserves creator source/material/runtime data, saves the enriched FilmProject, then computes the visible preproduction plan.
- Frontend behavior: after plan creation it reloads the project so the existing Edit Shot form displays H3 values.
- Evidence: 72 selected tests passed; cd web && npm run build passed.
- Still incomplete: approved-only generate_t2i worker must invoke existing TextToImageProvider, ingest successful images, bind shots, and transition ready/blocked. Current safety behavior remains: plans with generated-image gaps stay generating_assets and render is blocked.
- Next: implement controlled asset-generation worker and its provider/mock failure tests; do not declare objective complete until this closes.
