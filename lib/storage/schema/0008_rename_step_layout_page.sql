-- Aglaïa: rewrite pipeline step names layout(s)_* → page(s)_* (added 2026-06-19)
--
-- Companion to the domain layout→page rename. The pipeline YAMLs renamed the
-- split steps (layouts_2ppf → pages_2ppf, layout_1ppf → page_1ppf, …), and
-- nodes.step_name stores the per-step instance name ("03_layouts_2ppf"). New
-- projects already write the page_* names; this fixes the names recorded in
-- EXISTING projects so the gallery/table stage axis keeps matching their nodes
-- without a reprocess.
--
-- REPLACE the longer "layouts_" form first so the shorter "layout_" pass can't
-- mangle it. Non-matching names (e.g. "raw", "00_dpi_normalize") are untouched;
-- NULL step_name stays NULL. Idempotent in effect (no layout_* left after).
-- Runs exactly once per DB via the ensure_schema ledger.

UPDATE nodes SET step_name = REPLACE(step_name, 'layouts_', 'pages_')
  WHERE step_name LIKE '%layouts\_%' ESCAPE '\';

UPDATE nodes SET step_name = REPLACE(step_name, 'layout_', 'page_')
  WHERE step_name LIKE '%layout\_%' ESCAPE '\';
