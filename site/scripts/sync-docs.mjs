// Generate Starlight reference pages from the repo's `docs/` markdown so
// prose lives in exactly one place. Run automatically before dev/build.
//
// For each `../docs/*.md` it: derives a title from the first H1 (or the
// filename), strips that H1 (Starlight renders the title itself), rewrites
// intra-doc `foo.md` links to extensionless slugs, and writes the result
// under src/content/docs/docs/reference/ with Starlight frontmatter.

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.resolve(here, '..');
const DOCS_SRC = path.resolve(SITE, '..', 'docs');
const OUT = path.resolve(SITE, 'src/content/docs/docs/reference');
// Figures referenced by the docs are copied here and served statically;
// markdown image paths are rewritten to /docs-assets/<original path>.
const ASSETS_OUT = path.resolve(SITE, 'public/docs-assets');

// Internal planning notes — not user-facing reference.
const SKIP = new Set([
  'README.md',
  'sidebar_redesign_plan.md',
  'lessons.md',
  'psyinstaller.md',   // half-generated planning artifact, not reference
  'landing-page-arch.md',  // landing-page copy spec, not a reference doc
  'development.md',    // dev setup — not user-facing public reference
  'distribution.md',   // release/CI internals — not user-facing reference
  'subcommand-cli.md', // internal CLI-redesign plan, not user reference
]);

function titleFrom(md, fallback) {
  const m = md.match(/^\s*#\s+(.+?)\s*$/m);
  return (m ? m[1] : fallback).replace(/`/g, '');
}

function stripFirstH1(md) {
  return md.replace(/^\s*#\s+.+?\s*$/m, '').replace(/^\s+/, '');
}

// `[x](architecture.md)` / `(pipeline.md#step)` → extensionless slug so
// links resolve within /docs/reference/.
function rewriteLinks(md) {
  return md.replace(/\]\((\.\/)?([\w./-]+?)\.md(#[\w-]+)?\)/g,
    (_full, _dot, name, hash = '') => {
      const base = name.split('/').pop();
      return `](${base}${hash})`;
    });
}

// `![alt](figures/x.png)` → `![alt](/docs-assets/figures/x.png)` for any
// relative image path (leaves http(s) and already-absolute paths alone).
function rewriteImages(md) {
  return md.replace(/(!\[[^\]]*\]\()(\.\/)?([\w./-]+?\.(?:png|jpe?g|gif|svg|webp))(\))/gi,
    (_full, open, _dot, rel, close) => `${open}/docs-assets/${rel}${close}`);
}

function escapeYaml(s) {
  return s.replace(/"/g, '\\"');
}

const IMG_EXT = /\.(png|jpe?g|gif|svg|webp)$/i;

async function copyAssets(dir, rel = '') {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const abs = path.join(dir, e.name);
    const r = path.join(rel, e.name);
    if (e.isDirectory()) {
      await copyAssets(abs, r);
    } else if (IMG_EXT.test(e.name)) {
      const dest = path.join(ASSETS_OUT, r);
      await fs.mkdir(path.dirname(dest), { recursive: true });
      await fs.copyFile(abs, dest);
    }
  }
}

async function main() {
  await fs.rm(OUT, { recursive: true, force: true });
  await fs.mkdir(OUT, { recursive: true });
  await fs.rm(ASSETS_OUT, { recursive: true, force: true });
  await fs.mkdir(ASSETS_OUT, { recursive: true });
  await copyAssets(DOCS_SRC);

  const entries = await fs.readdir(DOCS_SRC, { withFileTypes: true });
  let n = 0;
  for (const e of entries) {
    if (!e.isFile() || !e.name.endsWith('.md')) continue;
    if (SKIP.has(e.name)) continue;
    const raw = await fs.readFile(path.join(DOCS_SRC, e.name), 'utf8');
    const title = titleFrom(raw, e.name.replace(/\.md$/, ''));
    const body = rewriteImages(rewriteLinks(stripFirstH1(raw)));
    const front = `---\ntitle: "${escapeYaml(title)}"\n---\n\n`;
    await fs.writeFile(path.join(OUT, e.name), front + body);
    n++;
  }
  console.log(`[sync-docs] wrote ${n} reference pages → ${path.relative(SITE, OUT)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
