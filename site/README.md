# aglaia.bibli.cc

Marketing landing page + documentation for Aglaïa, in one Astro project.

- **Landing** — bespoke parallax page at `/` (`src/pages/index.astro`,
  smooth scroll via Lenis, reveal-on-scroll via IntersectionObserver).
- **Docs** — [Starlight](https://starlight.astro.build) at `/docs/*`.
  Reference pages are **generated** from the repo's top-level `docs/*.md`
  by `scripts/sync-docs.mjs` (runs automatically before `dev`/`build`),
  so prose lives in one place. Hand-written pages (`Overview`, `Install`)
  live in `src/content/docs/docs/`.

## Develop

```bash
cd site
npm install
npm run dev       # syncs docs, then astro dev
```

## Build

```bash
npm run build     # → dist/   (deployed to Cloudflare Pages)
```

Deployment is automated by `.github/workflows/site.yml` on push to `main`.
Cloudflare Pages build settings: **root** `site`, **build command**
`npm run build`, **output** `dist`.

## Editing docs

Edit the markdown in the repo's `../docs/`. The Starlight sidebar
(`astro.config.mjs`) autogenerates the Reference section from
`docs/reference/`. Add a page → it appears after the next build.
