// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';

// Landing page is a bespoke Astro page at `/` (src/pages/index.astro).
// Starlight owns everything under `/docs/*` — achieved by nesting the
// docs content one folder deep (src/content/docs/docs/…), so no Starlight
// route collides with the custom homepage.
export default defineConfig({
  site: 'https://aglaia.bibli.cc',
  integrations: [
    // Render ```mermaid fenced blocks (client-side; no headless browser at
    // build). Must precede Starlight so it processes the markdown first.
    // `autoTheme` follows Starlight's light/dark toggle.
    mermaid({ theme: 'neutral', autoTheme: true }),
    starlight({
      title: 'Aglaïa',
      description: 'Turn a webcam and a stack of pages into clean, searchable PDFs.',
      social: {
        github: 'https://github.com/yb85/aglaia',
      },
      customCss: ['./src/styles/docs.css'],
      favicon: '/favicon.ico',
      head: [
        { tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/apple-touch-icon.png' } },
      ],
      // Send the masthead title/logo back to the marketing homepage.
      logo: { src: './src/assets/aglaia-dark.png', alt: 'Aglaïa', replacesTitle: true },
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
        ThemeSelect: './src/components/ThemeSelect.astro',
      },
      // Capability-grouped IA in the HashiCorp/Terraform spirit:
      // Introduction → Concepts → Guides → Reference.
      sidebar: [
        {
          label: 'Introduction',
          items: [
            { label: 'Overview', slug: 'docs' },
            { label: 'Install', slug: 'docs/install' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'How Aglaïa works', slug: 'docs/concepts/workflow' },
            { label: 'Import', slug: 'docs/concepts/import' },
            { label: 'Pipeline processing', slug: 'docs/concepts/pipeline-processing' },
            { label: 'OCR engines', slug: 'docs/concepts/ocr-engines' },
            { label: 'Export', slug: 'docs/concepts/export' },
            { label: 'The .AGL project file', slug: 'docs/concepts/agl-project-file' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Calibrate the camera', slug: 'docs/reference/calibration' },
            { label: 'Export to Markdown', slug: 'docs/reference/markdown_export' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Architecture', slug: 'docs/reference/architecture' },
            { label: 'The capture GUI', slug: 'docs/reference/gui' },
            { label: 'Pipeline', slug: 'docs/reference/pipeline' },
            { label: 'Processors', slug: 'docs/reference/processors' },
            { label: 'Configuration', slug: 'docs/reference/configuration' },
            { label: 'OCR engines', slug: 'docs/reference/ocr' },
            { label: 'Export', slug: 'docs/reference/export' },
            { label: 'ImageBuffer', slug: 'docs/reference/imagebuffer' },
            { label: 'Storage', slug: 'docs/reference/storage' },
            { label: 'APP_DATA', slug: 'docs/reference/app_data' },
            { label: 'Theme', slug: 'docs/reference/theme' },
            { label: 'Internationalization', slug: 'docs/reference/i18n' },
          ],
        },
      ],
    }),
  ],
});
