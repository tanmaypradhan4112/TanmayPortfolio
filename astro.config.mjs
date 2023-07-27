import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";

import image from "@astrojs/image";

// https://astro.build/config
import netlify from '@astrojs/netlify/functions';

// https://astro.build/config
export default defineConfig({
  site: 'https://tanmaypradhan.netlify.app/',
  base: '/',
  integrations: [mdx(), sitemap(), tailwind(), image({
    serviceEntryPoint: '@astrojs/image/sharp',
    cacheDir: "./.cache/image",
    logLevel: 'debug'
  })],
  // output: "server",
  // adapter: netlify()
});