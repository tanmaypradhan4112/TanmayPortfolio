---
title: "Build Personal Portfolio using Astro"
description: "Explore how to build a stunning and personalized portfolio website using Astro, a cutting-edge static site builder and deploy it on Netlify."
pubDate: "July 31, 2023"
heroImage: "https://astro.build/_astro/astro-netlify-social.2cd5322d.webp"
badge: "Latest"
---
# Getting Started
###  What is Astro?

The speed-focused all-in-one web framework is called Astro. Utilising your preferred UI components and frameworks, you may publish your content wherever. Astro optimizes your website with the help of [Zero-JS frontend architechture](https://docs.astro.build/en/concepts/islands/). 
Link 🔗 : [Astro](https://astro.build/)

> ## Astro is free, open source software

### What is Netlify?
It is all-in-one platform helps you combine your favorite tools and APIs to build the fastest sites, stores, and apps for the composable web. Use any frontend framework to build, preview, and deploy to our global network from Git.
Link 🔗 : [Netlify](https://www.netlify.com/)

## Prerequisites

 - Basic knowledge of JS and Astro
 - About Netlify
 - Command Line Interface (Terminal)
 - Node.js - v16 or higher
 - Text Editor - VS Codium

## Let's Astro Build a Portfolio Website 
### Step1: Chose a Static Template in Astro

Astro has many template that you can choose and are Content Focused. We have the following category you can work on like:

 - Blog
 - Marketing
 - Agencies 
 - E-Commerce
 - Portfolio

For our purpose, we will choose Portfolio catergory.
Link to Portfolio Catergory: [Portfolio Theme](https://astro.build/themes/?categories%5B%5D=portfolio)
In the portfolio category, I will be selecting [Astrofy | Personal Porfolio Website Template](https://astro.build/themes/details/astrofy/). It is build with Astro and TailwindCSS along with Blog, CV, Project Section, Store and RSS Feed.
![enter image description here](https://astro.build/_astro/astrofy-hero@2x.69ec4be4.webp)
You can choose any template according to your required skills and tech stack.

### Step2: Clone the template repository
In the particular template category you have  chosen, click on Get Started. You will be redirected to the respective Github repository. You can explore the template file structure and clone the repository.

Link to the template repository: [Astrofy](https://github.com/manuelernestog/astrofy)

Before we clone the project, open terminal and change to home directory

**Guide to clone the repository:**

    git clone https://github.com/manuelernestog/astrofy.git
    cd <filename>
    npm install
    npm run dev
**Explanation**
So we first clone the repository in our local system. We change the directory to that respective file. The third line of code install all the necessary dependencies that was required to build the template and also you don't end up in having lot of error. At last you run "npm run dev" to run.

**File Struture**
Taken from Astrofy public template from their Github repo for you clear understanding of overall file structure.

    ├── src/
    │   ├── components/
    │   │   ├── cv/
    │   │   │   ├── TimeLine
    │   │   ├── BaseHead.astro
    │   │   ├── Card.astro
    │   │   ├── Footer.astro
    │   │   ├── Header.astro
    │   │   └── HorizontalCard.astro
    │   │   └── SideBar.astro
    │   │   └── SideBarMenu.astro
    │   │   └── SideBarFooter.astro
    │   ├── content/
    │   │   ├── blog/
    │   │   │   ├── post1.md
    │   │   │   ├── post2.md
    │   │   │   └── post3.md
    │   │   ├── store/
    │   │   │   ├── item1.md
    │   │   │   ├── item2.md
    │   ├── layouts/
    │   │   └── BaseLayout.astro
    │   │   └── PostLayout.astro
    │   └── pages/
    │   │   ├── blog/
    │   │   │   ├── [...page].astro
    │   │   │   ├── [slug].astro
    │   │   └── cv.astro
    │   │   └── index.astro
    │   │   └── projects.astro
    │   │   └── rss.xml.js
    │   └── styles/
    │       └── global.css
    ├── public/
    │   ├── favicon.svg
    │   └── social-image.png
    │   └── sprofile.jpg
    │   └── social_img.webp
    ├── astro.config.mjs
    ├── tailwind.config.cjs
    ├── package.json
    └── tsconfig.json

### Step3 : Add you Magic ✨
Customize the way you want and use the CSS magic to attract viewer and make them sit for minutes. I chose it to be minimalistic and default style of template. Probably I will tweak a little CSS animation or maybe introduce advanced JS modules.

### Step4 : Deploy using Netlify 🌐
 I see you made beautiful or maybe magnificent website there. I'm amazed by the creative mind there.
Let's deploy your website using Netlify.

**Step 1**:  
Add the Netlify adapter to enable SSR in your Astro project. This wil create some changes in your *astro.config.mjs* file

    npx astro add netlify

**Step 2**: 

 1. We now deploy the website in Netlify. Open [Netlify Dashboard](https://app.netlify.com/)
 2. Click on add new site
 3. Choose Import an existing project. 
 4. Choose Git provider as it automatically prefill your configuration to deploy the website.
 5. After making sure you have review all the settings, click on deploy.

**Step 3** (Optional):
You can also create *netlify.toml* in your project directory. It configure your build command and publish directory, as well as other project settings including environment variables and redirects. Netlify will read this file and automatically configure your deployment.

      [build]
      command = "npm run build"
      publish = "dist"

**Note**
If you prefer to install the adapter manually instead, complete the following two steps:

Install the `@astrojs/netlify` adapter to your project’s dependencies using your preferred package manager. If you’re using npm or aren’t sure, run this in the terminal:

    npm install @astrojs/netlify

Add two new lines to your `astro.config.mjs` project configuration file.
```
import { defineConfig } from 'astro/config';
+ import netlify from '@astrojs/netlify/functions';

export default defineConfig({
 + output: 'server',
 + adapter: netlify(),
});
```

If you face any error while deploying the website or even routing the page. I suggest you to remove  the below two lines from `astro.config.mjs` 
```
 + output: 'server',
 + adapter: netlify(),
```
**Congratulation! You deployed your first Static Personal Portfolio Website.**

Feel free to contribute and mail me at `tanmaypradhan4112@gmail.com`

Astro | Netlify | NPM | JS | Markdown | Portfolio website
