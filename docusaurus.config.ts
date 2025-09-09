import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import type {ThemeConfig} from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'AI Lifecycles',
  tagline: 'Comprehensive AI & Machine Learning Lifecycle Documentation',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'your-org', // Usually your GitHub org/user name.
  projectName: 'ailifecycles', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/your-org/ailifecycles/edit/main/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/your-org/ailifecycles/edit/main/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    metadata: [
      {name: 'keywords', content: 'machine learning, AI, lifecycle, ML pipelines, data science, Python'},
      {name: 'description', content: 'Comprehensive documentation for machine learning lifecycles, from data collection to deployment'},
    ],
    navbar: {
      title: 'AI Lifecycles',
      logo: {
        alt: 'AI Lifecycles Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          to: '/docs/machine-learning-lifecycle/overview',
          label: 'Machine Learning',
          position: 'left',
        },
        {
          to: '/docs/nlp-lifecycle/overview',
          label: 'NLP',
          position: 'left',
        },
        {
          to: '/docs/computer-vision-lifecycle/overview',
          label: 'Computer Vision',
          position: 'left',
        },
        {
          to: '/docs/generative-ai-lifecycle/overview',
          label: 'Generative AI',
          position: 'left',
        },
        {
          to: '/docs/mlops-lifecycle/overview',
          label: 'MLOps',
          position: 'left',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          type: 'docsVersionDropdown',
          position: 'right',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/your-org/ailifecycles',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/intro',
            },
            {
              label: 'ML Lifecycle Overview',
              to: '/docs/machine-learning-lifecycle/overview',
            },
            {
              label: 'Data Preparation',
              to: '/docs/machine-learning-lifecycle/data-collection-preparation',
            },
            {
              label: 'Model Development',
              to: '/docs/machine-learning-lifecycle/model-development',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Learning Paths',
              to: '/docs/machine-learning-lifecycle/roadmaps-learning-paths',
            },
            {
              label: 'Tools & Frameworks',
              to: '/docs/machine-learning-lifecycle/resources',
            },
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/your-org/ailifecycles',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/machine-learning',
            },
            {
              label: 'Discord',
              href: 'https://discord.gg/docusaurus',
            },
            {
              label: 'Twitter',
              href: 'https://twitter.com/yourhandle',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI Lifecycles. Built with ❤️ using Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'sql'],
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    // algolia: {
    //   // The application ID provided by Algolia
    //   appId: 'YOUR_APP_ID',
    //   // Public API key: it is safe to commit it
    //   apiKey: 'YOUR_SEARCH_API_KEY',
    //   indexName: 'ailifecycles',
    //   // Optional: see doc section below
    //   contextualSearch: true,
    //   // Optional: Specify domains where the navigation should occur through window.location instead on history.push. Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
    //   externalUrlRegex: 'external\\.com|domain\\.com',
    //   // Optional: Replace parts of the item URLs from Algolia. Useful when using the same search index for multiple deployments using a different baseUrl. You can use regexp or string in the `from` param. For example: localhost:3000 vs myCompany.com/docs
    //   replaceSearchResultPathname: {
    //     from: '/docs/', // or as RegExp: /\/docs\//
    //     to: '/',
    //   },
    //   // Optional: Algolia search parameters
    //   searchParameters: {},
    //   // Optional: path for search page that enabled by default (`false` to disable it)
    //   searchPagePath: false, // Disable search page to avoid the error
    // },
  } satisfies ThemeConfig,
};

export default config;
