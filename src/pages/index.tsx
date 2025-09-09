import type {ReactNode} from 'react';
import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function EcosystemShowcase() {
  const popularTools = [
    { name: 'TensorFlow', icon: '🧠', color: '#FF6F00' },
    { name: 'PyTorch', icon: '🔥', color: '#EE4C2C' },
    { name: 'Hugging Face', icon: '🤗', color: '#FF6B6B' },
    { name: 'OpenAI', icon: '🤖', color: '#10A37F' },
    { name: 'Scikit-learn', icon: '🔬', color: '#F7931E' },
    { name: 'MLflow', icon: '🔄', color: '#0194E2' },
    { name: 'Docker', icon: '🐳', color: '#0db7ed' },
    { name: 'Kubernetes', icon: '☸️', color: '#326CE5' },
    { name: 'AWS', icon: '☁️', color: '#FF9900' },
    { name: 'Jupyter', icon: '📓', color: '#F37626' },
    { name: 'OpenCV', icon: '👁️', color: '#5C3EE8' },
    { name: 'LangChain', icon: '⛓️', color: '#FF4B4B' },
    { name: 'Pandas', icon: '🐼', color: '#150458' },
    { name: 'NumPy', icon: '🔢', color: '#013243' },
    { name: 'Streamlit', icon: '🎯', color: '#FF4B4B' },
    { name: 'FastAPI', icon: '🚀', color: '#009688' },
    { name: 'Plotly', icon: '📊', color: '#3F4F75' },
    { name: 'Apache Spark', icon: '⚡', color: '#E25A1C' },
    { name: 'Stable Diffusion', icon: '🎨', color: '#FF6B6B' },
    { name: 'YOLO', icon: '⚡', color: '#00FFFF' }
  ];

  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % popularTools.length);
    }, 2000);
    return () => clearInterval(interval);
  }, [popularTools.length]);

  return (
    <section className={styles.ecosystemSection}>
      <div className="container">
        <div className={styles.ecosystemContent}>
          <Heading as="h2" className={styles.ecosystemTitle}>
            Powered by the best AI ecosystem
          </Heading>
          <p className={styles.ecosystemSubtitle}>
            Comprehensive coverage of 100+ libraries, frameworks, and platforms
          </p>
          <div className={styles.toolsShowcase}>
            <div className={styles.toolsOrbit}>
              {popularTools.map((tool, index) => (
                <div
                  key={tool.name}
                  className={clsx(styles.toolItem, {
                    [styles.toolItemActive]: index === currentIndex
                  })}
                  style={{
                    '--tool-color': tool.color,
                    '--rotation': `${(index / popularTools.length) * 360}deg`
                  } as React.CSSProperties}
                >
                  <div className={styles.toolIcon}>{tool.icon}</div>
                  <div className={styles.toolName}>{tool.name}</div>
                </div>
              ))}
            </div>
            <div className={styles.centerLogo}>
              <div className={styles.logoIcon}>🤖</div>
              <div className={styles.logoText}>AI Lifecycles</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroLabel}>
            Documentation Platform
          </div>
          <Heading as="h1" className={styles.heroTitle}>
            The platform for comprehensive AI lifecycles.
          </Heading>
          <p className={styles.heroSubtitle}>
            Master end-to-end AI development with expert guidance, best practices, and production-ready workflows for every stage of your AI journey.
          </p>
          <div className={styles.heroButtons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/machine-learning-lifecycle/overview">
              Get Started
            </Link>
            <Link
              className="button button--outline button--lg"
              to="/docs/intro">
              View Documentation
            </Link>
          </div>
          <div className={styles.trustedBy}>
            <p className={styles.trustedByText}>Trusted by AI practitioners worldwide</p>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Comprehensive guides for AI and machine learning lifecycles including ML, NLP, computer vision, and generative AI.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <EcosystemShowcase />
      </main>
    </Layout>
  );
}
