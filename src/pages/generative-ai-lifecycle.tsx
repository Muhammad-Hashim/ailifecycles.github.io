import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './generative-ai-lifecycle.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function GenerativeAiLifecycle() {
  return (
    <Layout
      title="Generative AI Lifecycle"
      description="Complete guide to the Generative AI Lifecycle from model selection to deployment."
    >
      <main className={styles.main}>
        <div className="container">
          <div className={styles.header}>
            <div className={styles.robotContainer}>
              <img 
                src={useBaseUrl('/img/robots/genai-robot.svg')} 
                alt="Generative AI Robot"
                className={styles.robotImage}
              />
            </div>
            <h1>Generative AI Lifecycle</h1>
            <p>
              A comprehensive guide to developing and deploying generative AI systems.
              Learn about LLMs, diffusion models, and multimodal generation.
            </p>
          </div>

          <div className={styles.content}>
            <div className={styles.overviewCard}>
              <h2>Overview</h2>
              <p>
                Get started with the fundamentals of generative AI lifecycle including
                model selection, fine-tuning techniques, evaluation methods,
                and responsible deployment practices.
              </p>
              <Link to="/docs/generative-ai-lifecycle/overview" className={styles.link}>
                Read the Overview →
              </Link>
            </div>

            <div className={styles.comingSoon}>
              <h3>Coming Soon</h3>
              <p>
                Detailed guides for generative AI topics including:
              </p>
              <ul>
                <li>Large Language Models (LLMs)</li>
                <li>Diffusion Models & Image Generation</li>
                <li>Fine-tuning & Prompt Engineering</li>
                <li>Multimodal AI Systems</li>
                <li>Evaluation & Safety Measures</li>
                <li>Production Deployment</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
